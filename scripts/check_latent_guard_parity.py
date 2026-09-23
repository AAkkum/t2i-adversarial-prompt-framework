from __future__ import annotations

import argparse
import gc
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from transformers import CLIPModel, CLIPTokenizer

from t2i_framework.defenses.latent_guard_lite import _LatentGuardLiteScorer


class OfficialEmbeddingMappingLayer(nn.Module):
    """Independent transcription of the released Latent Guard mapping layer."""

    def __init__(self, num_heads: int, head_dim: int, out_dim: int = 128) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.key_d = head_dim * num_heads
        self.x1_to_key = nn.Linear(768, self.key_d)
        self.x2_to_query = nn.Linear(768, self.key_d)
        self.x1_to_value = nn.Linear(768, self.key_d)
        self.final_mlp = nn.Linear(self.key_d, out_dim)
        self.mlp_query1 = nn.Linear(self.key_d, out_dim)
        self.tempr = nn.Parameter(torch.tensor(1 / 0.07), requires_grad=True)

    def forward(self, x1, x2):
        batch_size, seq_len, _ = x1.shape
        key = self.x1_to_key(x1).view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
        )
        key = key.transpose(1, 2)
        value = self.x1_to_value(x1).view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
        )
        value = value.transpose(1, 2)
        query = self.x2_to_query(x2).view(
            batch_size,
            1,
            self.num_heads,
            self.head_dim,
        )
        query = query.transpose(1, 2)
        attention_scores = torch.einsum("bnqd,bnkd->bnqk", query, key)
        attention_scores = attention_scores / (self.head_dim**0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        value = torch.einsum("bnqk,bnkd->bnqd", attention_weights, value)
        value = self.final_mlp(value.view(batch_size, -1))
        query = self.mlp_query1(query.view(batch_size, -1))
        return value, query


class OfficialClipWrapper:
    """CLIP embedding path used by the official Latent Guard repository."""

    def __init__(self, model_id: str, device: str) -> None:
        self.device = device
        self.model = CLIPModel.from_pretrained(model_id).to(device)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id)
        self.model.eval()

    def get_embedding(self, text: str):
        encoded = self.tokenizer(
            [text],
            truncation=True,
            max_length=77,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        tokens = encoded["input_ids"].to(self.device)
        hidden = self.model.text_model(input_ids=tokens).last_hidden_state
        eos_index = int(tokens.argmax(dim=-1).item())
        pooled = hidden[:, eos_index, :].unsqueeze(1)
        embedding = torch.cat([pooled, hidden], dim=1)
        if embedding.shape != (1, 78, 768):
            raise RuntimeError(
                f"Official CLIP path returned {tuple(embedding.shape)}, expected (1, 78, 768)."
            )
        return embedding


def load_state_dict(path: Path, device: str):
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        return state["state_dict"]
    return state


def official_scores(
    prompt: str,
    concepts: list[str],
    weights_path: Path,
    model_id: str,
    device: str,
) -> list[float]:
    model = OfficialEmbeddingMappingLayer(num_heads=16, head_dim=32, out_dim=128).to(device)
    model.load_state_dict(load_state_dict(weights_path, device))
    model.eval()
    clip = OfficialClipWrapper(model_id, device)

    with torch.inference_mode():
        concept_embeddings = [clip.get_embedding(concept) for concept in concepts]
        concept_embeddings = torch.cat(concept_embeddings, dim=0)[:, 0, :]
        prompt_embedding = clip.get_embedding(prompt)
        repeated_prompt = prompt_embedding.repeat(len(concepts), 1, 1)
        value, query = model(repeated_prompt, concept_embeddings)
        value = F.normalize(value, p=2, dim=1)
        query = F.normalize(query, p=2, dim=1)
        scores = torch.sum(value * query, dim=1) * model.tempr
    return [float(score) for score in scores.detach().cpu().tolist()]


def load_concepts(path: Path, limit: int) -> list[str]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    concepts = list((data.get("concepts") or {}).keys())
    if not concepts:
        raise RuntimeError(f"No concepts found in {path}.")
    return concepts[:limit] if limit > 0 else concepts


def release_local_scorer(scorer: _LatentGuardLiteScorer, device: str) -> None:
    scorer._concept_embedding_cache.clear()
    scorer._model = None
    scorer._clip_model = None
    scorer._tokenizer = None
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare framework Latent Guard scores with the official scoring path."
    )
    parser.add_argument(
        "--prompt",
        default="a blue rabbit mascot running across a sunny plaza",
    )
    parser.add_argument(
        "--concepts-path",
        type=Path,
        default=Path("data/latent_guard/restricted_concepts_representative_100.yaml"),
    )
    parser.add_argument(
        "--weights-path",
        type=Path,
        default=Path("data/latent_guard/model_parameters.pth"),
    )
    parser.add_argument("--model-id", default="openai/clip-vit-large-patch14")
    parser.add_argument("--device", default=None)
    parser.add_argument("--threshold", type=float, default=9.0131)
    parser.add_argument(
        "--max-concepts",
        type=int,
        default=8,
        help="Number of blacklist concepts to compare; use 0 for all concepts.",
    )
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    concepts = load_concepts(args.concepts_path, args.max_concepts)

    local_scorer = _LatentGuardLiteScorer(
        weights_path=args.weights_path,
        model_id=args.model_id,
        device=device,
        num_heads=16,
        head_dim=32,
        out_dim=128,
        input_dim=768,
        batch_size=max(1, len(concepts)),
    )
    local_scores = local_scorer.score_many(args.prompt, concepts)
    release_local_scorer(local_scorer, device)
    reference_scores = official_scores(
        prompt=args.prompt,
        concepts=concepts,
        weights_path=args.weights_path,
        model_id=args.model_id,
        device=device,
    )

    mismatches = []
    print(f"device={device} concepts={len(concepts)} threshold={args.threshold}")
    print(f'prompt="{args.prompt}"')
    for concept, local, reference in zip(concepts, local_scores, reference_scores):
        delta = abs(local - reference)
        matches = math.isclose(local, reference, abs_tol=args.atol, rel_tol=args.rtol)
        status = "PASS" if matches else "FAIL"
        print(
            f"[{status}] concept={concept!r} framework={local:.8f} "
            f"official={reference:.8f} delta={delta:.3e}"
        )
        if not matches:
            mismatches.append(concept)

    local_max = max(local_scores)
    reference_max = max(reference_scores)
    local_blocked = local_max >= args.threshold
    reference_blocked = reference_max >= args.threshold
    decisions_match = local_blocked == reference_blocked
    print(
        "decision "
        f"framework={'BLOCKED' if local_blocked else 'ALLOWED'} ({local_max:.8f}) "
        f"official={'BLOCKED' if reference_blocked else 'ALLOWED'} ({reference_max:.8f})"
    )

    if mismatches or not decisions_match:
        raise SystemExit(
            f"Parity check failed: {len(mismatches)} score mismatch(es), "
            f"decision_match={decisions_match}."
        )
    print("Latent Guard parity check passed.")


if __name__ == "__main__":
    main()
