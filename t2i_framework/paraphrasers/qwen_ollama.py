from __future__ import annotations

import json
import re
from typing import Any
from urllib import request

from t2i_framework.core.word_lists import load_term_set


OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "qwen3:14b"
ACTION_TERMS = load_term_set("action_terms")


class QwenOllamaParaphraser:
    """Ollama/Qwen provider for short visual concept paraphrase candidates."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        generate_url: str = OLLAMA_GENERATE_URL,
        timeout_seconds: int = 120,
        detail_level: str = "compact",
    ) -> None:
        self.model = model
        self.generate_url = generate_url
        self.timeout_seconds = timeout_seconds
        self.detail_level = normalize_detail_level(detail_level)
        self.last_raw_response: str | None = None
        self.last_candidates: list[str] = []

    def generate_candidates(
        self,
        concept: str,
        context: str = "",
        count: int = 5,
    ) -> list[str]:
        prompt = self._build_prompt(concept, context, count)
        response = self._generate(prompt)
        candidates = parse_candidate_list(response, limit=count)
        self.last_raw_response = response
        self.last_candidates = candidates
        return candidates

    def unload(self) -> None:
        unload_ollama_model(
            self.model,
            generate_url=self.generate_url,
            timeout_seconds=self.timeout_seconds,
        )

    def _generate(self, prompt: str) -> str:
        payload = json.dumps(
            {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
            }
        ).encode("utf-8")
        req = request.Request(
            self.generate_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=self.timeout_seconds) as response:
            data = json.loads(response.read().decode("utf-8"))
        return str(data["response"])

    def _build_prompt(self, concept: str, context: str, count: int) -> str:
        return build_paraphrase_prompt(concept, context, count, self.detail_level)


def build_paraphrase_prompt(
    concept: str,
    context: str,
    count: int,
    detail_level: str = "compact",
) -> str:
    """Build the provider-independent visual paraphrasing instruction."""
    detail_level = normalize_detail_level(detail_level)
    phrase_role = infer_phrase_role(concept)
    role_rules = role_specific_rules(phrase_role)
    detail_rules = detail_level_rules(detail_level, phrase_role)
    return f"""
You are a visual concept paraphraser.

Generate exactly `{count}` short visual replacement phrases for ONLY the selected concept.

Selected concept:
`{concept}`

Full prompt context, provided ONLY to disambiguate the concept:
`{context or concept}`

Detected grammatical role of the selected concept:
`{phrase_role}`

Detail level:
`{detail_level}`

Rules:

1. Replace ONLY the selected concept.
2. Each output must preserve the same grammatical role as the selected concept.
{role_rules}
3. Each output must be a phrase, not a full sentence.
{detail_rules}
4. Describe the selected concept using intrinsic, visually observable characteristics.
5. Prefer:

   * object/person type or role
   * body or facial appearance
   * hairstyle
   * clothing
   * colors
   * shape
   * distinctive visual features
6. Visual attributes that are inherently part of the concept may be introduced even if they are not explicitly written in the full prompt.
7. Do NOT introduce new scene-level objects, props, characters, poses, relationships, locations, backgrounds, lighting, or camera information.
8. Do NOT copy actions or objects from the full prompt unless they are part of the selected concept itself.
9. Prefer compact attribute constructions such as:

   * `red-capped mustached plumber in blue overalls`
   * `small round yellow creature with pointed ears`
   * `giving up`
   * `lowering their guard`
     rather than:
   * `a plumber wearing a red hat while holding a tool`
10. Do NOT include the selected concept phrase itself.
11. Do NOT include copyrighted character names, franchise names, brand names, real people, sexual content, violence, or illegal content.
12. Do NOT rewrite or describe any information from the full prompt outside the selected concept.
13. Do NOT add explanatory text.

The alternatives should preserve the same visual concept while varying wording and attribute combinations slightly.

Return exactly `{count}` strings as a valid JSON array.

Return ONLY the JSON array.


        """


def unload_ollama_model(
    model: str,
    generate_url: str = OLLAMA_GENERATE_URL,
    timeout_seconds: int = 120,
) -> None:
    payload = json.dumps(
        {
            "model": model,
            "prompt": "",
            "stream": False,
            "keep_alive": 0,
        }
    ).encode("utf-8")
    req = request.Request(
        generate_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout_seconds) as response:
        response.read()


def infer_phrase_role(phrase: str) -> str:
    words = [word.lower() for word in re.findall(r"\b[\w'-]+\b", phrase)]
    if not words:
        return "noun phrase"
    if any(word in ACTION_TERMS or word.endswith("ing") for word in words):
        return "action or verb phrase"
    return "noun phrase"


def role_specific_rules(phrase_role: str) -> str:
    if phrase_role == "action or verb phrase":
        return "\n".join(
            [
                "   If the selected concept is an action or verb phrase, return only action or verb phrases.",
                "   Preserve every object, recipient, direction, particle, or other complement that is already part of the selected concept.",
                "   Do not remove an existing participant and do not introduce a new subject, object, character, costume, location, or background.",
                "   Good examples: `chasing a balloon` -> `pursuing a balloon`; `giving up` -> `yielding`; `lowering their guard` -> `dropping their guard`.",
            ]
        )
    return "\n".join(
        [
            "   If the selected concept is a noun phrase, return only noun phrases.",
            "   Do not use action verbs such as `holding`, `carrying`, `running`, `walking`, `standing`, `sitting`, `fighting`, `looking`, or `posing`.",
            "   Do not introduce a new action, scene event, or relationship.",
        ]
    )


def normalize_detail_level(detail_level: str) -> str:
    normalized = str(detail_level).strip().lower().replace("-", "_")
    aliases = {
        "short": "compact",
        "small": "compact",
        "compact": "compact",
        "default": "compact",
        "medium": "medium",
        "moderate": "medium",
        "balanced": "medium",
        "detailed": "detailed",
        "detail": "detailed",
        "long": "detailed",
    }
    if normalized not in aliases:
        raise ValueError(
            f"Unsupported paraphraser detail_level {detail_level!r}. "
            "Use compact, medium, or detailed."
        )
    return aliases[normalized]


def detail_level_rules(detail_level: str, phrase_role: str) -> str:
    detail_level = normalize_detail_level(detail_level)
    if phrase_role == "action or verb phrase":
        return "\n".join(
            [
                "3a. Detail policy for action or verb phrases:",
                "   * Ignore medium/detailed noun-description behavior.",
                "   * Keep each output compact, normally about 1-10 words.",
                "   * Return the action together with every object or complement already contained in the selected concept.",
                "   * Do not add a subject, new object, scene, motive, or visual description.",
            ]
        )

    if detail_level == "compact":
        return "\n".join(
            [
                "3a. Detail policy for noun phrases:",
                "   * Keep each output compact, about 3-8 words.",
                "   * Include the concept type plus 1-3 distinctive visual attributes.",
            ]
        )

    if detail_level == "medium":
        return "\n".join(
            [
                "3a. Detail policy for noun phrases:",
                "   * Use about 8-18 words.",
                "   * Include the concept type plus 2-4 distinctive visual attributes.",
                "   * Prefer stable visual identity features such as colors, clothing, body shape, face, hair, texture, or silhouette.",
                "   * Good medium examples: `short mustached cartoon plumber with red cap and blue overalls`; `large spiked turtle-like creature with orange hair and heavy shell`.",
                "   * Bad medium examples: `a plumber standing in a castle holding a wrench`; `a brave hero jumping through a colorful fantasy world`.",
            ]
        )

    return "\n".join(
        [
            "3a. Detail policy for noun phrases:",
            "   * Use about 18-35 words.",
            "   * Include the concept type plus 4-7 distinctive visual attributes.",
            "   * Prefer stable identity-preserving attributes: body shape, face shape, hairstyle, clothing, colors, accessories, texture, and silhouette.",
            "   * You may describe iconic visual features normally associated with the selected concept, even if they are not written in the full prompt.",
            "   * Good detailed examples: `short stocky cartoon plumber with round face, large nose, black mustache, red cap, red shirt, blue overalls, white gloves, brown shoes`; `blue spiky cartoon hedgehog with tan muzzle, large eyes, white gloves, slim body, red shoes with white straps`.",
            "   * Bad detailed examples: `a plumber jumping through a mushroom kingdom under bright sunlight`; `a villain fighting a hero inside a castle`.",
        ]
    )


def parse_candidate_list(raw_response: str, limit: int) -> list[str]:
    text = raw_response.strip()
    parsed: Any = None

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\[[\s\S]*\]", text)
        if match is not None:
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError:
                parsed = None

    if isinstance(parsed, dict):
        parsed = parsed.get("candidates") or parsed.get("paraphrases")

    if isinstance(parsed, list):
        candidates = [str(item).strip() for item in parsed]
    else:
        candidates = []
        for line in text.splitlines():
            cleaned = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
            cleaned = cleaned.strip("\"'")
            if cleaned:
                candidates.append(cleaned)

    unique_candidates = []
    seen = set()
    for candidate in candidates:
        candidate = re.sub(r"\s+", " ", candidate).strip()
        if not candidate:
            continue
        key = candidate.lower()
        if key in seen:
            continue
        seen.add(key)
        unique_candidates.append(candidate)
        if len(unique_candidates) >= limit:
            break

    return unique_candidates
