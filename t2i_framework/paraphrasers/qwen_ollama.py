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
    ) -> None:
        self.model = model
        self.generate_url = generate_url
        self.timeout_seconds = timeout_seconds
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
        phrase_role = infer_phrase_role(concept)
        role_rules = role_specific_rules(phrase_role)
        return f"""
You are a visual concept paraphraser.

Generate exactly `{count}` short visual replacement phrases for ONLY the selected concept.

Selected concept:
`{concept}`

Full prompt context, provided ONLY to disambiguate the concept:
`{context or concept}`

Detected grammatical role of the selected concept:
`{phrase_role}`

Rules:

1. Replace ONLY the selected concept.
2. Each output must preserve the same grammatical role as the selected concept.
{role_rules}
3. Each output must be a short phrase, not a full sentence.
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
                "   Do not introduce a new subject, object, character, costume, location, or background.",
                "   Good examples for an action phrase: `giving up`, `yielding`, `lowering their guard`.",
            ]
        )
    return "\n".join(
        [
            "   If the selected concept is a noun phrase, return only noun phrases.",
            "   Do not use action verbs such as `holding`, `carrying`, `running`, `walking`, `standing`, `sitting`, `fighting`, `looking`, or `posing`.",
            "   Do not introduce a new action, scene event, or relationship.",
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
