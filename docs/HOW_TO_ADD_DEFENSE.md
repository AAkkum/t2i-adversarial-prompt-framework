# How To Add A Defense

Defenses can run before generation on text prompts, after generation on images, or both.

```python
from pathlib import Path
from typing import Any

from t2i_framework.core.types import DefenseDecision
from t2i_framework.defenses.base import Defense


class MyDefense(Defense):
    name = "my_defense"

    def check_prompt(
        self,
        prompt: str,
        target_concept: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> DefenseDecision:
        return DefenseDecision(allowed=True, reason="prompt allowed")

    def check_image(
        self,
        image_path: Path,
        target_concept: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> DefenseDecision:
        return DefenseDecision(allowed=True, reason="image allowed")
```

Register it in `t2i_framework/core/registry.py` under `DEFENSE_REGISTRY`.

Pre-generation blocking prevents the model adapter from being called. Post-generation blocking marks the result as unsuccessful and prevents the blocked image from being published as a final output image.

`image_clip_filter` is the built-in post-generation image defense. It compares the generated image with the `target_concept` using CLIP image-text similarity. If the score is above the threshold, the defense blocks the image because the restricted synthetic target concept still appears to be present.

Defense-specific CLIP settings can be passed through the run config:

```yaml
defense:
  image_clip:
    model_id: openai/clip-vit-base-patch32
    threshold: 0.25
```
