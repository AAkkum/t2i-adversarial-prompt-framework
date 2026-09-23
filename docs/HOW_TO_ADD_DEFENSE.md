# Add A Defense

1. Create one class under `t2i_framework/defenses/` and subclass `Defense`.
2. Implement `check_prompt`, `check_image`, or both.
3. Return a `DefenseDecision` with `allowed`, `reason`, and optional metadata.
4. Register the class in `t2i_framework/core/registry.py`.
5. Add a YAML file under `configs/defenses/` only when the defense has settings.
6. Add focused tests in `tests/test_defenses.py`.

Minimal prompt defense:

```python
from t2i_framework.core.types import DefenseDecision
from t2i_framework.defenses.base import Defense


class ExampleDefense(Defense):
    name = "example"

    def check_prompt(self, prompt, target_concept=None, context=None):
        blocked = bool(target_concept and target_concept.lower() in prompt.lower())
        return DefenseDecision(
            allowed=not blocked,
            reason="target found" if blocked else "allowed",
        )
```

The defense must not decide experiment success. It only blocks or allows. The
shared LLM evaluator runs afterward.
