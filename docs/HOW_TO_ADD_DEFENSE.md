# Add A Defense

Use the smallest interface that implements the method. A defense may inspect a
prompt, inspect a completed image, or replace the Diffusers generation call.

## 1. Implement The Class

Create a module under `t2i_framework/defenses/` and subclass `Defense`.

```python
from t2i_framework.core.types import DefenseDecision
from t2i_framework.defenses.base import Defense


class ExampleDefense(Defense):
    name = "example"

    def check_prompt(self, prompt, target_concept=None, context=None):
        blocked = bool(target_concept and target_concept.casefold() in prompt.casefold())
        return DefenseDecision(
            allowed=not blocked,
            reason="target found" if blocked else "allowed",
        )
```

Override `check_image` only when the method examines the generated image. The
runner generates into quarantine and publishes the image only after this method
returns `allowed=True`.

Generation-time methods such as SAFREE and TraSCE additionally implement:

- `validate_model(model_id)` to reject incompatible pipelines early
- `generate_image(pipeline, *, context, **kwargs)` to perform generation

Their `check_prompt` method stores the prepared request in `context`, and
`check_image` verifies that protected generation completed. Never silently fall
back to undefended generation after an error.

## 2. Add Configuration

Add `configs/defenses/example.yaml` when settings are needed:

```yaml
defense:
  name: example
  threshold: 0.7
```

The class is constructed without configuration arguments. Read settings from
`context["config"]["defense"]` during the check and validate unknown or invalid
values explicitly.

## 3. Register And Export

Add the class to `DEFENSE_REGISTRY` in `t2i_framework/core/registry.py` and to
`t2i_framework/defenses/__init__.py`. The registry key, class `name`, and YAML
`name` must match.

## 4. Test

Add focused tests under `tests/` for allowed, blocked, malformed-config, and
failure paths. A generation-time defense also needs model-compatibility tests
and proof that failure never releases an unprotected image.

Run:

```bash
python main.py --list-components
pytest -q
ruff check .
```

A defense decides whether processing is allowed. It must not set experiment
success; the shared LLM evaluator runs afterward.
