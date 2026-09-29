# Add An Attack

An attack receives an original prompt and optional target concept, then returns
one or more `AttackCandidate` objects.

## 1. Implement The Class

Create a module under `t2i_framework/attacks/` and subclass `Attack`.

```python
from t2i_framework.attacks.base import Attack
from t2i_framework.core.types import AttackCandidate


class ExampleAttack(Attack):
    name = "example"

    def generate(self, prompt, target_concept=None, context=None):
        candidate = prompt
        if target_concept:
            candidate = prompt.replace(target_concept, "visual description", 1)
        return [
            AttackCandidate(
                text=candidate,
                metadata={"method": self.name, "query_count": 1},
            )
        ]
```

Do not modify the original prompt or shared context unexpectedly. Put
attack-specific diagnostics in candidate metadata. Implement `cleanup` when the
attack owns models, GPU memory, files, or server resources.

## 2. Adaptive Attacks

Set `adaptive = True` only when the attack must observe each completed result
before choosing its next candidate. Return the initial candidate from `generate`
and implement `next_candidate(previous, result, context)`. Return `None` when
the search should stop. Groot is the reference implementation.

`--max-candidates` is a runner limit, so attacks must not assume every possible
candidate will be evaluated.

## 3. Add Configuration

Add `configs/attacks/example.yaml` when needed:

```yaml
attack:
  name: example
  candidate_count: 3
```

Read settings from `context["config"]["attack"]` and reject invalid or unknown
values. Use `context["seed"]` for reproducible randomness.

## 4. Register And Export

Add the class to `ATTACK_REGISTRY` in `t2i_framework/core/registry.py` and to
`t2i_framework/attacks/__init__.py`. The registry key, class `name`, and YAML
`name` must match.

## 5. Test

Test deterministic output, target handling, candidate limits, invalid settings,
resource cleanup, and metadata. Mock learned models and network clients in unit
tests.

```bash
python main.py --list-components
pytest -q
ruff check .
```

The attack proposes prompts. It does not decide whether the defense was bypassed
or whether the generated image was successful.
