# How To Add An Attack

Create a new file in `t2i_framework/attacks/`:

```python
from typing import Any

from t2i_framework.attacks.base import Attack
from t2i_framework.core.types import AttackCandidate


class MyAttack(Attack):
    name = "my_attack"

    def generate(
        self,
        prompt: str,
        target_concept: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> list[AttackCandidate]:
        return [AttackCandidate(text=prompt, metadata={"method": "my_attack"})]
```

Register it in `t2i_framework/core/registry.py`:

```python
from t2i_framework.attacks.my_attack import MyAttack

ATTACK_REGISTRY = {
    ...
    "my_attack": MyAttack,
}
```

Then run:

```bash
python main.py --model mock --attack my_attack --defense none --prompt "a red cube robot holding a balloon" --target "red cube robot"
```

Keep default examples safe and synthetic. Do not include real harmful prompt examples in code, docs, or tests.

`groot_lite` is the built-in semantic decomposition example. It rewrites known synthetic target concepts into visual attribute descriptions, for example `blue rabbit mascot` becomes a phrase such as `blue long-eared costume character`. It is deterministic and does not call an external model. Its concept decompositions live in `data/groot_decompositions.yaml`, so new safe synthetic concepts can be added without changing Python source code.

The decomposition file can be switched per run:

```yaml
attack:
  name: groot_lite
  decompositions_path: data/groot_decompositions_external_template.yaml
```

Use `--max-candidates` to evaluate more than the first candidate returned by an attack.

Examples:

- `groot_lite` can return one candidate per decomposition for the target concept.
- `search_attack` returns the original prompt plus generated variants.
- attacks that only return one candidate ignore larger `--max-candidates` values in practice.

`--max-candidates 3` means "evaluate at most three candidates from this attack and write one result row per candidate." It does not mean three different input prompts. Batch input prompts are controlled with `--prompt-file`.
