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
