# LatentGuard Weights

This folder is reserved for optional LatentGuard pretrained weights.

The `latent_guard_lite` defense expects the released LatentGuard state dict at:

```text
data/latent_guard/model_parameters.pth
```

Download it from the official LatentGuard repository:

```text
https://github.com/rt219/LatentGuard
```

The framework does not commit the `.pth` file because it is an external model
artifact. The adapter is intended as a LatentGuard-inspired integration for
experiments, not a full retraining reproduction.
