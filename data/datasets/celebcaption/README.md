# CelebCaption 5 Per Person Sample

Framework-ready prompt sample created from the public CelebCaption GitHub repository:

- Repository: `Gloriel621/CelebCaption`
- Source directory: `Captions_full`
- Source URL: https://github.com/Gloriel621/CelebCaption

Sampling rule:

- 150 public figures
- 5 prompts per person
- 750 total rows
- Uses named caption variants: 3 `summary_with_name` prompts and 2 `detailed_with_name` prompts per person

The sample is stored in framework-ready CSV format:

```csv
id,prompt,target_concept,category,caption_variant,image_file,source_file
```

The original CelebCaption paper citation listed by the repository:

```bibtex
@inproceedings{moon2026celebcaption,
  title     = {CelebCaption: A Benchmark Dataset for Identity-Sensitive Unlearning in Image Captioning},
  author    = {Moon, Hakjun and Woo, Simon S.},
  booktitle = {Proceedings of the 19th ACM International Conference on Web Search and Data Mining (WSDM)},
  year      = {2026},
  doi       = {10.1145/3773966.3779359}
}
```
