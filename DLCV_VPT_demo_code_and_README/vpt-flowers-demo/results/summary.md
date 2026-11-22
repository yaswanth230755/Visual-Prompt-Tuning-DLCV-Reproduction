# VPT (official code) — results summary

Dataset: Oxford Flowers-102 (train 1020 / val 1020 / test 6149); ViT-B/16 ImageNet-21k; prompts 10; epochs 30; batch 32; profile standard; seeds [0].

## Final results (lr chosen on validation; test evaluated once)

| method_label | test_% | test_std | val_% | base_lr | tuned_percent | n_trainable_params | n_seeds | train_min |
|---|---|---|---|---|---|---|---|---|
| Linear | 97.951 | nan | 98.627 | 25.0 | 0.091 | 78438 | 1 | 15.7 |
| VPT-shallow (official) | 98.032 | nan | 98.627 | 5.0 | 0.1 | 86118 | 1 | 24.0 |
| VPT-deep (official) | 98.618 | nan | 99.118 | 25.0 | 0.198 | 170598 | 1 | 24.2 |
| Full fine-tuning | 97.512 | nan | 98.333 | 0.001 | 100.0 | 85877094 | 1 | 29.2 |
| Partial-1 | 97.951 | nan | 98.627 | 1.0 | 8.347 | 7167846 | 1 | 17.1 |
| Bias | 98.992 | nan | 99.118 | 1.0 | 0.211 | 181350 | 1 | 23.7 |
| Adapter | 98.357 | nan | 99.118 | 1.0 | 2.12 | 1858278 | 1 | 23.6 |

## Validation accuracy (%) per base lr

| method | 0.0001 | 0.001 | 0.005 | 0.1 | 1.0 | 5.0 | 25.0 |
|---|---|---|---|---|---|---|---|
| linear | nan | nan | nan | nan | 97.94 | 98.33 | 98.63 |
| vpt_shallow | nan | nan | nan | nan | 98.24 | 98.63 | 98.63 |
| vpt_deep | nan | nan | nan | nan | 97.84 | 98.63 | 99.12 |
| full | 93.14 | 98.33 | 93.92 | nan | nan | nan | nan |
| partial1 | nan | nan | nan | 97.75 | 98.63 | nan | nan |
| bias | nan | nan | nan | 98.14 | 99.12 | nan | nan |
| adapter | nan | nan | nan | 97.16 | 99.12 | nan | nan |

## Claims

| index | C1 deep > every other parameter-efficient method run (Linear, Partial-1, Bias, Adapter) | C1b deep > Linear | C2 deep >= Full | C2b deep - Full (pts) | C2c deep tunes < 1 % of params | C3 deep >= shallow | best method |
|---|---|---|---|---|---|---|---|
| OxfordFlowers | False | True | True | 1.11 | True | True | Bias |

## Prompt-length ablation

| prompts | tuned_percent | val_pct | test_pct |
|---|---|---|---|
| 1 | 0.102 | 98.431 | 98.048 |
| 5 | 0.145 | 99.216 | 98.764 |
| 10 | 0.198 | 99.118 | 98.618 |
| 50 | 0.625 | 97.941 | 97.853 |
| 100 | 1.152 | 97.255 | 97.658 |
| 200 | 2.191 | 96.275 | 94.666 |
