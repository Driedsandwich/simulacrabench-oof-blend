# Out-of-fold blended survey-answer predictor for SimulacraBench

Submission code for [SimulacraBench](https://simulacrabench.org/) (NeurIPS 2026 Competition Track): predict how respondents of UNHCR, UNICEF and World Bank surveys answered held-out questions, from the attributes that are given.

`main.py` defines `predict(frame, schema)` as the competition harness expects. Everything is learned inside `predict()` from the visible rows of the instrument at hand: no pretrained model, no external data, no offline training, CPU only.

## Method in brief

For every scored item, a few candidate predictors are fitted on the visible rows:

| name | what it is |
|---|---|
| `marg` | smoothed crowd marginal (+0.5 counts), the zero of the skill scale |
| `dir` | Dirichlet back-off along a ladder of GIVEN strata ranked by mutual information; depth and strength chosen out of fold |
| `lr` | multinomial logistic regression on one-hot GIVEN; regularisation chosen out of fold |
| `hgb` | `HistGradientBoostingClassifier` on categorical GIVEN codes, only when the clock allows |

Their 3-fold out-of-fold (OOF) predictions are scored with the grader's own transform (renormalise, mix with a flat 1e-3 floor, log). Convex blend weights are fitted on that loss by exponentiated gradient and shrunk slightly towards the marginal, then one temperature per item is chosen out of fold.

Skip logic: when a gate's parent is GIVEN and the visible rows obey the rule exactly, the sentinel slot is deterministic. Otherwise whole-vector candidates (gate chain, learned P(asked), and a direct model with the sentinel as a class) are blended out of fold the same way.

A single clock from module import splits the shared time budget across the three instruments in proportion to their work (visible rows x scored items) and drops candidates one by one as the deadline nears, ending at the marginal.

The code learns only from the `frame` passed to `predict()`. It writes no files, opens no sockets, starts no processes and prints nothing.

## Run it on the organizers' sandbox

```bash
git clone https://github.com/SituatedEvals/public situated-public
cd situated-public
pip install -r requirements.txt -r /path/to/this/repo/requirements.txt
python make_sandbox.py --schema data/unicef.json --out _sandbox/unicef
python score.py --submission /path/to/this/repo --data _sandbox/unicef --schema data/unicef.json --phase 1
```

Use `--phase 2` for the final-phase form, and `data/world_bank.json` or `data/unhcr.json` for the other instruments.

Exact skill on the sandbox in the final form (`--phase 2`; synthetic data, and the organizers note that sandbox scores are not a measure of value):

| UNICEF | World Bank | UNHCR |
|---|---|---|
| 0.3943 | 0.2807 | 0.2204 |

In the development form (`--phase 1`) the scorer adds noise and rounds the printed skill, as the leaderboard does, so those numbers vary between runs.

## Requirements

Python 3.13 with the pinned packages in `requirements.txt` (NumPy 2.2.6, pandas 2.3.3, scikit-learn 1.7.2).

## License

MIT (see `LICENSE`).
