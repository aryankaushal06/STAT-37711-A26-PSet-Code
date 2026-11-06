# STAT 37711 Problem Set 2

Code, results, and report for Homework 2: heteroscedastic regression (`p1.py`), categorical
encoding (`p2.py`), and Random Kitchen Sinks versus an MLP (`p3.py`).

## Setup

Run these from the `PS2/` directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Running

```bash
python p1.py
python p2.py
python p3.py
```

| Script | Input | Output |
| --- | --- | --- |
| `p1.py` | `data/taxi_trips.parquet` | `results/p1.json`, `figures/p1_*.png` |
| `p2.py` | `data/retail.csv` | `results/p2.json`, `figures/p2_*.png` |
| `p3.py` | `data/train.csv`, `val.csv`, `test.csv` | `results/p3.json`, `figures/p3_*.png`, `rks_test_predictions.csv`, `mlp_test_predictions.csv` |

`p3.py` finishes by checking that both prediction files have 200 finite values and that its
figures are not blank.

## Building the report

Requires a LaTeX distribution that provides `pdflatex`:

```bash
pdflatex -interaction=nonstopmode -halt-on-error PS2.tex
pdflatex -interaction=nonstopmode -halt-on-error PS2.tex
```

## Reproducibility

All seeds are fixed (42, plus further seeds for the robustness checks). On CPU, repeated runs
on the same machine produced byte-identical `results/*.json` and prediction files. Results may
differ across PyTorch versions or hardware.

The scripts use the CPU by default, which was about 8x faster than MPS on the Problem 1 model.
To use Apple Silicon for `p1.py` or `p2.py`, set `PS2_DEVICE=mps`; they fall back to the CPU if
MPS is unavailable. Results on MPS are not expected to match the reported values exactly.

## Files

- `plotting.py`: shared figure style and the 2D and 3D surface plots.
- `dataloader.py`: loads the Problem 3 CSV files.
- `models.py`: the course's unused starter skeleton, whose TODO methods are deliberately left
  unimplemented. The completed models are in `p3.py`.
