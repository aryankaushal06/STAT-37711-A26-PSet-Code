"""STAT 37711 PS2 Problem 2: target encoding, learned embeddings, and a hybrid model.

Run with: python p2.py
Writes results/p2.json and figures/p2_*.png.
"""
import copy
import json
import os
from pathlib import Path

import category_encoders as ce
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, train_test_split
from torch import nn

from plotting import SERIES_COLORS, SERIES_MARKERS, apply_style

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
ROBUSTNESS_SEEDS = [42, 43, 44]
COLS = ["StockCode", "CustomerID"]
TE_MIN_SAMPLES_LEAF = 20  # category_encoders defaults
TE_SMOOTHING = 10.0
RIDGE_ALPHA = 10.0
N_FOLDS = 5
MAX_EMBEDDING_DIM = 32
HIDDEN = (64, 32)
LR = 3e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 4096
HUBER_BETA = 0.5
MAX_EPOCHS = 100
PATIENCE = 5
MIN_DELTA = 1e-5
# Index 0 means "unknown". No training row would otherwise reach it, leaving a random,
# untrained vector for unseen test categories, so each training category is masked to
# unknown with this probability.
UNK_RATE = 0.01
COUNT_BINS = [1, 10, 50, 200, 1000, np.inf]
COUNT_LABELS = ["1-9", "10-49", "50-199", "200-999", "1000+"]


def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    """CPU by default for reproducibility; set PS2_DEVICE=mps to use Apple Silicon."""
    if os.environ.get("PS2_DEVICE", "cpu") == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_encoder():
    return ce.TargetEncoder(
        cols=COLS, min_samples_leaf=TE_MIN_SAMPLES_LEAF, smoothing=TE_SMOOTHING,
        handle_unknown="value", handle_missing="value",
    )


def out_of_fold_encode(X_train, target):
    """Encode each training row with an encoder that never saw that row's target."""
    encoded = np.zeros((len(X_train), len(COLS)), dtype="float64")
    for fit_idx, enc_idx in KFold(N_FOLDS, shuffle=True, random_state=SEED).split(X_train):
        encoder = make_encoder().fit(X_train.iloc[fit_idx], target.iloc[fit_idx])
        encoded[enc_idx] = encoder.transform(X_train.iloc[enc_idx])[COLS].to_numpy("float64")
    return encoded


def encode_all(X_train, X_val, X_test, target):
    """Training rows are out-of-fold; held-out rows use an encoder fitted on all training rows."""
    full = make_encoder().fit(X_train, target)
    return {
        "train": out_of_fold_encode(X_train, target),
        "val": full.transform(X_val)[COLS].to_numpy("float64"),
        "test": full.transform(X_test)[COLS].to_numpy("float64"),
        "prior": float(target.mean()),
    }


def score(t_true, quantity_true, t_pred, t_low, t_high):
    """Metrics on log1p and original scales; predictions are clipped to the training range."""
    t_pred = np.clip(t_pred, t_low, t_high)
    quantity_error = np.abs(quantity_true - np.expm1(t_pred))
    return {
        "log1p_mae": float(np.mean(np.abs(t_true - t_pred))),
        "log1p_rmse": float(np.sqrt(np.mean((t_true - t_pred) ** 2))),
        "quantity_mae": float(np.mean(quantity_error)),
        "quantity_median_ae": float(np.median(quantity_error)),
    }


class CategoricalNet(nn.Module):
    """MLP on any of: an embedding per categorical feature, and target-encoded features."""

    def __init__(self, sizes, use_embeddings, use_target_encoding):
        super().__init__()
        self.use_embeddings = use_embeddings
        self.use_target_encoding = use_target_encoding
        width = 0
        if use_embeddings:
            self.embeddings = nn.ModuleList(
                [nn.Embedding(size + 1, embedding_dim(size)) for size in sizes]
            )
            width += sum(e.embedding_dim for e in self.embeddings)
        if use_target_encoding:
            width += len(sizes)
        layers = []
        for next_width in HIDDEN:
            layers += [nn.Linear(width, next_width), nn.ReLU()]
            width = next_width
        self.layers = nn.Sequential(*layers, nn.Linear(width, 1))

    def forward(self, ids, encoded):
        parts = []
        if self.use_embeddings:
            parts += [emb(ids[:, j]) for j, emb in enumerate(self.embeddings)]
        if self.use_target_encoding:
            parts.append(encoded)
        return self.layers(torch.cat(parts, dim=1)).squeeze(-1)


def embedding_dim(cardinality):
    return min(MAX_EMBEDDING_DIM, max(4, int(np.ceil(np.sqrt(cardinality)))))


def fit_network(data, sizes, use_embeddings, use_target_encoding, seed):
    """Smooth-L1 on log1p(Quantity); validation early stopping restores the best parameters."""
    set_seeds(seed)
    device = data["ids_train"].device
    net = CategoricalNet(sizes, use_embeddings, use_target_encoding).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.SmoothL1Loss(beta=HUBER_BETA)
    best_val, best_state, best_epoch = float("inf"), None, 0
    history = {"train": [], "val": []}
    n = len(data["t_train"])
    for epoch in range(1, MAX_EPOCHS + 1):
        net.train()
        total = 0.0
        for batch in torch.randperm(n, device=device).split(BATCH_SIZE):
            ids = data["ids_train"][batch].clone()
            encoded = data["enc_train"][batch].clone()
            mask = torch.rand(ids.shape, device=device) < UNK_RATE
            ids[mask] = 0
            encoded = torch.where(mask, data["enc_prior"], encoded)
            optimizer.zero_grad()
            loss = loss_fn(net(ids, encoded), data["t_train"][batch])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite training loss at epoch {epoch}")
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch)
        val_pred = predict_network(net, data["ids_val"], data["enc_val"])
        val_loss = float(loss_fn(torch.from_numpy(val_pred), data["t_val"].cpu()).item())
        history["train"].append(total / n)
        history["val"].append(val_loss)
        if val_loss < best_val - MIN_DELTA:
            best_val, best_epoch = val_loss, epoch
            best_state = copy.deepcopy(net.state_dict())
        elif epoch - best_epoch >= PATIENCE:
            break
    net.load_state_dict(best_state)
    summary = {"best_epoch": best_epoch, "epochs_run": len(history["val"]), "val_smooth_l1": best_val}
    return net, summary, history


def predict_network(net, ids, encoded, batch_size=65536):
    net.eval()
    outputs = []
    with torch.no_grad():
        for i, e in zip(ids.split(batch_size), encoded.split(batch_size)):
            outputs.append(net(i, e).cpu())
    return torch.cat(outputs).numpy()


def paired_ci(difference):
    """Normal-approximation 95% interval for the mean of per-row differences."""
    half = 1.96 * difference.std(ddof=1) / np.sqrt(len(difference))
    return {"mean": float(difference.mean()), "ci95": [float(difference.mean() - half), float(difference.mean() + half)]}


def main():
    set_seeds(SEED)
    torch.set_num_threads(4)
    apply_style()
    device = get_device()

    df = pd.read_csv(DATA_DIR / "retail.csv", dtype={"StockCode": "string", "CustomerID": "string"})
    if df.isna().any().any() or not (df["Quantity"] > 0).all():
        raise ValueError("retail.csv must have no missing values and positive quantities")
    for col in COLS:
        df[col] = df[col].astype("category")
    quantity = df["Quantity"].to_numpy("float64")

    # (a) Exploration and splits.
    n_columns = sum(df[c].nunique() for c in COLS)
    pair_counts = df.groupby(COLS, observed=True).size()
    exploration = {
        "rows": len(df),
        "cardinality": {c: int(df[c].nunique()) for c in COLS},
        "one_hot_columns": int(n_columns),
        "one_hot_dense_gb_float32": len(df) * n_columns * 4 / 1e9,
        "one_hot_nonzero_fraction": 2 / n_columns,
        "fraction_rows_repeating_a_pair": float(1 - len(pair_counts) / len(df)),
        "exact_duplicate_rows": int(df.duplicated().sum()),
        "rows_per_category_median": {c: float(df[c].value_counts().median()) for c in COLS},
        "categories_with_at_most_5_rows": {c: int((df[c].value_counts() <= 5).sum()) for c in COLS},
    }
    target_stats = {
        "mean": float(quantity.mean()),
        "std": float(quantity.std()),
        "median": float(np.median(quantity)),
        "q90": float(np.quantile(quantity, 0.9)),
        "q99": float(np.quantile(quantity, 0.99)),
        "q999": float(np.quantile(quantity, 0.999)),
        "max": float(quantity.max()),
        "skew": float(stats.skew(quantity)),
        "log1p_mean": float(np.log1p(quantity).mean()),
        "log1p_std": float(np.log1p(quantity).std()),
        "log1p_skew": float(stats.skew(np.log1p(quantity))),
    }

    idx_train, idx_held = train_test_split(np.arange(len(df)), test_size=0.30, random_state=SEED)
    idx_val, idx_test = train_test_split(idx_held, test_size=0.50, random_state=SEED)
    train, val, test = df.iloc[idx_train], df.iloc[idx_val], df.iloc[idx_test]

    def log_target(frame):
        return np.log1p(frame["Quantity"].to_numpy("float64"))

    t_train, t_val, t_test = log_target(train), log_target(val), log_target(test)
    q_val, q_test = val["Quantity"].to_numpy("float64"), test["Quantity"].to_numpy("float64")
    t_low, t_high = float(np.log(2.0)), float(t_train.max())  # Quantity >= 1 implies t >= log 2

    split_stats = {}
    for name, frame in [("train", train), ("val", val), ("test", test)]:
        q = frame["Quantity"]
        split_stats[name] = {
            "n": len(frame), "mean": float(q.mean()), "median": float(q.median()),
            "q99": float(q.quantile(0.99)), "max": float(q.max()),
        }
    train_counts = {c: train[c].astype("string").value_counts() for c in COLS}
    unseen = {
        name: {c: float((~frame[c].astype("string").isin(train_counts[c].index)).mean()) for c in COLS}
        for name, frame in [("val", val), ("test", test)]
    }

    def scores(t_pred_val, t_pred_test):
        return {
            "val": score(t_val, q_val, t_pred_val, t_low, t_high),
            "test": score(t_test, q_test, t_pred_test, t_low, t_high),
        }

    def clipped_absolute_errors(t_pred):
        return np.abs(t_test - np.clip(t_pred, t_low, t_high))

    results_models, test_preds = {}, {}

    # Reference: a constant at the training median, which minimizes training absolute error.
    median = float(np.median(t_train))
    results_models["constant_median"] = scores(np.full_like(t_val, median), np.full_like(t_test, median))
    test_preds["constant_median"] = np.full_like(t_test, median)

    # (b) Target encoding: the assignment's mean of raw Quantity, and the mean of log1p(Quantity).
    # Both variants feed a ridge regression for log1p(Quantity); validation picks one.
    X_train, X_val, X_test = train[COLS], val[COLS], test[COLS]
    target_series = {
        "raw_quantity_mean": train["Quantity"].astype("float64"),
        "log1p_quantity_mean": pd.Series(t_train, index=train.index),
    }
    encodings = {}
    for name, target in target_series.items():
        encodings[name] = encode_all(X_train, X_val, X_test, target)
        ridge = Ridge(alpha=RIDGE_ALPHA).fit(encodings[name]["train"], t_train)
        pred_val, pred_test = ridge.predict(encodings[name]["val"]), ridge.predict(encodings[name]["test"])
        results_models[f"te_ridge_{name}"] = scores(pred_val, pred_test)
        results_models[f"te_ridge_{name}"]["ridge_intercept"] = float(ridge.intercept_)
        results_models[f"te_ridge_{name}"]["ridge_coefficients"] = dict(zip(COLS, map(float, ridge.coef_)))
        test_preds[f"te_ridge_{name}"] = pred_test
        print(name, results_models[f"te_ridge_{name}"], flush=True)
    selected_te = min(target_series, key=lambda k: results_models[f"te_ridge_{k}"]["val"]["log1p_mae"])
    encoding = encodings[selected_te]

    # Neural-network inputs, using the selected encoding standardized on training rows.
    vocab = {c: {v: i + 1 for i, v in enumerate(pd.unique(train[c].astype("string")))} for c in COLS}
    sizes = [len(vocab[c]) for c in COLS]

    def category_ids(frame):
        columns = [frame[c].astype("string").map(vocab[c]).fillna(0).to_numpy("int64") for c in COLS]
        return np.stack(columns, axis=1)

    enc_mean, enc_std = encoding["train"].mean(axis=0), encoding["train"].std(axis=0)

    def standardize(array):
        return ((array - enc_mean) / enc_std).astype("float32")

    def to_tensor(array):
        return torch.from_numpy(array).to(device)

    data = {
        "ids_train": to_tensor(category_ids(train)), "ids_val": to_tensor(category_ids(val)),
        "ids_test": to_tensor(category_ids(test)),
        "enc_train": to_tensor(standardize(encoding["train"])), "enc_val": to_tensor(standardize(encoding["val"])),
        "enc_test": to_tensor(standardize(encoding["test"])),
        "enc_prior": to_tensor(standardize(np.full((1, 2), encoding["prior"]))),
        "t_train": to_tensor(t_train.astype("float32")), "t_val": to_tensor(t_val.astype("float32")),
    }

    # (c) embeddings, (d) hybrid, plus a target-encoding MLP that isolates the encoding
    # from the model class and loss when comparing against ridge.
    configurations = {
        "embedding_nn": (True, False),
        "hybrid_nn": (True, True),
        "te_mlp": (False, True),
    }
    histories = {}
    for name, (use_embeddings, use_te) in configurations.items():
        seed_runs = []
        for seed in ROBUSTNESS_SEEDS:
            net, summary, history = fit_network(data, sizes, use_embeddings, use_te, seed)
            pred_val = predict_network(net, data["ids_val"], data["enc_val"])
            pred_test = predict_network(net, data["ids_test"], data["enc_test"])
            metrics = scores(pred_val, pred_test)
            seed_runs.append({"seed": seed, **summary, **metrics})
            if seed == SEED:
                results_models[name] = {**summary, **metrics}
                test_preds[name] = pred_test.astype("float64")
                histories[name] = history
            print(name, seed, summary, metrics["test"], flush=True)
        values = [r["test"]["log1p_mae"] for r in seed_runs]
        results_models[name]["seed_runs"] = seed_runs
        results_models[name]["test_log1p_mae_across_seeds"] = {
            "mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)),
        }

    selected_name = f"te_ridge_{selected_te}"
    errors = {k: clipped_absolute_errors(v) for k, v in test_preds.items()}
    differences = {
        "embedding_minus_target_encoding": paired_ci(errors["embedding_nn"] - errors[selected_name]),
        "hybrid_minus_target_encoding": paired_ci(errors["hybrid_nn"] - errors[selected_name]),
        "hybrid_minus_embedding": paired_ci(errors["hybrid_nn"] - errors["embedding_nn"]),
        "te_mlp_minus_te_ridge": paired_ci(errors["te_mlp"] - errors[selected_name]),
        "embedding_minus_te_mlp": paired_ci(errors["embedding_nn"] - errors["te_mlp"]),
    }

    # Test log-MAE by how often the row's category occurred in training.
    by_frequency = {}
    plotted = {"Target encoding + ridge": selected_name, "Embeddings": "embedding_nn", "Hybrid": "hybrid_nn"}
    for col in COLS:
        counts = test[col].astype("string").map(train_counts[col]).fillna(0).to_numpy("float64")
        by_frequency[col] = []
        masks = [("unseen", counts == 0)] + [
            (label, (counts >= low) & (counts < high))
            for label, low, high in zip(COUNT_LABELS, COUNT_BINS[:-1], COUNT_BINS[1:])
        ]
        for label, mask in masks:
            row = {"train_count_bin": label, "n_test": int(mask.sum())}
            for name, key in plotted.items():
                row[name] = float(errors[key][mask].mean()) if mask.any() else None
            by_frequency[col].append(row)

    results = {
        "seed": SEED,
        "device": str(device),
        **exploration,
        "target_statistics": target_stats,
        "splits": split_stats,
        "unseen_category_fraction": unseen,
        "settings": {
            "te_min_samples_leaf": TE_MIN_SAMPLES_LEAF, "te_smoothing": TE_SMOOTHING, "ridge_alpha": RIDGE_ALPHA,
            "n_folds": N_FOLDS, "embedding_dims": dict(zip(COLS, map(embedding_dim, sizes))),
            "hidden": list(HIDDEN), "lr": LR, "weight_decay": WEIGHT_DECAY, "batch_size": BATCH_SIZE,
            "huber_beta": HUBER_BETA, "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "unk_rate": UNK_RATE,
        },
        "selected_target_encoding": selected_te,
        "models": results_models,
        "paired_log1p_mae_differences_test": differences,
        "test_log1p_mae_by_train_frequency": by_frequency,
    }
    (RESULTS_DIR / "p2.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({k: results[k] for k in ("selected_target_encoding", "paired_log1p_mae_differences_test")}, indent=2))

    # Figures.
    values, freq = np.unique(quantity, return_counts=True)
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.plot(values, freq, linestyle="none", marker="o", markersize=3, color=SERIES_COLORS[0])
    ax.set(xscale="log", yscale="log", xlabel="Quantity", ylabel="Number of rows",
           title="Quantity distribution, all rows")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "p2_target_distribution.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    labels = {"embedding_nn": "Embeddings", "hybrid_nn": "Hybrid"}
    for i, (key, label) in enumerate(labels.items()):
        epochs = np.arange(1, len(histories[key]["val"]) + 1)
        color = SERIES_COLORS[i + 1]
        ax.plot(epochs, histories[key]["train"], color=color, linestyle=":", label=f"{label}, training")
        ax.plot(epochs, histories[key]["val"], color=color, label=f"{label}, validation")
    ax.set(xlabel="Epoch", ylabel="Smooth-$L_1$ loss on $\\log(1+y)$", title="Network training curves, seed 42")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "p2_training_curves.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, col in zip(axes, COLS):
        rows = [r for r in by_frequency[col] if r["n_test"] > 0]
        xs = np.arange(len(rows))
        for i, name in enumerate(plotted):
            ax.plot(xs, [r[name] for r in rows], color=SERIES_COLORS[i], marker=SERIES_MARKERS[i], label=name)
        ax.set_xticks(xs, [f"{r['train_count_bin']}\n($n={r['n_test']}$)" for r in rows])
        ax.tick_params(axis="x", labelsize=8)
        ax.set(xlabel=f"Training rows for the row's {col}", title=col)
    axes[0].set_ylabel("Test MAE on $\\log(1+y)$")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "p2_error_by_frequency.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
