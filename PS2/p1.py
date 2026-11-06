"""STAT 37711 PS2 Problem 1: heteroscedastic Gaussian MLP for NYC taxi trip durations.

Run with: python p1.py
Writes results/p1.json and figures/p1_*.png.
"""
import copy
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats
from scipy.special import log_ndtr, ndtr
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn

from plotting import SERIES_COLORS, SERIES_MARKERS, apply_style

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
FEATURES = [
    "pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude",
    "min_of_day", "day_of_week", "day_of_year",
]
TARGET = "trip_duration"
THRESHOLD = 45.0
SIGMA_FLOOR = 1e-3
HIDDEN_CANDIDATES = [(64, 64), (128, 128), (256, 256, 256)]
LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 2048
MAX_EPOCHS = 200
PATIENCE = 15
MIN_DELTA = 1e-4
GRAD_CLIP = 10.0
LOG_LOSS_CLIP = 1e-7
LOG_SQRT_2PI = 0.5 * np.log(2 * np.pi)
# Bin edges for grouping test trips by predicted P(Y >= 45) in the reliability table.
TAIL_BIN_EDGES = [0.0, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]


def set_seeds(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    """CPU by default: on this model it measured about 8x faster than MPS and is reproducible.

    Set PS2_DEVICE=mps to use Apple Silicon; it falls back to CPU if MPS is unavailable.
    """
    if os.environ.get("PS2_DEVICE", "cpu") == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class GaussianMLP(nn.Module):
    """MLP with identity link for the mean and softplus link for the standard deviation.

    With homoscedastic=True, sigma is one learned scalar that does not depend on x.
    The output bias starts at the training mean and standard deviation of the target.
    """

    def __init__(self, hidden, y_mean, y_std, homoscedastic=False):
        super().__init__()
        layers, width = [], len(FEATURES)
        for next_width in hidden:
            layers += [nn.Linear(width, next_width), nn.ReLU()]
            width = next_width
        self.body = nn.Sequential(*layers)
        self.homoscedastic = homoscedastic
        self.out = nn.Linear(width, 1 if homoscedastic else 2)
        raw_sigma = float(np.log(np.expm1(y_std)))
        with torch.no_grad():
            self.out.bias[0] = y_mean
            if homoscedastic:
                self.raw_sigma = nn.Parameter(torch.tensor(raw_sigma))
            else:
                self.out.bias[1] = raw_sigma

    def forward(self, x):
        out = self.out(self.body(x))
        raw = self.raw_sigma if self.homoscedastic else out[:, 1]
        sigma = nn.functional.softplus(raw) + SIGMA_FLOOR
        if self.homoscedastic:
            sigma = sigma.expand(out.shape[0])
        return out[:, 0], sigma


def gaussian_nll(mu, sigma, y):
    return (torch.log(sigma) + 0.5 * ((y - mu) / sigma) ** 2 + LOG_SQRT_2PI).mean()


def predict(model, x, batch_size=65536):
    model.eval()
    mus, sigmas = [], []
    with torch.no_grad():
        for batch in x.split(batch_size):
            mu, sigma = model(batch)
            mus.append(mu.cpu())
            sigmas.append(sigma.cpu())
    return torch.cat(mus).numpy().astype("float64"), torch.cat(sigmas).numpy().astype("float64")


def fit(hidden, tensors, y_mean, y_std, homoscedastic=False):
    """Train on Gaussian NLL; stop on validation NLL and restore the best parameters."""
    set_seeds()
    xtr, ytr, xv, yv = tensors
    model = GaussianMLP(hidden, y_mean, y_std, homoscedastic).to(xtr.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    history = {"train_nll": [], "val_nll": [], "grad_norm_mean": [], "grad_norm_max": []}
    best_val, best_state, best_epoch = float("inf"), None, 0
    clipped_steps = total_steps = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        order = torch.randperm(len(ytr), device=xtr.device)
        loss_sum, norms = 0.0, []
        for batch in order.split(BATCH_SIZE):
            optimizer.zero_grad()
            mu, sigma = model(xtr[batch])
            loss = gaussian_nll(mu, sigma, ytr[batch])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite training loss at epoch {epoch}")
            loss.backward()
            norm = float(nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP))
            if not np.isfinite(norm):
                raise FloatingPointError(f"Nonfinite gradient norm at epoch {epoch}")
            optimizer.step()
            loss_sum += loss.item() * len(batch)
            norms.append(norm)
            clipped_steps += norm > GRAD_CLIP
            total_steps += 1

        mu_v, sigma_v = predict(model, xv)
        y_v = yv.cpu().numpy().astype("float64")
        val_nll = float(np.mean(np.log(sigma_v) + 0.5 * ((y_v - mu_v) / sigma_v) ** 2 + LOG_SQRT_2PI))
        history["train_nll"].append(loss_sum / len(ytr))
        history["val_nll"].append(val_nll)
        history["grad_norm_mean"].append(float(np.mean(norms)))
        history["grad_norm_max"].append(float(np.max(norms)))

        if val_nll < best_val - MIN_DELTA:
            best_val, best_epoch = val_nll, epoch
            best_state = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= PATIENCE:
            break

    model.load_state_dict(best_state)
    summary = {
        "hidden": list(hidden),
        "best_epoch": best_epoch,
        "epochs_run": len(history["val_nll"]),
        "best_val_nll": best_val,
        "fraction_steps_clipped": clipped_steps / total_steps,
        "max_grad_norm": max(history["grad_norm_max"]),
    }
    return model, summary, history


def score_probabilities(z, p, log_p=None, log_1mp=None):
    """Brier score, log-loss and accuracy at 0.5 for the event z = 1{Y < 45}.

    "log_loss" is exact. "log_loss_clipped" floors each probability at LOG_LOSS_CLIP, to
    show how much of the exact score comes from a few extremely overconfident rows.
    """
    p = np.asarray(p, dtype="float64")
    if log_p is None:
        log_p, log_1mp = np.log(p), np.log1p(-p)
    pred = p >= 0.5
    event_rows = z == 1
    row_loss = -(z * log_p + (1 - z) * log_1mp)
    floor = -np.log(LOG_LOSS_CLIP)
    return {
        "brier": float(np.mean((p - z) ** 2)),
        "log_loss": float(row_loss.mean()),
        "log_loss_clipped": float(np.minimum(row_loss, floor).mean()),
        "log_loss_median_row": float(np.median(row_loss)),
        "log_loss_max_row": float(row_loss.max()),
        "log_loss_top10_rows_share": float(np.sort(row_loss)[-10:].sum() / row_loss.sum()),
        "rows_with_clipped_loss": int((row_loss > floor).sum()),
        "accuracy": float(np.mean(pred == event_rows)),
        "predicted_at_least_45": int((~pred).sum()),
        "true_at_least_45_detected": int(((~pred) & (~event_rows)).sum()),
    }


def gaussian_probabilities(mu, sigma):
    """p(x) = Phi((45 - mu) / sigma); log_ndtr keeps both tails accurate for the log-loss."""
    t = (THRESHOLD - mu) / sigma
    return ndtr(t), log_ndtr(t), log_ndtr(-t)


def per_row_brier_and_log_loss(z, p, log_p, log_1mp):
    return (p - z) ** 2, -(z * log_p + (1 - z) * log_1mp)


def paired_ci(difference):
    """Normal-approximation 95% interval for the mean of per-row score differences."""
    half = 1.96 * difference.std(ddof=1) / np.sqrt(len(difference))
    return [float(difference.mean() - half), float(difference.mean() + half)]


def reliability_table(z, p):
    tail = 1.0 - p
    rows = []
    for low, high in zip(TAIL_BIN_EDGES[:-1], TAIL_BIN_EDGES[1:]):
        mask = (tail >= low) & (tail < high if high < 1.0 else tail <= high)
        if mask.any():
            rows.append({
                "low": low, "high": high, "n": int(mask.sum()),
                "mean_predicted_tail": float(tail[mask].mean()),
                "observed_tail": float((1 - z[mask]).mean()),
            })
    return rows


def plot_training(history, summary):
    epochs = np.arange(1, len(history["val_nll"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].plot(epochs, history["train_nll"], color=SERIES_COLORS[0], label="Training")
    axes[0].plot(epochs, history["val_nll"], color=SERIES_COLORS[1], linestyle="--", label="Validation")
    # The first epoch's training NLL is far off scale; limit the axis to show the later curve.
    later = history["train_nll"][1:] + history["val_nll"]
    low, high = min(later), max(later)
    axes[0].set_ylim(low - 0.05 * (high - low), high + 0.05 * (high - low))
    axes[0].axvline(summary["best_epoch"], color="#52514e", linewidth=1, linestyle=":")
    axes[0].annotate("Restored epoch", (summary["best_epoch"], high), xytext=(-4, -2),
                     textcoords="offset points", ha="right", va="top", color="#52514e")
    axes[0].set(xlabel="Epoch", ylabel="Gaussian NLL (nats)",
                title=f"Selected MLP loss (epoch 1 training NLL {history['train_nll'][0]:.0f} off scale)")
    axes[0].legend()
    axes[1].plot(epochs, history["grad_norm_mean"], color=SERIES_COLORS[0], label="Mean in epoch")
    axes[1].plot(epochs, history["grad_norm_max"], color=SERIES_COLORS[1], linestyle="--", label="Max in epoch")
    axes[1].axhline(GRAD_CLIP, color="#52514e", linewidth=1, linestyle=":")
    axes[1].annotate("Clip threshold", (epochs[-1], GRAD_CLIP), xytext=(0, 4),
                     textcoords="offset points", ha="right", color="#52514e")
    axes[1].set(xlabel="Epoch", ylabel="Gradient norm", title="Gradient norm before clipping", yscale="log")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "p1_training.png", dpi=200)
    plt.close(fig)


def plot_calibration(tables):
    linthresh = 1e-4
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    ax.plot([0, 1], [0, 1], color="#52514e", linewidth=1, linestyle=":", label="Perfect calibration")
    for i, (name, rows) in enumerate(tables.items()):
        ax.plot([r["mean_predicted_tail"] for r in rows], [r["observed_tail"] for r in rows],
                color=SERIES_COLORS[i], marker=SERIES_MARKERS[i], label=name)
    ax.set_xscale("symlog", linthresh=linthresh)
    ax.set_yscale("symlog", linthresh=linthresh)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted $P(Y\\geq 45)$ in bin",
           ylabel="Observed frequency of $Y\\geq 45$ in bin", title="Test-set reliability, tail event")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "p1_calibration.png", dpi=200)
    plt.close(fig)


def plot_predictions(y, mu, sigma):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    limit = 120
    hb = axes[0].hexbin(mu, y, gridsize=70, extent=(0, limit, 0, limit), bins="log", cmap="viridis", mincnt=1)
    axes[0].plot([0, limit], [0, limit], color="#52514e", linewidth=1, linestyle=":")
    axes[0].axhline(THRESHOLD, color="#eb6834", linewidth=1)
    axes[0].grid(False)
    axes[0].set(xlabel="Predicted mean $\\hat\\mu(x)$ (minutes)", ylabel="Observed duration (minutes)",
                title="Observed vs predicted mean (durations up to 120 min)")
    fig.colorbar(hb, ax=axes[0], label="Trips (log scale)")
    hb = axes[1].hexbin(mu, sigma, gridsize=70, extent=(0, np.ceil(mu.max() / 10) * 10, 0, np.ceil(sigma.max() / 5) * 5), bins="log", cmap="viridis", mincnt=1)
    axes[1].grid(False)
    axes[1].set(xlabel="Predicted mean $\\hat\\mu(x)$ (minutes)", ylabel="Predicted SD $\\hat\\sigma(x)$ (minutes)",
                title="Predicted noise vs predicted mean")
    fig.colorbar(hb, ax=axes[1], label="Trips (log scale)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "p1_pred_vs_obs.png", dpi=200)
    plt.close(fig)


def plot_pit(y, mu, sigma):
    pit = ndtr((y - mu) / sigma)
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.hist(pit, bins=20, range=(0, 1), density=True, color=SERIES_COLORS[0], edgecolor="white", linewidth=1)
    ax.axhline(1, color="#52514e", linewidth=1, linestyle=":")
    ax.set(xlabel="$\\Phi((y-\\hat\\mu(x))/\\hat\\sigma(x))$", ylabel="Density",
           title="Probability integral transform, test set")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "p1_pit.png", dpi=200)
    plt.close(fig)


def main():
    set_seeds()
    torch.set_num_threads(4)
    apply_style()
    device = get_device()

    df = pd.read_parquet(DATA_DIR / "taxi_trips.parquet")
    missing = sorted(set(FEATURES + [TARGET]) - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns in taxi_trips.parquet: {missing}")
    schema = {
        "rows": len(df),
        "columns": {c: str(t) for c, t in df.dtypes.items()},
        "missing_values": int(df[FEATURES + [TARGET]].isna().sum().sum()),
        "nonfinite_values": int((~np.isfinite(df[FEATURES + [TARGET]].to_numpy("float64"))).sum()),
        "nonpositive_durations": int((df[TARGET] <= 0).sum()),
        "duration_summary_minutes": {
            k: float(v) for k, v in df[TARGET].describe(percentiles=[0.5, 0.95, 0.99, 0.999]).items()
        },
        "fraction_under_45": float((df[TARGET] < THRESHOLD).mean()),
    }
    if schema["missing_values"] or schema["nonfinite_values"] or schema["nonpositive_durations"]:
        raise ValueError(f"Unexpected invalid rows: {schema}")

    X = df[FEATURES].to_numpy("float32")
    y = df[TARGET].to_numpy("float32")
    idx_train, idx_held = train_test_split(np.arange(len(y)), test_size=0.30, random_state=SEED)
    idx_val, idx_test = train_test_split(idx_held, test_size=0.50, random_state=SEED)

    # The supplied columns were standardized on the whole file; this extra affine map
    # uses training rows only, so the held-out rows contribute nothing further.
    scaler = StandardScaler().fit(X[idx_train])
    X = scaler.transform(X).astype("float32")

    def to_tensor(array):
        return torch.from_numpy(array).to(device)

    tensors = (to_tensor(X[idx_train]), to_tensor(y[idx_train]), to_tensor(X[idx_val]), to_tensor(y[idx_val]))
    y_train = y[idx_train].astype("float64")
    y_test = y[idx_test].astype("float64")
    y_mean, y_std = float(y_train.mean()), float(y_train.std())

    # Architecture selection uses validation NLL only; test rows are untouched here.
    sweep, candidates = [], {}
    for hidden in HIDDEN_CANDIDATES:
        model, summary, history = fit(hidden, tensors, y_mean, y_std)
        sweep.append(summary)
        candidates[hidden] = (model, summary, history)
        print(f"hidden={hidden} {summary}", flush=True)
    best_hidden = tuple(min(sweep, key=lambda s: s["best_val_nll"])["hidden"])
    model, summary, history = candidates[best_hidden]
    homo_model, homo_summary, _ = fit(best_hidden, tensors, y_mean, y_std, homoscedastic=True)
    print(f"homoscedastic {homo_summary}", flush=True)

    # Final test evaluation, on one common set of test rows.
    z = (y_test < THRESHOLD).astype("float64")
    xt = to_tensor(X[idx_test])
    mu, sigma = predict(model, xt)
    mu_h, sigma_h = predict(homo_model, xt)
    if not (np.isfinite(mu).all() and np.isfinite(sigma).all() and (sigma > 0).all()):
        raise FloatingPointError("Nonfinite or nonpositive test predictions")

    gaussians = {
        "heteroscedastic_mlp": (mu, sigma),
        "homoscedastic_mlp": (mu_h, sigma_h),
        "constant_gaussian": (np.full_like(y_test, y_mean), np.full_like(y_test, y_std)),
    }
    metrics, per_row, probs = {}, {}, {}
    for name, (m, s) in gaussians.items():
        p, log_p, log_1mp = gaussian_probabilities(m, s)
        metrics[name] = score_probabilities(z, p, log_p, log_1mp)
        row_nll = np.log(s) + 0.5 * ((y_test - m) / s) ** 2 + LOG_SQRT_2PI
        metrics[name]["gaussian_nll"] = float(row_nll.mean())
        metrics[name]["gaussian_nll_median_row"] = float(np.median(row_nll))
        metrics[name]["gaussian_nll_max_row"] = float(row_nll.max())
        # Mean after dropping the worst 0.1% of rows, to separate typical fit from outliers.
        metrics[name]["gaussian_nll_trimmed_top_0.1pct"] = float(
            np.sort(row_nll)[: int(np.ceil(0.999 * len(row_nll)))].mean()
        )
        per_row[name] = per_row_brier_and_log_loss(z, p, log_p, log_1mp)
        probs[name] = p
    p0 = float((y_train < THRESHOLD).mean())
    p_const = np.full_like(y_test, p0)
    metrics["constant_probability"] = score_probabilities(z, p_const)
    metrics["constant_probability"]["gaussian_nll"] = None
    per_row["constant_probability"] = per_row_brier_and_log_loss(z, p_const, np.log(p_const), np.log1p(-p_const))
    probs["constant_probability"] = p_const

    # Cross-check the scoring code against scikit-learn. The clipped log-loss is the one
    # comparable to sklearn, which cannot represent probabilities below machine precision.
    clipped = np.clip(probs["heteroscedastic_mlp"], LOG_LOSS_CLIP, 1 - LOG_LOSS_CLIP)
    assert abs(brier_score_loss(z, probs["heteroscedastic_mlp"]) - metrics["heteroscedastic_mlp"]["brier"]) < 1e-12
    assert abs(log_loss(z, p_const, labels=[0, 1]) - metrics["constant_probability"]["log_loss"]) < 1e-9
    assert abs(log_loss(z, clipped, labels=[0, 1]) - metrics["heteroscedastic_mlp"]["log_loss_clipped"]) < 1e-4

    reference = metrics["constant_probability"]
    for name in metrics:
        metrics[name]["brier_skill_vs_constant_probability"] = 1 - metrics[name]["brier"] / reference["brier"]
        metrics[name]["log_loss_skill_vs_constant_probability"] = 1 - metrics[name]["log_loss"] / reference["log_loss"]
        metrics[name]["log_loss_clipped_skill_vs_constant_probability"] = (
            1 - metrics[name]["log_loss_clipped"] / reference["log_loss_clipped"]
        )

    comparisons = {}
    for name in ["homoscedastic_mlp", "constant_gaussian", "constant_probability"]:
        comparisons[f"heteroscedastic_mlp_minus_{name}"] = {
            "brier_ci95": paired_ci(per_row["heteroscedastic_mlp"][0] - per_row[name][0]),
            "log_loss_ci95": paired_ci(per_row["heteroscedastic_mlp"][1] - per_row[name][1]),
            "log_loss_clipped_ci95": paired_ci(
                np.minimum(per_row["heteroscedastic_mlp"][1], -np.log(LOG_LOSS_CLIP))
                - np.minimum(per_row[name][1], -np.log(LOG_LOSS_CLIP))
            ),
        }

    # The same outlier breakdown on validation rows, to explain any validation-test gap.
    mu_v, sigma_v = predict(model, tensors[2])
    y_val = y[idx_val].astype("float64")
    val_row_nll = np.log(sigma_v) + 0.5 * ((y_val - mu_v) / sigma_v) ** 2 + LOG_SQRT_2PI
    val_nll_breakdown = {
        "mean": float(val_row_nll.mean()),
        "median_row": float(np.median(val_row_nll)),
        "max_row": float(val_row_nll.max()),
        "trimmed_top_0.1pct": float(np.sort(val_row_nll)[: int(np.ceil(0.999 * len(val_row_nll)))].mean()),
    }

    residual = (y_test - mu) / sigma
    worst = np.argsort(-np.abs(residual))[:5]
    worst_rows = [
        {"observed": float(y_test[i]), "mu": float(mu[i]), "sigma": float(sigma[i]), "z": float(residual[i])}
        for i in worst
    ]
    coverage = {
        str(level): float(np.mean(np.abs(residual) <= stats.norm.ppf(0.5 + level / 2)))
        for level in (0.5, 0.8, 0.9, 0.95, 0.99)
    }
    diagnostics = {
        "sigma_quantiles_test": {
            str(q): float(np.quantile(sigma, q)) for q in (0, 0.01, 0.5, 0.99, 1)
        },
        "standardized_residual_mean": float(residual.mean()),
        "standardized_residual_std": float(residual.std()),
        "standardized_residual_skew": float(stats.skew(residual)),
        "standardized_residual_excess_kurtosis": float(stats.kurtosis(residual)),
        "central_interval_coverage": coverage,
        "mean_prediction_rmse_minutes": float(np.sqrt(np.mean((y_test - mu) ** 2))),
        "mean_prediction_mae_minutes": float(np.mean(np.abs(y_test - mu))),
        "constant_mean_rmse_minutes": float(np.sqrt(np.mean((y_test - y_mean) ** 2))),
        "constant_mean_mae_minutes": float(np.mean(np.abs(y_test - y_mean))),
        "fraction_negative_mu": float(np.mean(mu < 0)),
        "worst_test_rows": worst_rows,
        "val_nll_breakdown": val_nll_breakdown,
    }
    reliability = {
        "Heteroscedastic MLP": reliability_table(z, probs["heteroscedastic_mlp"]),
        "Homoscedastic MLP": reliability_table(z, probs["homoscedastic_mlp"]),
    }

    results = {
        "seed": SEED,
        "device": str(device),
        "schema": schema,
        "n_train": len(idx_train), "n_val": len(idx_val), "n_test": len(idx_test),
        "test_fraction_under_45": float(z.mean()),
        "test_trips_at_least_45": int((1 - z).sum()),
        "training_settings": {
            "lr": LR, "weight_decay": WEIGHT_DECAY, "batch_size": BATCH_SIZE, "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE, "min_delta": MIN_DELTA, "grad_clip": GRAD_CLIP, "sigma_floor": SIGMA_FLOOR,
            "log_loss_clip": LOG_LOSS_CLIP,
        },
        "architecture_sweep": sweep,
        "selected_hidden": list(best_hidden),
        "selected_model": summary,
        "homoscedastic_model": homo_summary,
        "baseline_parameters": {"p0": p0, "constant_mu": y_mean, "constant_sigma": y_std},
        "test_metrics": metrics,
        "paired_differences_test": comparisons,
        "diagnostics": diagnostics,
        "reliability_test": reliability,
    }
    (RESULTS_DIR / "p1.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))

    plot_training(history, summary)
    plot_calibration(reliability)
    plot_predictions(y_test, mu, sigma)
    plot_pit(y_test, mu, sigma)


if __name__ == "__main__":
    main()
