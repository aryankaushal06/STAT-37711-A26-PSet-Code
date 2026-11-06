"""STAT 37711 PS2 Problem 3: Random Kitchen Sinks versus a one-hidden-layer MLP.

Run with: python p3.py
Writes results/p3.json, figures/p3_*.png, rks_test_predictions.csv and mlp_test_predictions.csv.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial.distance import pdist
from torch import nn

from dataloader import HW2Dataset
from plotting import SERIES_COLORS, VIEW_AZIM, VIEW_ELEV, apply_style, plot_surface_2d, plot_surface_3d

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
ROBUSTNESS_SEEDS = [42, 43, 44, 45, 46]
BOUNDS = (-3, 3)

D_VALUES = [10, 20, 50, 100, 200, 400, 800, 1600]
RIDGE_LAMBDA = 1.0
SENSITIVITY_SIGMAS = [0.5, 1.0, 2.0]  # plus the median-heuristic value
SENSITIVITY_LAMBDAS = [0.01, 0.1, 1.0, 10.0]

WIDTHS = [8, 16, 32, 64, 128]
MLP_LR = 0.01
MLP_EPOCHS = 5000
LEGACY_EPOCHS = 300


def mse(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))


class RKS:
    """Random Kitchen Sinks with ridge regression; D = 2h features as in the assignment.

    Directions come from one seeded draw of size h_max truncated to h, so a larger D
    adds features to a smaller D's set and the sweep measures D rather than a new draw.
    """

    def __init__(self, D, sigma, ridge_lambda=RIDGE_LAMBDA, seed=SEED, h_max=max(D_VALUES) // 2):
        if D % 2:
            raise ValueError("D = 2h must be even")
        self.D, self.h, self.sigma, self.ridge_lambda = D, D // 2, sigma, ridge_lambda
        rng = np.random.default_rng(seed)
        self.w = rng.normal(size=(h_max, 2))[: self.h] / sigma  # w_j ~ N(0, I / sigma^2)
        self.b = rng.uniform(0, 2 * np.pi, size=h_max)[: self.h]  # b_j ~ Unif[0, 2 pi]

    def features(self, X):
        """z = (sin_1, cos_1, ..., sin_h, cos_h), with no additional rescaling."""
        phase = X @ self.w.T + self.b
        Z = np.empty((X.shape[0], self.D))
        Z[:, 0::2] = np.sin(phase)
        Z[:, 1::2] = np.cos(phase)
        return Z

    def fit(self, X, y):
        Z = self.features(X)
        self.intercept = y.mean()
        system = Z.T @ Z + self.ridge_lambda * np.eye(self.D)
        self.coef = np.linalg.solve(system, Z.T @ (y - self.intercept))
        return self

    def predict(self, X):
        return self.intercept + self.features(X) @ self.coef


class MLP(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, width),
            nn.Tanh(),
            nn.Linear(width, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def fit_mlp(width, X, y, epochs=MLP_EPOCHS, lr=MLP_LR, cosine=True, seed=SEED):
    """Full-batch Adam on training MSE for a fixed number of epochs; validation is not consulted.

    Runs on CPU: with 600 rows and full batches, MPS offers no benefit.
    """
    torch.manual_seed(seed)
    model = MLP(width)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs) if cosine else None
    xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = ((model(xt) - yt) ** 2).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite MLP loss at epoch {epoch}")
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
    model.eval()
    return model


def predict_mlp(model, X):
    with torch.no_grad():
        return model(torch.as_tensor(X, dtype=torch.float32)).numpy().astype("float64")


def plot_sweep(xs, ys, xlabel, ylabel, title, filename, best=None, log_x=False, log_y=False):
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.plot(xs, ys, marker="o", color=SERIES_COLORS[0])
    if best is not None:
        i = xs.index(best)
        ax.annotate(f"Selected: {best}", (xs[i], ys[i]), xytext=(0, 10),
                    textcoords="offset points", ha="center", color="#52514e")
    if log_x:
        ax.set_xscale("log")
        ax.set_xticks(xs, [str(x) for x in xs])
        ax.minorticks_off()
    if log_y:
        ax.set_yscale("log")
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    fig.tight_layout()
    fig.savefig(filename, dpi=200)
    plt.close(fig)


def verify_outputs():
    """Check prediction files against test.csv and reject empty or blank figures."""
    test = np.loadtxt(DATA_DIR / "test.csv", delimiter=",", skiprows=1)
    assert test.shape == (200, 2)
    for filename in ["rks_test_predictions.csv", "mlp_test_predictions.csv"]:
        predictions = np.loadtxt(ROOT / filename, delimiter=",")
        assert predictions.shape == (200,)
        assert np.isfinite(predictions).all()
        print(filename, "PASS")
    expected = [
        "p3_train.png", "p3_train_3d.png", "p3_rks_train.png", "p3_rks_val.png", "p3_mlp_train.png",
        "p3_mlp_val.png", "p3_rks_surface.png", "p3_mlp_surface.png", "p3_rks_surface_3d.png",
        "p3_mlp_surface_3d.png",
    ]
    for name in expected:
        image = plt.imread(FIG_DIR / name)
        assert image.shape[0] > 100 and image.shape[1] > 100 and image[..., :3].std() > 0.01, name
        print(name, image.shape, "PASS")


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    apply_style()

    ds = HW2Dataset.from_csv(DATA_DIR / "train.csv", DATA_DIR / "val.csv", DATA_DIR / "test.csv")
    X, y = ds.train.X, ds.train.y
    XV, yV = ds.val.X, ds.val.y
    XT = ds.test.X
    assert X.shape == (600, 2) and XV.shape == (200, 2) and XT.shape == (200, 2)
    assert max(np.abs(a).max() for a in (X, XV, XT)) <= 3

    # Median heuristic: a kernel scale fixed from training inputs alone, before any validation.
    sigma = float(np.median(pdist(X)))

    # (b) RKS sweep over D = 2h; validation MSE selects D.
    rks_records = []
    for D in D_VALUES:
        model = RKS(D, sigma).fit(X, y)
        across = [mse(yV, RKS(D, sigma, seed=s).fit(X, y).predict(XV)) for s in ROBUSTNESS_SEEDS]
        rks_records.append({
            "D": D, "train_mse": mse(y, model.predict(X)), "val_mse": mse(yV, model.predict(XV)),
            "val_mse_seed_mean": float(np.mean(across)), "val_mse_seed_std": float(np.std(across, ddof=1)),
        })
    best_D = min(rks_records, key=lambda r: r["val_mse"])["D"]

    # (c) MLP sweep over the hidden width; validation MSE selects the width.
    mlp_records = []
    for width in WIDTHS:
        model = fit_mlp(width, X, y)
        across = [mse(yV, predict_mlp(fit_mlp(width, X, y, seed=s), XV)) for s in ROBUSTNESS_SEEDS]
        short = fit_mlp(width, X, y, epochs=LEGACY_EPOCHS, cosine=False)
        mlp_records.append({
            "width": width, "train_mse": mse(y, predict_mlp(model, X)), "val_mse": mse(yV, predict_mlp(model, XV)),
            "val_mse_seed_mean": float(np.mean(across)), "val_mse_seed_std": float(np.std(across, ddof=1)),
            "train_mse_300_epochs_constant_lr": mse(y, predict_mlp(short, X)),
            "val_mse_300_epochs_constant_lr": mse(yV, predict_mlp(short, XV)),
        })
        print(mlp_records[-1], flush=True)
    best_width = min(mlp_records, key=lambda r: r["val_mse"])["width"]

    # (d) Refit at the selected values and compare.
    rks = RKS(best_D, sigma).fit(X, y)
    mlp = fit_mlp(best_width, X, y)
    rks_val_pred, mlp_val_pred = rks.predict(XV), predict_mlp(mlp, XV)
    rks_val, mlp_val = mse(yV, rks_val_pred), mse(yV, mlp_val_pred)
    assert abs(rks_val - next(r["val_mse"] for r in rks_records if r["D"] == best_D)) < 1e-12
    assert abs(mlp_val - next(r["val_mse"] for r in mlp_records if r["width"] == best_width)) < 1e-6

    difference = (yV - rks_val_pred) ** 2 - (yV - mlp_val_pred) ** 2
    half = 1.96 * difference.std(ddof=1) / np.sqrt(len(difference))

    # Sensitivity to the two fixed RKS settings; reported only, not used for selection.
    sensitivity = [
        {"sigma": s, "lambda": lam,
         "val_mse_seed_mean": float(np.mean(
             [mse(yV, RKS(best_D, s, lam, seed=k).fit(X, y).predict(XV)) for k in ROBUSTNESS_SEEDS]))}
        for s in SENSITIVITY_SIGMAS + [sigma] for lam in SENSITIVITY_LAMBDAS
    ]

    np.savetxt(ROOT / "rks_test_predictions.csv", rks.predict(XT), delimiter=",", fmt="%.10f")
    np.savetxt(ROOT / "mlp_test_predictions.csv", predict_mlp(mlp, XT), delimiter=",", fmt="%.10f")

    # (a) Training-data figures.
    fig, ax = plt.subplots(figsize=(6, 5))
    points = ax.scatter(X[:, 0], X[:, 1], c=y, s=16, cmap="viridis")
    ax.grid(False)
    fig.colorbar(points, ax=ax, label="$y$")
    ax.set(xlim=BOUNDS, ylim=BOUNDS, xlabel="$x_1$", ylabel="$x_2$", title="Training observations")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "p3_train.png", dpi=200)
    plt.close(fig)

    # Sweep figures.
    d_axis = [r["D"] for r in rks_records]
    w_axis = [r["width"] for r in mlp_records]
    d_label, w_label = "Number of random features $D=2h$", "Hidden width"
    plot_sweep(d_axis, [r["train_mse"] for r in rks_records], d_label, "Training MSE", "RKS training MSE",
               FIG_DIR / "p3_rks_train.png", log_x=True, log_y=True)
    plot_sweep(d_axis, [r["val_mse"] for r in rks_records], d_label, "Validation MSE", "RKS validation MSE",
               FIG_DIR / "p3_rks_val.png", best=best_D, log_x=True, log_y=True)
    plot_sweep(w_axis, [r["train_mse"] for r in mlp_records], w_label, "Training MSE", "MLP training MSE",
               FIG_DIR / "p3_mlp_train.png", log_x=True)
    plot_sweep(w_axis, [r["val_mse"] for r in mlp_records], w_label, "Validation MSE", "MLP validation MSE",
               FIG_DIR / "p3_mlp_val.png", best=best_width, log_x=True)

    # (d) Fitted surfaces on one colour scale, vertical range and viewing angle.
    def predict_selected_mlp(points):
        return predict_mlp(mlp, points)

    axis = np.linspace(*BOUNDS, 150)
    G = np.column_stack([a.ravel() for a in np.meshgrid(axis, axis)])
    grid_rks, grid_mlp = rks.predict(G), predict_selected_mlp(G)
    low = float(min(grid_rks.min(), grid_mlp.min(), y.min()))
    high = float(max(grid_rks.max(), grid_mlp.max(), y.max()))
    common = {"vmin": low, "vmax": high}
    limits = {**common, "zlim": (low, high)}

    plot_surface_2d(rks.predict, BOUNDS, BOUNDS, "RKS fitted surface", FIG_DIR / "p3_rks_surface.png", **common)
    plot_surface_2d(predict_selected_mlp, BOUNDS, BOUNDS, "MLP fitted surface", FIG_DIR / "p3_mlp_surface.png", **common)
    plot_surface_3d(rks.predict, FIG_DIR / "p3_rks_surface_3d.png", "RKS fitted surface", **limits)
    plot_surface_3d(predict_selected_mlp, FIG_DIR / "p3_mlp_surface_3d.png", "MLP fitted surface", **limits)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    points = ax.scatter(X[:, 0], X[:, 1], y, c=y, s=12, cmap="viridis", vmin=low, vmax=high)
    ax.set_xlim(*BOUNDS)
    ax.set_ylim(*BOUNDS)
    ax.set_zlim(low, high)
    ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_zlabel("$y$")
    ax.set_title("Training observations")
    fig.colorbar(points, ax=ax, shrink=0.65, pad=0.12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "p3_train_3d.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    results = {
        "seed": SEED, "n_train": len(X), "n_val": len(XV), "n_test": len(XT),
        "response_variance_train": float(y.var()),
        "rks_settings": {"sigma_median_heuristic": sigma, "ridge_lambda": RIDGE_LAMBDA},
        "mlp_settings": {"optimizer": "Adam", "lr": MLP_LR, "schedule": "cosine", "epochs": MLP_EPOCHS,
                         "activation": "tanh", "device": "cpu"},
        "rks_sweep": rks_records, "mlp_sweep": mlp_records,
        "best_rks_D": best_D, "best_mlp_width": best_width,
        "rks_train_mse": mse(y, rks.predict(X)), "mlp_train_mse": mse(y, predict_mlp(mlp, X)),
        "rks_val_mse": rks_val, "mlp_val_mse": mlp_val,
        "val_mse_rks_minus_mlp": {"mean": float(difference.mean()),
                                  "ci95": [float(difference.mean() - half), float(difference.mean() + half)]},
        "surface_mean_abs_difference": float(np.mean(np.abs(grid_rks - grid_mlp))),
        "surface_max_abs_difference": float(np.max(np.abs(grid_rks - grid_mlp))),
        "surface_max_difference_location": G[np.argmax(np.abs(grid_rks - grid_mlp))].tolist(),
        "surface_argmin": {"rks": G[np.argmin(grid_rks)].tolist(), "mlp": G[np.argmin(grid_mlp)].tolist()},
        "surface_argmax": {"rks": G[np.argmax(grid_rks)].tolist(), "mlp": G[np.argmax(grid_mlp)].tolist()},
        "surface_range": {"rks": [float(grid_rks.min()), float(grid_rks.max())],
                          "mlp": [float(grid_mlp.min()), float(grid_mlp.max())]},
        "rks_sensitivity_at_selected_D": sensitivity,
    }
    (RESULTS_DIR / "p3.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    verify_outputs()


if __name__ == "__main__":
    main()
