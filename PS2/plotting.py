"""Shared plotting helpers for STAT 37711 PS2 (fitted surfaces and figure style)."""
from typing import Callable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# Fixed categorical order, validated for colour-vision deficiency; series also
# differ by marker or line style so that identity never relies on colour alone.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SERIES_MARKERS = ["o", "s", "^", "D"]

# One viewing angle for every 3D figure so that the panels can be compared.
VIEW_ELEV = 28
VIEW_AZIM = -60


def apply_style() -> None:
    """Thin marks and a recessive grid for every figure."""
    plt.rcParams.update({
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e4e3df",
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "lines.linewidth": 2,
        "lines.markersize": 6,
        "font.size": 10,
        "legend.frameon": False,
    })


def _predict_on_grid(model_fn, Xgrid: np.ndarray) -> np.ndarray:
    """Evaluate a callable or a PyTorch module on grid points, returning a 1D array."""
    if isinstance(model_fn, nn.Module):
        device = next(model_fn.parameters()).device
        with torch.no_grad():
            tensor = torch.from_numpy(Xgrid).float().to(device)
            return model_fn(tensor).cpu().numpy().reshape(-1)
    return np.asarray(model_fn(Xgrid)).reshape(-1)


def plot_surface_2d(
    model_fn: Callable[[np.ndarray], np.ndarray] | nn.Module,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    title: str = "Fitted surface",
    filename: str = "surface.png",
    grid: int = 200,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
):
    """Filled-contour plot of a fitted function; accepts model.predict or an nn.Module.

    Pass the same vmin and vmax to several calls to give the panels one colour scale.
    """
    xs = np.linspace(xlim[0], xlim[1], grid)
    ys = np.linspace(ylim[0], ylim[1], grid)
    X1, X2 = np.meshgrid(xs, ys)
    Xgrid = np.stack([X1.ravel(), X2.ravel()], axis=1)
    Z = _predict_on_grid(model_fn, Xgrid).reshape(grid, grid)

    low = Z.min() if vmin is None else vmin
    high = Z.max() if vmax is None else vmax
    fig, ax = plt.subplots(figsize=(6, 5))
    contour = ax.contourf(X1, X2, Z, levels=np.linspace(low, high, 31), cmap="viridis", extend="both")
    ax.grid(False)
    ax.set_title(title)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    # Integer ticks inside the plotted range, so that every label sits at its true value.
    ticks = np.arange(np.ceil(low), np.floor(high) + 1)
    fig.colorbar(contour, ax=ax, label="Predicted $y$", ticks=ticks)
    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.close(fig)


def plot_surface_3d(
    predict_fn,
    filename,
    title,
    grid_size: int = 150,
    xlim: Tuple[float, float] = (-3, 3),
    ylim: Tuple[float, float] = (-3, 3),
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    zlim: Optional[Tuple[float, float]] = None,
    elev: float = VIEW_ELEV,
    azim: float = VIEW_AZIM,
):
    """3D surface of a fitted function.

    Calls sharing xlim, ylim, vmin, vmax, zlim, elev and azim are directly comparable.
    """
    xs = np.linspace(xlim[0], xlim[1], grid_size)
    ys = np.linspace(ylim[0], ylim[1], grid_size)
    X1, X2 = np.meshgrid(xs, ys)
    Xgrid = np.column_stack([X1.ravel(), X2.ravel()])
    Z = _predict_on_grid(predict_fn, Xgrid).reshape(X1.shape)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    surface = ax.plot_surface(
        X1, X2, Z, cmap="viridis", vmin=vmin, vmax=vmax, linewidth=0, antialiased=True
    )
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    if zlim is not None:
        ax.set_zlim(*zlim)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_zlabel("Predicted $y$")
    ax.set_title(title)
    fig.colorbar(surface, ax=ax, shrink=0.65, pad=0.12)
    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
