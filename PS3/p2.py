import itertools
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "winequality-white.csv")
FIG = os.path.join(HERE, "figures")
RES = os.path.join(HERE, "results")
os.makedirs(FIG, exist_ok=True)
os.makedirs(RES, exist_ok=True)
SEED = 42

def load():
    df = pd.read_csv(DATA, sep=None, engine="python")
    df.columns = [c.strip().strip('"') for c in df.columns]
    y = df["quality"].to_numpy(dtype=float)
    X = df.drop(columns="quality").to_numpy(dtype=float)
    return X, y, df

def split(X, y):
    X_tr, X_rest, y_tr, y_rest = train_test_split(X, y, train_size=0.7, random_state=SEED)
    X_va, X_te, y_va, y_te = train_test_split(X_rest, y_rest, train_size=1 / 3, random_state=SEED)
    return X_tr, X_va, X_te, y_tr, y_va, y_te

def mse(y, p):
    return float(np.mean((y - p) ** 2))

def knn_grid():
    for k, w, p in itertools.product(range(1, 151), ["uniform", "distance"], [1, 2]):
        params = dict(n_neighbors=k, weights=w, p=p)
        yield params, make_pipeline(StandardScaler(), KNeighborsRegressor(**params))

def rf_grid():
    for n, d, leaf, mf in itertools.product(
        [50, 100, 200],
        [None, 4, 8, 12, 16, 24],
        [1, 2, 4, 8, 16],
        [0.2, 0.33, 0.5, 0.75, 1.0, "sqrt"],
    ):
        params = dict(n_estimators=n, max_depth=d, min_samples_leaf=leaf, max_features=mf)
        yield params, RandomForestRegressor(**params, random_state=SEED, n_jobs=-1)

def search(name, grid, data):
    X_tr, X_va, X_te, y_tr, y_va, y_te = data
    rows = []
    t0 = time.time()
    for params, model in grid:
        model.fit(X_tr, y_tr)
        rows.append(dict(model=name, params=str(params),
                         train_mse=mse(y_tr, model.predict(X_tr)),
                         val_mse=mse(y_va, model.predict(X_va)),
                         test_mse=mse(y_te, model.predict(X_te))))
    df = pd.DataFrame(rows)
    print(f"{name}: {len(df)} configurations in {time.time() - t0:.0f}s")
    return df

def main():
    X, y, raw = load()
    data = split(X, y)
    X_tr, X_va, X_te, y_tr, y_va, y_te = data
    lines = [f"n={len(y)}, features={X.shape[1]}; train/val/test = {len(y_tr)}/{len(y_va)}/{len(y_te)}",
             f"quality counts: {raw['quality'].value_counts().sort_index().to_dict()}"]
    base = mse(y_te, np.full_like(y_te, y_tr.mean()))
    lines.append(f"baseline (train mean {y_tr.mean():.3f}): val MSE {mse(y_va, np.full_like(y_va, y_tr.mean())):.4f}, test MSE {base:.4f}")

    results = []
    for name, grid in [("KNN", knn_grid()), ("RandomForest", rf_grid())]:
        df = search(name, grid, data)
        assert len(df) > len(y_va), "grid must exceed validation-set size"
        results.append(df)
        best = df.loc[df.val_mse.idxmin()]
        oracle = df.loc[df.test_mse.idxmin()]
        rank = int((df.test_mse < best.test_mse).sum()) + 1
        r = np.corrcoef(df.val_mse, df.test_mse)[0, 1]
        rho = df.val_mse.rank().corr(df.test_mse.rank())
        lines += [
            f"--- {name} ({len(df)} configs > {len(y_va)} validation points)",
            f"selected: {best.params}",
            f"  train MSE {best.train_mse:.4f}, val MSE {best.val_mse:.4f}, test MSE {best.test_mse:.4f}, "
            f"test - val = {best.test_mse - best.val_mse:+.4f}",
            f"  rank of selected config by test MSE: {rank} of {len(df)}",
            f"best test MSE in grid (oracle, not usable): {oracle.test_mse:.4f} ({oracle.params}); its val MSE {oracle.val_mse:.4f}",
            f"across grid: Pearson corr(val, test) = {r:.3f}, Spearman = {rho:.3f}; "
            f"mean(test - val) = {(df.test_mse - df.val_mse).mean():+.4f}",
        ]

    allres = pd.concat(results)
    allres.to_csv(os.path.join(RES, "p2_grid_results.csv"), index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, df in zip(axes, results):
        name = df.model.iloc[0]
        ax.scatter(df.val_mse, df.test_mse, s=6, alpha=0.5)
        b = df.loc[df.val_mse.idxmin()]
        ax.scatter([b.val_mse], [b.test_mse], color="C3", s=40, label="selected (min validation MSE)")
        lo = min(df.val_mse.min(), df.test_mse.min())
        hi = max(df.val_mse.quantile(0.98), df.test_mse.quantile(0.98))
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="test = validation")
        ax.set_xlim(lo - 0.01, hi)
        ax.set_ylim(lo - 0.01, hi)
        ax.set_xlabel("Validation MSE")
        ax.set_ylabel("Test MSE")
        ax.set_title(f"{name}: {len(df)} configurations")
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "p2_val_vs_test.png"), dpi=150)

    text = "\n".join(lines)
    with open(os.path.join(RES, "p2_summary.txt"), "w") as f:
        f.write(text + "\n")
    print(text)

if __name__ == "__main__":
    main()
