import itertools
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)
SEED = 42

def load():
    # The UCI release is ';'-separated with quoted headers
    df = pd.read_csv(os.path.join(HERE, "data", "winequality-white.csv"), sep=None, engine="python")
    df.columns = [c.strip().strip('"') for c in df.columns]
    return df.drop(columns="quality").to_numpy(dtype=float), df["quality"].to_numpy(dtype=float)

def mse(y, p):
    return float(np.mean((y - p) ** 2))

def knn_grid():
    # KNN on standardized features (distance-based, so scale matters)
    for k, w, p in itertools.product(range(1, 151), ["uniform", "distance"], [1, 2]):
        params = dict(n_neighbors=k, weights=w, p=p)
        yield params, make_pipeline(StandardScaler(), KNeighborsRegressor(**params))

def rf_grid():
    for n, d, leaf, mf in itertools.product(
        [50, 100, 200],
        [None, 4, 8, 12, 16, 24],
        [1, 2, 4, 8, 16],
        [0.2, 0.33, 0.5, 0.75, 1.0, "sqrt"],):
        params = dict(n_estimators=n, max_depth=d, min_samples_leaf=leaf, max_features=mf)
        yield params, RandomForestRegressor(**params, random_state=SEED, n_jobs=-1)

def main():
    X, y = load()
    X_tr, X_rest, y_tr, y_rest = train_test_split(X, y, train_size=0.7, random_state=SEED)
    X_va, X_te, y_va, y_te = train_test_split(X_rest, y_rest, train_size=1 / 3, random_state=SEED)
    lines = [f"train/val/test = {len(y_tr)}/{len(y_va)}/{len(y_te)}"]
    m = y_tr.mean()
    lines.append(f"baseline (train mean {m:.3f}): val MSE {mse(y_va, np.full_like(y_va, m)):.4f}, "
                 f"test MSE {mse(y_te, np.full_like(y_te, m)):.4f}")
    rows = []
    for name, grid in [("KNN", knn_grid()), ("RandomForest", rf_grid())]:
        best_val, best_params, best_model, n_configs = np.inf, None, None, 0
        for params, model in grid:
            model.fit(X_tr, y_tr)
            v = mse(y_va, model.predict(X_va))
            rows.append(dict(model=name, params=str(params), val_mse=v))
            n_configs += 1
            if v < best_val:
                best_val, best_params, best_model = v, params, model
        assert n_configs > len(y_va), "grid must exceed validation-set size"
        lines.append(f"{name} ({n_configs} configs): selected {best_params}; "
                     f"train MSE {mse(y_tr, best_model.predict(X_tr)):.4f}, val MSE {best_val:.4f}, "
                     f"test MSE {mse(y_te, best_model.predict(X_te)):.4f}")

    pd.DataFrame(rows).to_csv(os.path.join(RES, "p2_grid_results.csv"), index=False)
    text = "\n".join(lines)
    with open(os.path.join(RES, "p2_summary.txt"), "w") as f:
        f.write(text + "\n")
    print(text)

if __name__ == "__main__":
    main()
