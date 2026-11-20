import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
RES = os.path.join(HERE, "results")
os.makedirs(FIG, exist_ok=True)
os.makedirs(RES, exist_ok=True)

M_REPORT = [1, 10, 100, 300, 1000]
M_MAX = 1000
N_SPLITS = 20

def err_curve(model, X, y):
    return np.array([np.mean(pred != y) for pred in model.staged_predict(X)])

def run(name):
    d = pd.read_csv(os.path.join(HERE, "data", f"{name}.csv"), header=None)
    X, y = d.iloc[:, 1:].to_numpy(), d.iloc[:, 0].to_numpy().astype(int)
    train, test = [], []
    for seed in range(N_SPLITS):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
        gb = GradientBoostingClassifier(n_estimators=M_MAX, random_state=seed).fit(Xtr, ytr)
        train.append(err_curve(gb, Xtr, ytr))
        test.append(err_curve(gb, Xte, yte))
    return np.array(train), np.array(test)

def main():
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    Ms = np.arange(1, M_MAX + 1)
    for ax, name in zip(axes, ["dset0", "dset1"]):
        train, test = run(name)
        for M in M_REPORT:
            rows.append(dict(dataset=name, M=M,
                             train_err=train[:, M - 1].mean(),
                             test_err=test[:, M - 1].mean(),
                             test_err_sd=test[:, M - 1].std()))
        for arr, lab in [(train, "train"), (test, "test")]:
            m, s = arr.mean(0), arr.std(0)
            ax.plot(Ms, m, label=lab)
            ax.fill_between(Ms, m - s, m + s, alpha=0.2)
        ax.set_xscale("log")
        ax.set_xlabel("Number of trees $M$")
        ax.set_title(name)
        ax.legend()
    axes[0].set_ylabel("Classification error")
    fig.suptitle(f"Gradient boosting (depth 3, learning rate 0.1), mean $\\pm$ sd over {N_SPLITS} splits")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "p3_error_curves.png"), dpi=150)

    df = pd.DataFrame(rows).round(4)
    df.to_csv(os.path.join(RES, "p3_results.csv"), index=False)
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
