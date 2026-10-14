import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

url = "https://archive.ics.uci.edu/static/public/844/average+localization+error+(ale)+in+sensor+node+localization+process+in+wsns.zip"
df = pd.read_csv(url)
X = df.iloc[:, :4]   # anchor_ratio, trans_range, node_density, iterations
y = df["ale"]

models = {
    "Linear Regression": LinearRegression(),
    "SVR": make_pipeline(StandardScaler(), SVR()),
    "Decision Tree": DecisionTreeRegressor(max_depth=4),
    "MLP Regressor": make_pipeline(StandardScaler(), MLPRegressor(max_iter=5000)),
}

fig, axes = plt.subplots(1, len(models), figsize=(16, 4))
for ax, (name, model) in zip(axes, models.items()):
    model.fit(X, y)
    residual = y - model.predict(X)
    stats.probplot(residual, dist="norm", plot=ax)
    ax.set_title(name)
    stat, p = stats.normaltest(residual)
    print(name, "p-value:", p)

plt.tight_layout()
plt.show()
