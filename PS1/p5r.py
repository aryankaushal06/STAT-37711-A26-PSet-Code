import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

torch.manual_seed(0)

url = "https://archive.ics.uci.edu/static/public/844/average+localization+error+(ale)+in+sensor+node+localization+process+in+wsns.zip"
df = pd.read_csv(url)
X = df.iloc[:, :4].to_numpy()
y = df["ale"].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler().fit(X_train) 
X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

baseline = LinearRegression().fit(X_train, y_train)
print("Baseline (sklearn) test MAE:", mean_absolute_error(y_test, baseline.predict(X_test)))

X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32)

def mse_loss(y, pred):
    return ((y - pred) ** 2).mean()
student_t = torch.distributions.StudentT(df=5.0)

def t_loss(y, pred):
    return -student_t.log_prob(y - pred).mean() 

def train(loss_fn, epochs=500, lr=0.01, batch_size=16):
    w = torch.zeros(X_train_t.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.SGD([w, b], lr=lr)
    n = len(y_train_t)
    for epoch in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            loss = loss_fn(y_train_t[idx], X_train_t[idx] @ w + b)
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        mae = (y_test_t - (X_test_t @ w + b)).abs().mean().item()
    return mae

print("(a) Linear regression, MSE loss (PyTorch, SGD) test MAE:", train(mse_loss))
print("(b) Linear regression, t-loss (PyTorch, SGD) test MAE:", train(t_loss))
