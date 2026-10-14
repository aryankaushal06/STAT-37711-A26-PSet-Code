import torch
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

torch.manual_seed(0)

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler().fit(X_train)
X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

baseline = LogisticRegression(max_iter=1000).fit(X_train, y_train)
print("Baseline (sklearn) accuracy:", baseline.score(X_test, y_test))

X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32)

normal = torch.distributions.Normal(0.0, 1.0)
eps = 1e-7

def bce_loss(y, p):
    p = torch.clamp(p, eps, 1 - eps)
    return -(y * torch.log(p) + (1 - y) * torch.log(1 - p)).mean()

def train(link, epochs=200, lr=0.1, batch_size=32):
    w = torch.zeros(X_train_t.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.SGD([w, b], lr=lr)
    n = len(y_train_t)
    for epoch in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            p = link(X_train_t[idx] @ w + b)
            loss = bce_loss(y_train_t[idx], p)
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        p_test = link(X_test_t @ w + b)
        acc = ((p_test >= 0.5).float() == y_test_t).float().mean().item()
    return acc

print("(a) Logistic regression (PyTorch, SGD) accuracy:", train(torch.sigmoid))
print("(b) Probit regression (PyTorch, SGD) accuracy:", train(normal.cdf))
