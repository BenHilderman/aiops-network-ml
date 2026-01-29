"""
AIOps Lab - Network Metrics ML Demo

Generates fake network data and demonstrates:
1. Basic ML (regression, classification, clustering)
2. EDA with Pandas/NumPy + plots
3. PyTorch neural network regression

Install: pip install numpy pandas scikit-learn matplotlib seaborn torch
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import root_mean_squared_error, accuracy_score

import torch
import torch.nn as nn
import torch.optim as optim


def generate_fake_network_data(n_samples=500):
    """
    Create fake network telemetry data.

    Returns DataFrame with:
    - traffic_load: network utilization (0-100%)
    - packet_loss: packet loss percentage (0-5%)
    - latency_ms: network latency in milliseconds
    - high_latency: 1 if latency > 40ms, else 0
    """
    rng = np.random.default_rng(42) # starting value to base rng off of for consistency

    traffic_load = rng.uniform(0, 100, n_samples)

    # Packet loss increases with traffic load
    packet_loss = 0.1 + 0.02 * (traffic_load / 100) + rng.normal(0, 0.05, n_samples) # packet loss formula (Traffic load -> packet loss)
    packet_loss = np.clip(packet_loss, 0, 5) # Range: 0 or 5 (Realistic)

    # Latency depends on load and packet loss
    latency_ms = 10 + 0.2 * traffic_load + 5.0 * packet_loss + rng.normal(0, 5, n_samples)

    high_latency = (latency_ms > 40).astype(int) # Ciena likely has lower threshold (Known for low latency solutions)

    return pd.DataFrame({
        "traffic_load": traffic_load,
        "packet_loss": packet_loss,
        "latency_ms": latency_ms,
        "high_latency": high_latency,
    })


def block1_basic_ml(df):
    """Regression, classification, and clustering."""
    print("\n===== BLOCK 1: BASIC ML =====")

    features = df[["traffic_load", "packet_loss"]].values

    # Regression: predict latency from load and packet loss
    print("\n--- Regression: predict latency_ms ---")
    
    X_train, X_test, y_train, y_test = train_test_split(
        features, df["latency_ms"].values, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, predictions)

    print(f"RMSE: {rmse:.2f} ms")
    print(f"Coefficients (load, loss): {model.coef_}") # How much load/loss affect prediction
    print(f"Intercept: {model.intercept_:.2f}") # baseline predict when input is 0

    # Classification: predict high vs low latency
    print("\n--- Classification: predict high_latency ---")
    labels = df["high_latency"].values
    X_train, X_test, y_train, y_test = train_test_split(  # category
        features, labels, test_size=0.2, random_state=42, stratify=labels.tolist()
    )

    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, classifier.predict(X_test))
    print(f"Accuracy: {accuracy:.3f}")

    # Clustering: group data points by behavior
    print("\n--- Clustering: KMeans ---")
    cluster_features = df[["traffic_load", "latency_ms"]].values
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(cluster_features)

    print("Cluster centers (load, latency):")
    for i, center in enumerate(kmeans.cluster_centers_):
        print(f"  Cluster {i}: load={center[0]:.1f}, latency={center[1]:.1f}")

    # Plot clusters
    plt.figure(figsize=(6, 4))
    plt.scatter(df["traffic_load"], df["latency_ms"], c=df["cluster"], cmap="viridis", s=15, alpha=0.7)
    plt.title("KMeans Clusters")
    plt.xlabel("Traffic Load (%)")
    plt.ylabel("Latency (ms)")
    plt.tight_layout()
    plt.show()


def block2_eda(df):
    """Exploratory data analysis with Pandas and plots."""
    print("\n===== BLOCK 2: EDA =====")

    print("\n--- Data Preview ---")
    print(df.head())

    print("\n--- Statistics ---")
    print(df.describe())

    print("\n--- Correlations ---")
    print(df.corr(numeric_only=True))

    # Latency distribution
    plt.figure(figsize=(6, 4))
    sns.histplot(df, x="latency_ms", bins=30, kde=True)
    plt.title("Latency Distribution")
    plt.tight_layout()
    plt.show()

    # Load vs latency scatter plot
    plt.figure(figsize=(6, 4))
    sns.scatterplot(df, x="traffic_load", y="latency_ms", hue="high_latency", palette="coolwarm", s=30)
    plt.title("Load vs Latency")
    plt.tight_layout()
    plt.show()

    # Correlation heatmap
    plt.figure(figsize=(5, 4))
    sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="mako")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()


class TinyNet(nn.Module):
    """Simple neural network: 2 inputs -> 16 hidden -> 1 output."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x)


def block3_pytorch(df):
    """Train a PyTorch model to predict latency."""
    print("\n===== BLOCK 3: PYTORCH =====")

    # Prepare data
    X = df[["traffic_load", "packet_loss"]].values.astype(np.float32)
    y = df["latency_ms"].values.astype(np.float32).reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert to tensors
    X_train = torch.from_numpy(X_train)
    y_train = torch.from_numpy(y_train)
    X_test = torch.from_numpy(X_test)
    y_test = torch.from_numpy(y_test)

    model = TinyNet()
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    for epoch in range(500):
        model.train()
        optimizer.zero_grad()
        loss = loss_fn(model(X_train), y_train)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}/500 - Loss: {loss.item():.3f}")

    # Evaluate
    model.eval()
    with torch.no_grad():
        test_preds = model(X_test)
        test_loss = loss_fn(test_preds, y_test).item()
    print(f"Test MSE: {test_loss:.2f}")

    # Show sample predictions
    print("\nSample predictions:")
    X_test_np = X_test.numpy()
    y_test_np = y_test.numpy()
    for i in range(5):
        print(f"  load={X_test_np[i, 0]:.1f}, loss={X_test_np[i, 1]:.2f} -> "
              f"true={y_test_np[i, 0]:.2f}, pred={test_preds[i, 0].item():.2f}")


if __name__ == "__main__":
    df = generate_fake_network_data(500)

    block1_basic_ml(df)
    block2_eda(df)
    block3_pytorch(df)
