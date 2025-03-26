import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load MNIST dataset
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist.data, mnist.target.astype(int)  # Convert labels to integers

# Normalize pixel values
X = X / 255.0

# Reduce data size for faster training
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.9, random_state=42)  # Keep 10% for training
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"Train shape: {X_train.shape}, Validation shape: {X_val.shape}, Test shape: {X_test.shape}")

# Apply PCA to reduce dimensions
pca = PCA(n_components=100)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Train a logistic regression model
log_reg = LogisticRegression(max_iter=1000, solver='saga', multi_class='multinomial', random_state=42)
log_reg.fit(X_train_pca, y_train)

# Make predictions
y_pred = log_reg.predict(X_test_pca)

# Compute performance metrics
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred))
