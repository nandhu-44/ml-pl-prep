import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import GridSearchCV

# Load MNIST dataset
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist.data, mnist.target.astype(int)  # Convert labels to integers

# Normalize pixel values
X = X / 255.0

# Split dataset into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"Train shape: {X_train.shape}, Validation shape: {X_val.shape}, Test shape: {X_test.shape}")

######################## CELL 2 ################################
# Train a logistic regression model
log_reg = LogisticRegression(max_iter=500, solver='saga', multi_class='multinomial', random_state=42)
log_reg.fit(X_train, y_train)

# Make predictions
y_pred = log_reg.predict(X_test)

######################## CELL 3 ################################
# Compute performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Print detailed classification report
print(classification_report(y_test, y_pred))


######### CELL 4 #########
# Define hyperparameters for tuning
param_grid = {
    'C': [0.01, 0.1, 1, 10],  # Regularization strength
    'max_iter': [200, 500]
}

# Perform Grid Search Cross-Validation
grid_search = GridSearchCV(LogisticRegression(solver='saga', multi_class='multinomial', random_state=42), 
                           param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Print best parameters
print(f"Best Hyperparameters: {grid_search.best_params_}")

# Evaluate the best model
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)

print(f"Optimized Model Accuracy: {accuracy_score(y_test, y_pred_best):.4f}")


################## CELL 5 ################
from sklearn.manifold import TSNE

# Reduce dimensions to 2D using t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_test_2D = tsne.fit_transform(X_test[:1000])  # Use only 1000 samples for better visualization

# Plot decision boundary
plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_test_2D[:, 0], X_test_2D[:, 1], c=y_pred_best[:1000], cmap="jet", alpha=0.6)
plt.colorbar(scatter, label="Predicted Labels")
plt.title("Decision Boundary Visualization using t-SNE")
plt.xlabel("t-SNE Feature 1")
plt.ylabel("t-SNE Feature 2")
plt.show()
