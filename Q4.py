# Step 1: Load dataset
from sklearn.datasets import load_iris

iris = load_iris()
X, y = iris.data, iris.target

# Step 2: Split into train and test sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Quick EDA - View basic statistics
import pandas as pd

df = pd.DataFrame(X, columns=iris.feature_names)
print(df.describe())

# Step 4: Feature Selection 1 - Univariate Selection (SelectKBest)
from sklearn.feature_selection import SelectKBest, f_classif

selector = SelectKBest(score_func=f_classif, k=2)
X_train_uni = selector.fit_transform(X_train, y_train)
X_test_uni = selector.transform(X_test)

# Step 5: Feature Selection 2 - Random Forest Feature Importance
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
print("Feature Importances (Random Forest):", rf.feature_importances_)

# Step 6: Feature Selection 3 - Recursive Feature Elimination (RFE) using SVM
from sklearn.feature_selection import RFE
from sklearn.svm import SVC

svm = SVC(kernel="linear")
rfe = RFE(estimator=svm, n_features_to_select=2)
rfe.fit(X_train, y_train)
X_train_rfe, X_test_rfe = rfe.transform(X_train), rfe.transform(X_test)

# Step 7: Train Logistic Regression Model (Before Feature Selection)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy (Before Feature Selection):", accuracy_score(y_test, y_pred))

# Step 8: Train Model After Univariate Feature Selection
model.fit(X_train_uni, y_train)
y_pred_uni = model.predict(X_test_uni)
print("Accuracy (After Univariate Selection):", accuracy_score(y_test, y_pred_uni))
