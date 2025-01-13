import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Random data generation for the dataset
np.random.seed(42)
n_samples = 500

# Creating dataset
data = {
    "CustomerID": np.arange(1, n_samples + 1),
    "Gender": np.random.choice(["Male", "Female"], size=n_samples),
    "Age": np.random.randint(18, 70, size=n_samples),
    "MonthlyCharges": np.round(np.random.uniform(20, 100, size=n_samples), 2),
    "TotalCharges": np.round(np.random.uniform(200, 5000, size=n_samples), 2),
    "Contract": np.random.choice(["Month-to-month", "One year", "Two year"], size=n_samples),
    "Tenure": np.random.randint(1, 60, size=n_samples),
    "Churn": np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3]),
}

df = pd.DataFrame(data)

# Handling missing values (simulating some missing data)
nan_indices = np.random.choice(df.index, size=20, replace=False)
df.loc[nan_indices, "TotalCharges"] = np.nan
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

# Adding new features
df["AverageChargesPerMonth"] = (df["TotalCharges"] / df["Tenure"]).round(2)
bins = [18, 30, 45, 60, 70]
labels = ["18-29", "30-44", "45-59", "60-69"]
df["AgeGroup"] = pd.cut(df["Age"], bins=bins, labels=labels)
df["HighCharges"] = (df["MonthlyCharges"] > 75).astype(int)

# Adding Tenure groups
bins = [0, 12, 24, 36, 48, 60]
labels = ["0-12", "13-24", "25-36", "37-48", "49-60"]
df["TenureGroup"] = pd.cut(df["Tenure"], bins=bins, labels=labels)

# Visualization: Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# Visualization: Monthly Charges vs Tenure by Churn
plt.figure(figsize=(10, 6))
sns.scatterplot(x="MonthlyCharges", y="Tenure", hue="Churn", data=df, palette="coolwarm")
plt.title("Monthly Charges vs Tenure by Churn")
plt.xlabel("Monthly Charges")
plt.ylabel("Tenure")
plt.show()

# Visualization: Churn by Contract Type
plt.figure(figsize=(10, 6))
sns.countplot(x="Contract", hue="Churn", data=df, palette="Set2")
plt.title("Churn Rate by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Count")
plt.legend(title="Churn", labels=["No", "Yes"])
plt.show()

# Machine Learning: Preparing data
features = ["Gender", "Age", "MonthlyCharges", "Tenure", "Contract", "HighCharges"]
df_encoded = pd.get_dummies(df[features], drop_first=True)
X = df_encoded
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model training
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Feature importance visualization
feature_importances = model.feature_importances_
features_list = X.columns
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances, y=features_list)
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.show()

# Stratified Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), 
                           param_grid=param_grid, 
                           cv=3, 
                           verbose=2, 
                           n_jobs=-1)

grid_search.fit(X_train, y_train)
print("Best Parameters from Grid Search:", grid_search.best_params_)

# Retraining with the best parameters
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)

print("\nBest Model Confusion Matrix:\n", confusion_matrix(y_test, y_pred_best))
print("\nBest Model Classification Report:\n", classification_report(y_test, y_pred_best))

# Save dataset to CSV
df.to_csv("enhanced_customer_churn.csv", index=False)
print("Dataset saved to 'enhanced_customer_churn.csv'")
