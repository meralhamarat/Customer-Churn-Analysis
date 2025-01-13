import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
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

# Save dataset to CSV
df.to_csv("enhanced_customer_churn.csv", index=False)
print("Dataset saved to 'enhanced_customer_churn.csv'")
