import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Random data generation for the dataset
np.random.seed(42)

# Number of samples
n_samples = 500

# Creating columns
data = {
    "CustomerID": np.arange(1, n_samples + 1),
    "Gender": np.random.choice(["Male", "Female"], size=n_samples),
    "Age": np.random.randint(18, 70, size=n_samples),
    "MonthlyCharges": np.round(np.random.uniform(20, 100, size=n_samples), 2),
    "TotalCharges": np.round(np.random.uniform(200, 5000, size=n_samples), 2),
    "Contract": np.random.choice(["Month-to-month", "One year", "Two year"], size=n_samples),
    "Tenure": np.random.randint(1, 60, size=n_samples),  # months
    "Churn": np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3]),  # 70% no churn, 30% churn
}

# Creating a DataFrame
df = pd.DataFrame(data)

# Adding a new column: AverageChargesPerMonth
df["AverageChargesPerMonth"] = (df["TotalCharges"] / df["Tenure"]).round(2)

# Checking for missing values
print("Missing values:\n", df.isnull().sum())

# Basic statistics
print("\nDataset statistics:")
print(df.describe())

# Dataset information
print("\nDataset info:")
print(df.info())

# Save to CSV file
file_path = "customer_churn.csv"
df.to_csv(file_path, index=False)
print(f"\nDataset saved to {file_path}")

# Visualization 1: Distribution of Monthly Charges
plt.figure(figsize=(10, 6))
sns.histplot(df["MonthlyCharges"], kde=True, bins=30, color="blue")
plt.title("Distribution of Monthly Charges")
plt.xlabel("Monthly Charges")
plt.ylabel("Frequency")
plt.show()

# Visualization 2: Churn rate by Contract type
plt.figure(figsize=(10, 6))
sns.countplot(x="Contract", hue="Churn", data=df, palette="Set2")
plt.title("Churn Rate by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Count")
plt.legend(title="Churn", labels=["No", "Yes"])
plt.show()

# Visualization 3: Gender distribution
plt.figure(figsize=(6, 6))
df["Gender"].value_counts().plot.pie(autopct="%1.1f%%", colors=["skyblue", "lightpink"], startangle=90)
plt.title("Gender Distribution")
plt.ylabel("")
plt.show()

# Visualization 4: Age distribution
plt.figure(figsize=(10, 6))
sns.boxplot(x="Churn", y="Age", data=df, palette="pastel")
plt.title("Age Distribution by Churn")
plt.xlabel("Churn (0 = No, 1 = Yes)")
plt.ylabel("Age")
plt.show()

# Adding Age Groups
bins = [18, 30, 45, 60, 70]
labels = ["18-29", "30-44", "45-59", "60-69"]
df["AgeGroup"] = pd.cut(df["Age"], bins=bins, labels=labels)

# Visualization
sns.countplot(x="AgeGroup", hue="Churn", data=df, palette="pastel")
plt.title("Churn by Age Group")
plt.show()
