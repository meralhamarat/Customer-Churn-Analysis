import pandas as pd
import numpy as np

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

# Save to CSV file
file_path = "customer_churn.csv"
df.to_csv(file_path, index=False)
print(f"Dataset saved to {file_path}")
