import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

# 1. Data creation
def create_dataset(n_samples=500, seed=42):
    np.random.seed(seed)
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
    return df

# 2. Data preprocessing
def preprocess_data(df):
    # Creating and filling missing values
    nan_indices = np.random.choice(df.index, size=20, replace=False)
    df.loc[nan_indices, "TotalCharges"] = np.nan
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # Creating new features
    df["AverageChargesPerMonth"] = (df["TotalCharges"] / df["Tenure"]).round(2)
    bins_age = [18, 30, 45, 60, 70]
    labels_age = ["18-29", "30-44", "45-59", "60-69"]
    df["AgeGroup"] = pd.cut(df["Age"], bins=bins_age, labels=labels_age)
    df["HighCharges"] = (df["MonthlyCharges"] > 75).astype(int)
    bins_tenure = [0, 12, 24, 36, 48, 60]
    labels_tenure = ["0-12", "13-24", "25-36", "37-48", "49-60"]
    df["TenureGroup"] = pd.cut(df["Tenure"], bins=bins_tenure, labels=labels_tenure)

    return df

# 3. Data visualization
def visualize_data(df):
    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.show()

    # Monthly charges vs. tenure by churn
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x="MonthlyCharges", y="Tenure", hue="Churn", data=df, palette="coolwarm")
    plt.title("Monthly Charges vs Tenure by Churn")
    plt.xlabel("Monthly Charges")
    plt.ylabel("Tenure")
    plt.show()

    # Churn rate by contract type
    plt.figure(figsize=(10, 6))
    sns.countplot(x="Contract", hue="Churn", data=df, palette="Set2")
    plt.title("Churn Rate by Contract Type")
    plt.xlabel("Contract Type")
    plt.ylabel("Count")
    plt.legend(title="Churn", labels=["No", "Yes"])
    plt.show()

# 4. Machine learning modeling
def train_models(df):
    features = ["Gender", "Age", "MonthlyCharges", "Tenure", "Contract", "HighCharges"]
    df_encoded = pd.get_dummies(df[features], drop_first=True)
    X = df_encoded
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    models = {
        "Random Forest": RandomForestClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(random_state=42),
        "XGBoost": XGBClassifier(random_state=42)
    }

    for name, model in models.items():
        print(f"\n{name} Model Training...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        print("\nClassification Report:\n", classification_report(y_test, y_pred))

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_prob)
            print(f"ROC-AUC Score: {roc_auc:.2f}")

            fpr, tpr, _ = roc_curve(y_test, y_prob)
            plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.show()

# 5. Hyperparameter tuning
def hyperparameter_tuning(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42), 
        param_grid=param_grid, 
        cv=3, 
        verbose=2, 
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_

# Main script
if __name__ == "__main__":
    # Create and preprocess the dataset
    df = create_dataset()
    df = preprocess_data(df)

    # Visualize the data
    visualize_data(df)

    # Train models
    train_models(df)

    # Save the dataset
    df.to_csv("enhanced_customer_churn.csv", index=False)
    print("Dataset saved to 'enhanced_customer_churn.csv'")
