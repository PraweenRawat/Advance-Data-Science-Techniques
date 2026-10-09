import json

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Mini Project 6: Logistic Regression Analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Initial Analysis\n",
    "- Load and inspect the dataset.\n",
    "- Check shape, data types, missing values, duplicates, and statistics.\n",
    "- Identify numerical and categorical variables.\n",
    "- Identify the target variable."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load the dataset\n",
    "df = pd.read_csv('claimants (1).csv')\n",
    "\n",
    "# Inspect the dataset\n",
    "display(df.head())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Check shape\n",
    "print(f\"Dataset Shape: {df.shape}\")\n",
    "\n",
    "# Check data types and non-null counts\n",
    "print(\"\\nData Types and Info:\")\n",
    "df.info()\n",
    "\n",
    "# Check missing values\n",
    "print(\"\\nMissing Values:\")\n",
    "print(df.isnull().sum())\n",
    "\n",
    "# Check duplicates\n",
    "print(f\"\\nNumber of Duplicate Rows: {df.duplicated().sum()}\")\n",
    "\n",
    "# Statistics\n",
    "print(\"\\nSummary Statistics:\")\n",
    "display(df.describe(include='all'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Identify numerical and categorical variables\n",
    "numerical_cols = ['CLMAGE', 'LOSS']\n",
    "categorical_cols = ['CLMSEX', 'CLMINSUR', 'SEATBELT']\n",
    "target = 'ATTORNEY'\n",
    "\n",
    "print(f\"Numerical Variables: {numerical_cols}\")\n",
    "print(f\"Categorical Variables: {categorical_cols}\")\n",
    "print(f\"Target Variable: {target}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Data Pre-processing\n",
    "- Handle missing values and duplicates.\n",
    "- Identify and handle outliers.\n",
    "- Encode categorical variables.\n",
    "- Scale features if required.\n",
    "- Select relevant features."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Drop CASENUM as it's just an identifier\n",
    "if 'CASENUM' in df.columns:\n",
    "    df.drop('CASENUM', axis=1, inplace=True)\n",
    "\n",
    "# Handle duplicates\n",
    "df.drop_duplicates(inplace=True)\n",
    "\n",
    "# Handle missing values\n",
    "# For categorical columns, fill with mode. For numerical, fill with median.\n",
    "for col in df.columns:\n",
    "    if df[col].isnull().sum() > 0:\n",
    "        if col in categorical_cols:\n",
    "            df[col].fillna(df[col].mode()[0], inplace=True)\n",
    "        else:\n",
    "            df[col].fillna(df[col].median(), inplace=True)\n",
    "\n",
    "print(\"Missing values after imputation:\\n\", df.isnull().sum())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Identify and handle outliers\n",
    "# Using IQR method for LOSS and CLMAGE\n",
    "def cap_outliers(data, col):\n",
    "    Q1 = data[col].quantile(0.25)\n",
    "    Q3 = data[col].quantile(0.75)\n",
    "    IQR = Q3 - Q1\n",
    "    lower_bound = Q1 - 1.5 * IQR\n",
    "    upper_bound = Q3 + 1.5 * IQR\n",
    "    data[col] = np.where(data[col] > upper_bound, upper_bound, \n",
    "                         np.where(data[col] < lower_bound, lower_bound, data[col]))\n",
    "    return data\n",
    "\n",
    "for col in numerical_cols:\n",
    "    df = cap_outliers(df, col)\n",
    "\n",
    "# Encode categorical variables (They are already 0/1 in this dataset, but we'll ensure they are integers)\n",
    "for col in categorical_cols:\n",
    "    df[col] = df[col].astype(int)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Scale features (StandardScaler for Logistic Regression is good practice)\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "\n",
    "scaler = StandardScaler()\n",
    "df[numerical_cols] = scaler.fit_transform(df[numerical_cols])\n",
    "\n",
    "# Select relevant features\n",
    "X = df.drop(target, axis=1)\n",
    "y = df[target]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Logistic Regression\n",
    "- Select independent and dependent variables.\n",
    "- Split the data into 80% training and 20% testing.\n",
    "- Build and train the Logistic Regression model.\n",
    "- Display coefficients and intercept."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "\n",
    "# Split the data into 80% training and 20% testing\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)\n",
    "\n",
    "# Build and train the Logistic Regression model\n",
    "log_model = LogisticRegression()\n",
    "log_model.fit(X_train, y_train)\n",
    "\n",
    "# Display coefficients and intercept\n",
    "print(\"Model Intercept:\", log_model.intercept_[0])\n",
    "print(\"\\nModel Coefficients:\")\n",
    "for feature, coef in zip(X.columns, log_model.coef_[0]):\n",
    "    print(f\"{feature}: {coef}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Prediction and Evaluation\n",
    "- Predict target values.\n",
    "- Calculate Accuracy.\n",
    "- Display the Confusion Matrix.\n",
    "- Compare actual and predicted values."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import accuracy_score, confusion_matrix, classification_report\n",
    "\n",
    "# Predict target values\n",
    "y_pred = log_model.predict(X_test)\n",
    "\n",
    "# Calculate Accuracy\n",
    "accuracy = accuracy_score(y_test, y_pred)\n",
    "print(f\"Accuracy: {accuracy:.4f}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Display the Confusion Matrix\n",
    "conf_matrix = confusion_matrix(y_test, y_pred)\n",
    "print(\"Confusion Matrix:\\n\", conf_matrix)\n",
    "\n",
    "plt.figure(figsize=(6, 4))\n",
    "sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', \n",
    "            xticklabels=['No Attorney', 'Attorney'], \n",
    "            yticklabels=['No Attorney', 'Attorney'])\n",
    "plt.xlabel('Predicted')\n",
    "plt.ylabel('Actual')\n",
    "plt.title('Confusion Matrix')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Compare actual and predicted values\n",
    "comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})\n",
    "display(comparison_df.head(10))\n",
    "\n",
    "# Classification report\n",
    "print(\"\\nClassification Report:\\n\", classification_report(y_test, y_pred))"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

with open("Logistic_Regression_Analysis.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)
print("Notebook generated successfully!")
