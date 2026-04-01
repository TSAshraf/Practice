# Author: Taashfeen Ashraf
# Date last modified: 22/11/2024
# Coursework Task 1

# Step 0: Import Libraries
import pandas as pd  # Import Pandas for data manipulation and analysis
from sklearn.preprocessing import StandardScaler  # Import StandardScaler for feature scaling

# Step 1: Load the dataset
# Read the CSV file containing the dataset into a pandas DataFrame
data = pd.read_csv('/Users/taashfeen/Desktop/train.csv')

# Step 2: Handling Missing Values

# Remove columns where all values are missing
data = data.dropna(axis=1, how='all')
# dropna() removes rows or columns with missing values.
# axis=1 indicates columns, and 'how=all' removes columns only if all values are NaN.

# Fill missing values based on the data type and context:
# fillna() replaces missing (NaN) values with specified values to avoid issues during analysis or modeling.

# Use mode (most frequent value) for categorical or binary columns.
data['Gender'] = data['Gender'].fillna(data['Gender'].mode()[0])  # Replace NaN with the most frequent value
data['Married'] = data['Married'].fillna(data['Married'].mode()[0])
data['Dependents'] = data['Dependents'].fillna(data['Dependents'].mode()[0])
data['Self_Employed'] = data['Self_Employed'].fillna(data['Self_Employed'].mode()[0])
data['Credit_History'] = data['Credit_History'].fillna(data['Credit_History'].mode()[0])
data['Loan_Status'] = data['Loan_Status'].fillna(data['Loan_Status'].mode()[0])

# Use median (middle value) for numerical columns to minimise the impact of outliers
# Median is less sensitive to extreme values, making it suitable for financial data.
data['LoanAmount'] = data['LoanAmount'].fillna(data['LoanAmount'].median()) # Replace NaN with the middle value
data['Loan_Amount_Term'] = data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].median())

# Step 3: Detecting and Removing Outliers
# Define a function to calculate the Interquartile Range (IQR) and remove outliers
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)  # First Quartile (25th percentile)
    Q3 = df[column].quantile(0.75)  # Third Quartile (75th percentile)
    IQR = Q3 - Q1  # Calculate the IQR (Q3 - Q1)
    lower_bound = Q1 - 1.5 * IQR  # Define the lower bound for outliers
    upper_bound = Q3 + 1.5 * IQR  # Define the upper bound for outliers
    # Keep only rows within the defined bounds
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Apply the function to specified numerical columns to remove outliers
data = remove_outliers(data, 'ApplicantIncome')  # Remove outliers in ApplicantIncome
data = remove_outliers(data, 'CoapplicantIncome')  # Remove outliers in CoapplicantIncome
data = remove_outliers(data, 'LoanAmount')  # Remove outliers in LoanAmount

# Step 4: Encode Categorical Variables
# Map multi-category columns to separate binary columns
# 'pd.get_dummies' converts categorical variables into binary columns
# 'drop_first=True' removes the first category to avoid redundancy and prevent multicollinearity
# 'dtype=int' ensures the binary columns are stored as integers for efficient computation
data = pd.get_dummies(data, columns=['Dependents', 'Property_Area'], drop_first=True, dtype=int)

# Step 5: Feature Scaling (Optional)
# Standardize numerical columns to ensure consistency in scale
# StandardScaler scales each column to have a mean of 0 and a standard deviation of 1
# This ensures that no feature dominates the model due to its range or magnitude
scaler = StandardScaler()  # Initialize the Scaler
# Define specific numerical columns to scale
numerical_columns = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
# Apply scaling to the selected columns
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

# Step 6: Display the DataFrame
# Adjust display settings to show all rows and columns in the DataFrame
pd.set_option('display.max_rows', None)  # Display all rows
pd.set_option('display.max_columns', None)  # Display all columns
print(data)  # Display the processed DataFrame

# Step 7: Save to a CSV file
# Save the DataFrame as a CSV file named 'preprocessed_loan_data.csv'
# The 'index=False' parameter ensures row indices are not included in the output
data.to_csv('/Users/taashfeen/Desktop/preprocessed_loan_data.csv', index=False)

