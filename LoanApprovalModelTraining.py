# Author: Taashfeen Ashraf
# Date last modified: 22/11/2024

# Step 0: Import Libraries
import pandas as pd  # For data manipulation and analysis
from sklearn.preprocessing import LabelEncoder, StandardScaler  # For encoding categorical variables and scaling
from sklearn.model_selection import train_test_split  # For splitting the dataset into training and testing sets
from sklearn import svm  # Support Vector Machine (SVM) for classification
from sklearn.linear_model import LogisticRegression  # Logistic Regression for model comparison
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # For model Evaluation metrics


# Step 1: Load the Dataset
# Load the preprocessed dataset from Task 1 into a pandas DataFrame
data = pd.read_csv('/Users/taashfeen/Desktop/preprocessed_loan_data.csv')

# Step 2: Drop Irrelevant Columns
# Remove the 'Loan_ID' column as it is a unique identifier and not relevant to predictions
if 'Loan_ID' in data.columns:
    data = data.drop('Loan_ID', axis=1)

# Step 3: Split Features (X) and Target (y)
# Separate the independent features (X) and the target variable (y)
X = data.drop('Loan_Status', axis=1)  # X contains all columns except 'Loan_Status'
y = data['Loan_Status']  # y contains the 'Loan_Status' column

# Step 4: Encode the Target Variable
# LabelEncoder is used to convert binary categorical data into numerical values (0 and 1)
label_encoder = LabelEncoder()  # Initialize the LabelEncoder
y = label_encoder.fit_transform(y)  # Encode 'Y' as 1 and 'N' as 0

# Step 5: Encode Categorical Features in X
# One-hot encode categorical features in X for machine learning compatibility
X = pd.get_dummies(X, drop_first=True)  # drop_first=True avoids multicollinearity

# Step 6: Standardize Numerical Features
# StandardScaler scales numerical columns to have a mean of 0 and a standard deviation of 1
scaler = StandardScaler()  # Initialize the Scaler
X = scaler.fit_transform(X)  # Apply scaling to all numerical features in X

# Step 7: Split the Dataset into Training and Testing Sets
# Use train_test_split to divide the data into 80% training and 20% testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Train the SVM Classifier
# Initialize and train the Support Vector Machine (SVM) model using the RBF kernel
svm_model = svm.SVC(kernel='rbf', random_state=42)  # RBF kernel handles non-linear classification
svm_model.fit(X_train, y_train)  # Fit the model on the training data

# Step 9: Make Predictions
# Use the trained SVM model to predict the target variable on the test set
y_pred_svm = svm_model.predict(X_test)

# Step 10: Evaluate the SVM Model
# Evaluate the model using a confusion matrix, accuracy score, and classification report

# Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_svm))

# Accuracy Score
accuracy = accuracy_score(y_test, y_pred_svm)
print("SVM Accuracy:", accuracy)

# Classification Report
print("SVM Classification Report:")
print(classification_report(y_test, y_pred_svm))

# Step 11: Train a Logistic Regression Model for Comparison

# Initialize and train the Logistic Regression model
logreg_model = LogisticRegression(random_state=42, max_iter=1000)  # Increase max_iter to ensure convergence
logreg_model.fit(X_train, y_train)  # Train the model on the training data

# Step 12: Make Predictions with Logistic Regression
y_pred_logreg = logreg_model.predict(X_test)  # Make predictions on the test data

# Step 13: Evaluate the Logistic Regression Model
# Calculate accuracy
logreg_accuracy = accuracy_score(y_test, y_pred_logreg)

# Print evaluation metrics
print("Logistic Regression Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_logreg))  # Display confusion matrix

print("Logistic Regression Accuracy:", logreg_accuracy)  # Display accuracy

print("Logistic Regression Classification Report:\n")
print(classification_report(y_test, y_pred_logreg))  # Display classification report

# Step 14: Compare SVM and Logistic Regression
print("\nComparison of Model Performances:")
print(f"SVM Accuracy: {accuracy:.2f}")  # Display SVM accuracy
print(f"Logistic Regression Accuracy: {logreg_accuracy:.2f}")  # Display Logistic Regression accuracy

if accuracy > logreg_accuracy:
    print("SVM outperforms Logistic Regression in this dataset.")
elif accuracy < logreg_accuracy:
    print("Logistic Regression outperforms SVM in this dataset.")
else:
    print("Both models have the same accuracy.")

# Step 15: Save Predictions to CSV
# Save predictions from both models for analysis and future reference
output = pd.DataFrame({'SVM Predictions': y_pred_svm, 'Logistic Regression Predictions': y_pred_logreg})
output.to_csv('/Users/taashfeen/Desktop/model_predictions.csv', index=False)
