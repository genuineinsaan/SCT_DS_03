# To run this code at first go to link "https://archive.ics.uci.edu/dataset/222/bank+marketing" and download the file "bank-full.csv"
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Load the dataset from the local file
data = pd.read_csv("bank-full.csv", sep=';')  # Replace with the correct path if needed

# Display basic information about the dataset
print("Dataset Overview:")
print(data.head())
print(data.info())

# Encode categorical variables
categorical_columns = data.select_dtypes(include=['object']).columns
data_encoded = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

# Split the dataset into features and target variable
X = data_encoded.drop("y_yes", axis=1)  # "y_yes" indicates if a customer subscribed
y = data_encoded["y_yes"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Decision Tree Classifier
clf = DecisionTreeClassifier(criterion="entropy", random_state=42, max_depth=5)

# Train the model
clf.fit(X_train, y_train)

# Predict on the test data
y_pred = clf.predict(X_test)

# Evaluate the model
print("Model Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Plot the decision tree
plt.figure(figsize=(20, 10))
plot_tree(clf, feature_names=X.columns, class_names=["No", "Yes"], filled=True, rounded=True)
plt.title("Decision Tree")
plt.show()
