import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Load the dataset
df = pd.read_csv("student_marks.csv")

# Features and labels
X = df[["Hours"]]
y = df["Marks"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=51
)

# Train the model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Scores (optional)
print("Training Score:", lr.score(X_train, y_train))
print("Testing Score:", lr.score(X_test, y_test))

# Save model
joblib.dump(lr, "student_mark_model.pkl")
print("Model saved successfully!")





