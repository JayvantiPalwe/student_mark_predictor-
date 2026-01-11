from flask import Flask, render_template, request
import numpy as np
import joblib


# Create Flask app
app = Flask(__name__)

# Load the trained model
# (You will create this file with train_model.py)
model = joblib.load("student_mark_predictor (1).pkl")



@app.route("/")
def home():
    # initial page load (no prediction yet)
    return render_template("index.html", prediction=None)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get the value from the form
        hours_str = request.form.get("study_hours", "")
        hours = float(hours_str)

        # Model expects 2D array: [[hours]]
        predicted_marks = model.predict(np.array([[hours]]))[0]

        # Round for display
        predicted_marks = round(predicted_marks, 2)

        message = f"Predicted Marks for {hours} hours of study: {predicted_marks}"

        return render_template("index.html", prediction=message, last_hours=hours_str)

    except Exception as e:
        # Show error on page (helps while learning)
        return render_template("index.html", prediction=f"Error: {e}")


if __name__ == "__main__":
    # debug=True only for development
    app.run(debug=True)


