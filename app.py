from flask import Flask, render_template, request
import pandas as pd
import joblib
from datetime import datetime

app = Flask(__name__)

# Load trained model
model = joblib.load('save.pkl')  # make sure save.pkl exists

@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction = None
    probability = None

    if request.method == 'POST':
        try:
            data = {
                'Gender': request.form.get('Gender'),
                'Blood Type': request.form.get('BloodType'),
                'Medical Condition': request.form.get('MedicalCondition'),
                'Insurance Provider': request.form.get('InsuranceProvider'),
                'Admission Type': request.form.get('AdmissionType'),
                'Medication': request.form.get('Medication'),
                'Test Results': request.form.get('TestResults'),
                'Date of Admission': request.form.get('DateOfAdmission'),
                'Discharge Date': request.form.get('DischargeDate')
            }

            df = pd.DataFrame([data])
            df['Date of Admission'] = pd.to_datetime(df['Date of Admission'])
            df['Discharge Date'] = pd.to_datetime(df['Discharge Date'])
            df['Length_of_Stay'] = (df['Discharge Date'] - df['Date of Admission']).dt.days

            df.drop(['Date of Admission', 'Discharge Date'], axis=1, inplace=True)

            prob = model.predict_proba(df)[0][1]
            prediction = "Readmitted" if prob >= 0.70 else "Not Readmitted"
            probability = round(prob * 100, 2)

        except Exception as e:
            prediction = f"Error: {str(e)}"
            probability = None

    return render_template('index.html',
                           prediction=prediction,
                           probability=probability,
                           current_date=datetime.today().strftime('%Y-%m-%d'))

if __name__ == '__main__':
    app.run(debug=True)
