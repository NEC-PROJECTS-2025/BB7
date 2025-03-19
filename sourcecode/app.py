import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import pandas as pd
scaler = StandardScaler()

app = Flask(__name__)

# Load the trained model
model = joblib.load('model.pkl')
@app.route('/')
def index():
    return render_template('home.html')

@app.route('/test')
def test():
    return render_template('test.html')  # Render the test.html page
@app.route('/login')
def login():
    return render_template('login.html')  # Render the test.html page
@app.route('/reg')
def reg():
    return render_template('reg.html')  # Render the test.html page
@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    

    # Collect features from form data
    Age=int(request.form['Age'])
    Gender=int(request.form['Gender'])
    Total_Bilirubin=float(request.form['Total_Bilirubin'])
    Direct_Bilirubin=float(request.form['Direct_Bilirubin'])
    Alkaline_Phosphotase=int(request.form['Alkaline_Phosphotase'])
    Alamine_Aminotransferase=int(request.form['Alamine_Aminotransferase'])
    Aspartate_Aminotransferase=int(request.form['Aspartate_Aminotransferase'])
    Total_Protiens=float(request.form['Total_Protiens'])
    Albumin=float(request.form['Albumin'])
    Albumin_and_Globulin_Ratio=float(request.form['Albumin_and_Globulin_Ratio'])
    

    
    
    # feature_array=[[gender,age,hypertension,heart_disease,ever_married,work_type,Residence_type,
    #                  avg_glucose_level,bmi,smoking_status]]
    # feature_array = scaler.transform(feature_array)
    # features_array = np.array(feature_array).reshape(1, -1)
    # print(feature_array)
    input_data = pd.DataFrame([[Age,Gender,Total_Bilirubin,Direct_Bilirubin,Alkaline_Phosphotase,Alamine_Aminotransferase,Aspartate_Aminotransferase,Total_Protiens,Albumin,Albumin_and_Globulin_Ratio	]],
    columns=['Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin', 'Alkaline_Phosphotase', 'Alamine_Aminotransferase', 'Aspartate_Aminotransferase', 'Total_Protiens', 'Albumin', 'Albumin_and_Globulin_Ratio']) # Replace with your actual column names

# Scale the input data using the previously fitted scaler
    # input_data_scaled = scaler.transform(input_data)

# Convert the scaled data back to a DataFrame with original column names
    input_data = pd.DataFrame(input_data, columns=input_data.columns)

# Now you can use column names for selection
    input_data = input_data[['Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin', 'Alkaline_Phosphotase', 'Alamine_Aminotransferase', 'Aspartate_Aminotransferase', 'Total_Protiens', 'Albumin', 'Albumin_and_Globulin_Ratio']] # Replace with your actual column names
    prediction = model.predict(input_data)
    print(prediction)

    if prediction == 0:
        return render_template("stroke.html", prediction="Liver Disease Detected. Consult a Doctor!")  
    elif prediction ==1:
        return render_template("nostroke.html", prediction="Your Liver is in Good condition...!")

if __name__ == "__main__":
    app.run(debug=True,port=5050)
