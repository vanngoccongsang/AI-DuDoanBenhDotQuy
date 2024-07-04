from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Đường dẫn đến tệp mô hình đã lưu
model_filename = 'logictic_Regression.joblib'

# Load mô hình từ tệp joblib
model = joblib.load(model_filename)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Lấy dữ liệu từ form
        gender = int(request.form['gender'])
        age = int(request.form['age'])
        hypertension = int(request.form['hypertension'])
        heart_disease = int(request.form['heart_disease'])
        ever_married = int(request.form['ever_married'])
        work_type = int(request.form['work_type'])
        Residence_type = int(request.form['Residence_type'])
        avg_glucose_level = int(request.form['avg_glucose_level'])
        bmi = int(request.form['bmi'])
        smoking_status = int(request.form['smoking_status'])
        # Sử dụng mô hình để dự đoán
        new_patient = [[gender, age, hypertension, heart_disease,
                         ever_married, work_type, Residence_type, 
                         avg_glucose_level, bmi, smoking_status]]
        predicted_label = model.predict(new_patient)

        result = "Xin chia buồn bạn đã mắc bệnh đột quỵ" \
            if predicted_label == 1 \
                else "Chúc mừng bạn đã không có bệnh. kết quả rất tốt"

        return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
