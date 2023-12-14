from flask import Flask,render_template,request
import pickle
import numpy as np

model = pickle.load(open('model.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict_heartdisease():
    x1 = float(request.form.get('curricular_units_2nd_sem'))
    x2 = int(request.form.get('curricular_units_1st_sem'))
    x3 = int(request.form.get('age_at_enrollment'))
    x4 = int(request.form.get('tuition_fees'))
    x5 = int(request.form.get('scholarship_holder'))
    x6 = int(request.form.get('gender'))

    # prediction
    result = model.predict(np.array([x1,x2,x3,x4,x5,x6]).reshape(1,6))
    fin_result = float(round(result[0],2))
    
    if fin_result==1:
        fin_result='Success'
    else:
        fin_result='Dropout'

    return render_template('index.html',fin_result=fin_result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)

app.static_folder = 'static'
