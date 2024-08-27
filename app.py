
from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np
import os

app = Flask(__name__)

model=pickle.load(open('model.pkl','rb'))
scaler = pickle.load(open('scaler.sav', 'rb'))


@app.route('/')
def home():
    return render_template("flood_probability_template.html")

@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features = []
    form_values = request.form.values()
    try:
        int_features = [int(x) for x in form_values if x.isdigit()]
    except ValueError:
         return render_template('flood_probability_template.html', alert_message= 'One or more inputs could not be converted to an integer.',reset_form=False)
    if len(int_features) < 20:
        return render_template('flood_probability_template.html', alert_message="Not enough features. At least 20 features are required.",reset_form=False)
        
    transformed_features= scaler.transform(np.array([int_features])).reshape(1, -1)
    prediction=model.predict(transformed_features)
    output='{0:.{1}f}'.format(prediction[0][0], 2)

    if output>str(0.5):
        return render_template('flood_probability_template.html',pred='Your are in Danger.\nProbability of flood occuring is {}'.format(output),bhai="kuch karna hain iska ab?" ,reset_form=False)
    else:
        return render_template('flood_probability_template.html',pred='Your are safe.\n Probability of flood occuring is {}'.format(output),bhai="Your Forest is Safe for now", reset_form=False)


if __name__ == '__main__':
    app.run(debug=True)