from flask import Flask, render_template, request
import numpy as np
import pickle
app = Flask(__name__)
filename = 'redwine.pkl'
model = pickle.load(open(filename, 'rb'))    # load the model
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])  # The user input is processed here
def predict():
    volatile_acidity = request.form['volatile_acidity']
    citric_acid = request.form['citric_acid']
    residual_sugar = request.form['residual_sugar']
    total_sulfur_dioxide = request.form['total_sulfur_dioxide']
    alcohol = request.form['alcohol']
    pred = model.predict(np.array([[volatile_acidity,citric_acid,residual_sugar,total_sulfur_dioxide,alcohol]]))
    print(pred)
    return render_template('index.html', predict=str(pred))
if __name__ == '__main__':
    app.run