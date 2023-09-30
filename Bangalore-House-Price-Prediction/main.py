import os

from flask import Flask, render_template, request, send_from_directory, redirect, url_for
import pandas as pd
import pickle
import  numpy as np
app = Flask(__name__)
data = pd.read_csv("Cleaned_data.csv")
pipe = pickle.load(open('LinearRegression.pkl','rb'))
@app.route('/')
def index():

    locations = sorted(data['location'].unique())
    return render_template('index.html',locations=locations)

@app.route('/predict',methods = ['POST'])
def predict():
    location = request.form.get('location')
    bhk = request.form.get('bhk')
    bath = request.form.get('bath')
    sqft = request.form.get('total_sqft')
    input_data = pd.DataFrame([[location, sqft, bath, bhk]], columns=['location', 'total_sqft', 'bath', 'bhk'])
    prediction = pipe.predict(input_data)[0] * 1e5

    # Format the prediction with commas and two decimal places
    formatted_prediction = '{:,.2f}'.format(prediction)

    return formatted_prediction
@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']

        # Specify the full path to the contact_details.txt file
        file_path = os.path.join(os.path.dirname(__file__), 'messages', 'contact_details.txt')

        # Save the contact details to the text file
        with open(file_path, 'a') as file:
            file.write(f'Name: {name}\n')
            file.write(f'Email: {email}\n')
            file.write(f'Message:\n{message}\n\n')
    else:
        return render_template('contact.html')
    return redirect(url_for('index'))
@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')

# Serve the logo image from the "images" directory outside the "static" folder
@app.route('/images/<filename>')
def custom_static(filename):
    return send_from_directory('images', filename)

    return send_from_directory(images_directory, filename)
if __name__ == '__main__':
    app.run(debug=True)
