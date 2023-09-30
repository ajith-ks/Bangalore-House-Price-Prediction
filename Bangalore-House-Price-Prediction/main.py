import json
import os
from flask import Flask, render_template, request, send_from_directory, redirect, url_for, jsonify

import pandas as pd
import pickle

app = Flask(__name__)

# Load your data and model outside the route functions for efficiency
data = pd.read_csv("Cleaned_data.csv")
pipe = pickle.load(open('LinearRegression.pkl', 'rb'))
# Initialize the contact counter
contact_counter = 1
@app.route('/')
def index():
    locations = sorted(data['location'].unique())
    return render_template('index.html', locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    #print("response got : " + request.method)
    location = request.form.get('location')
    bhk = float(request.form.get('bhk'))
    bath = float(request.form.get('bath'))
    sqft = float(request.form.get('total_sqft'))
    #print([location, sqft, bath, bhk])
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
    global contact_counter

    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']

        # Specify the full path to the contact_details.json file
        file_path = os.path.join(os.path.dirname(__file__), 'messages', 'contact_details.json')

        # Create a dictionary with the contact details
        contact_data = {
            'Name': name,
            'Email': email,
            'Message': message
        }

        # Load existing data from the JSON file if it exists
        existing_data = {}
        if os.path.exists(file_path):
            with open(file_path, 'r') as json_file:
                existing_data = json.load(json_file)

        # Add the new contact_data to the dictionary with an incremental numerical key
        existing_data[contact_counter] = contact_data

        # Increment the contact_counter for the next entry
        contact_counter += 1

        # Save the updated data back to the JSON file
        with open(file_path, 'w') as json_file:
            json.dump(existing_data, json_file, indent=4)

        # Respond with a JSON confirmation
        return redirect(url_for('index'))

    else:
        return render_template('contact.html')


# Serve the logo image from the "images" directory outside the "static" folder
@app.route('/images/<filename>')
def custom_static(filename):
    return send_from_directory('images', filename)

if __name__ == '__main__':
    app.run(debug=True)
