# -*- coding: utf-8 -*-
"""
Created on Mon Nov 6 15:20:48 2023

@author: Punam
"""

from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__, template_folder="templates")

# Load the saved model
model_file_path = "D:/Job_ready/projects/hotel_booking_Main/trained_model.sav"

try:
    loaded_model = pickle.load(open(model_file_path, 'rb'))
except Exception as e:
    print(f"Error loading the model: {e}")
    loaded_model = None

# Create a function for prediction
def diabetes_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshaped)
    return prediction[0]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        input_data = {
            'hotel': int(request.form['hotel']),
            'meal': int(request.form['meal']),
            'market_segment': int(request.form['market_segment']),
            'distribution_channel': int(request.form['distribution_channel']),
            'reserved_room_type': int(request.form['reserved_room_type']),
            'deposit_type': int(request.form['deposit_type']),
            'customer_type': int(request.form['customer_type']),
            'year': float(request.form['year']),
            'month': int(request.form['month']),
            'day': int(request.form['day']),
            'lead_time': float(request.form['lead_time']),
            'arrival_date_week_number': float(request.form['arrival_date_week_number']),
            'arrival_date_day_of_month': float(request.form['arrival_date_day_of_month']),
            'stays_in_weekend_nights': int(request.form['stays_in_weekend_nights']),
            'stays_in_week_nights': int(request.form['stays_in_week_nights']),
            'adults': int(request.form['adults']),
            'children': float(request.form['children']),
            'babies': int(request.form['babies']),
            'is_repeated_guest': int(request.form['is_repeated_guest']),
            'previous_cancellations': int(request.form['previous_cancellations']),
            'previous_bookings_not_canceled': int(request.form['previous_bookings_not_canceled']),
            'agent': float(request.form['agent']),
            'company': float(request.form['company']),
            'adr': float(request.form['adr']),
            'required_car_parking_spaces': int(request.form['required_car_parking_spaces']),
            'total_of_special_requests': int(request.form['total_of_special_requests'])
        }

        result = diabetes_prediction(list(input_data.values()))

        if result == 0:
            diagnosis = 'The Customer Does Not Cancelled Their Hotel Booking'
        else:
            diagnosis = 'The Customer Cancelled Their Hotel Booking'

        return render_template('result.html', diagnosis=diagnosis)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0",port=5000)


