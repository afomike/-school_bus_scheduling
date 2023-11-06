from flask import Flask, request, render_template
import pandas as pd
import joblib
import pickle

with open(f'model\he_model_rfe.pkl', 'rb') as f:
    random_forest_model = pickle.load(f)
# Load the encoding configuration
encoding_config = joblib.load('model\encoding_config.pkl')
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_input = {
        'Arrival Time': request.form['Arrival_Time'],
        'Days': request.form['Days'],
        'Departure Point (Felele /Adankolo)': request.form['Departure_Point'],
        'Bus Route (Felele Road/Crusher Road)': request.form['Bus_Route'],
        'Arrival Point(Felele/Adankolo)': request.form['Arrival_Point']
    }

    # Convert 'Arrival Time' to total minutes
    
    arrival_time = pd.to_datetime(user_input['Arrival Time'], format='%H:%M').time()
    total_minutes = arrival_time.hour * 60 + arrival_time.minute
    user_input['Arrival Time'] = total_minutes

    # Convert user input into a DataFrame
    user_data = pd.DataFrame([user_input])

    # Perform the same one-hot encoding on user input
    X_new = pd.get_dummies(user_data, columns=encoding_config['categorical_columns'])
    X_new = X_new.reindex(columns=encoding_config['column_order'], fill_value=0)

    # Make predictions using the model
    prediction = random_forest_model.predict(X_new)

    predicted_total_minutes = round(prediction[0])

    # Convert back to hours and minutes
    predicted_hours = predicted_total_minutes // 60
    predicted_minutes = predicted_total_minutes % 60

    # Determine AM or PM
    if predicted_hours >= 12:
        am_pm = 'PM'
        if predicted_hours > 12:
            predicted_hours -= 12
    else:
        am_pm = 'AM'

    # Construct the string in the original format
    predicted_time_original_format = f'{predicted_hours:02d}:{predicted_minutes:02d} {am_pm}'

    return render_template('index.html', predicted_class=predicted_time_original_format)

if __name__ == '__main__':
    app.run(debug=True)

