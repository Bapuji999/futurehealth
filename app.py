from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import json
import pandas as pd
from sklearn.linear_model import LinearRegression

app = Flask(__name__)
CORS(app)

# Updated prediction function using file content
def predict_weekly_until_fixed(file_content, target_date: str = "2025-07-05"):
    print("Starting prediction function...")  # Debugging print statement
    raw_data = json.load(file_content)
    df = pd.DataFrame(raw_data)
    df['datetime'] = pd.to_datetime(df['datetime']).dt.tz_localize(None)
    df['time_ordinal'] = df['datetime'].map(datetime.toordinal)
    
    print("Data loaded and transformed.")  # Debugging print statement

    start_date = df['datetime'].max()
    end_date = pd.to_datetime(target_date)
    future_dates = pd.date_range(start=start_date, end=end_date, freq='7D')
    future_ordinals = future_dates.map(datetime.toordinal)

    predictions = {'date': future_dates}
    for col in ['fat', 'smm', 'weight']:
        model = LinearRegression()
        model.fit(df[['time_ordinal']], df[col])
        predictions[col] = model.predict(future_ordinals.values.reshape(-1, 1))
    
    print("Predictions completed.")  # Debugging print statement

    return pd.DataFrame(predictions)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("Starting prediction function...")  # Debugging print statement
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        file = request.files['file']
        target_date = request.form.get('target_date', "2025-07-05")
        predictions = predict_weekly_until_fixed(file, target_date)
        result = predictions.to_dict(orient='records')
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/status', methods=['GET'])
def status():
    return jsonify({'status': 'Running'})

if __name__ == '__main__':
    app.run(debug=True)
