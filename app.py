from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import json
import pandas as pd
from sklearn.linear_model import LinearRegression

app = Flask(__name__)
CORS(app)

# Updated prediction function with R² scores
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
    r2_scores = {}

    for col in ['fat', 'smm', 'weight']:
        model = LinearRegression()
        X = df[['time_ordinal']]
        y = df[col]
        model.fit(X, y)
        r2_scores[col] = round(model.score(X, y), 4)  # R² value rounded for readability
        predictions[col] = model.predict(future_ordinals.values.reshape(-1, 1))
    
    print("Predictions completed.")  # Debugging print statement

    pred_df = pd.DataFrame(predictions)
    pred_df['date'] = pred_df['date'].astype(str)  # Convert datetime to string for JSON serialization
    return pred_df, r2_scores

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("Starting prediction endpoint...")  # Debugging print statement
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        file = request.files['file']
        target_date = request.form.get('target_date', "2025-07-05")
        predictions, r2_scores = predict_weekly_until_fixed(file, target_date)
        result = {
            'predictions': predictions.to_dict(orient='records'),
            'r2_scores': r2_scores
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/status', methods=['GET'])
def status():
    return jsonify({'status': 'Running'})

if __name__ == '__main__':
    app.run(debug=True)
