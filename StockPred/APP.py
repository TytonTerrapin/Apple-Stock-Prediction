import numpy as np
import pandas as pd
from flask import Flask, render_template, request
import joblib


app = Flask(__name__)


regressor = joblib.load('models/regressor.pkl')
regressor2 = joblib.load('models/regressor2.pkl')
classifier = joblib.load('models/classifier.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    close_price = None
    next_day_values = None
    market_status = None

    if request.method == 'POST':
        
        if 'predict_regressor' in request.form:
            open_price = float(request.form['open'])
            high_price = float(request.form['high'])
            low_price = float(request.form['low'])
            
            input_data = pd.DataFrame([[open_price, high_price, low_price]], columns=['Open', 'High', 'Low'])
            close_price = regressor.predict(input_data)[0]

        
        elif 'predict_regressor2' in request.form:
            open_price = float(request.form['open'])
            high_price = float(request.form['high'])
            low_price = float(request.form['low'])
            close_price_input = float(request.form['close'])
            adj_close_price = float(request.form['adj_close'])
            
            input_data = pd.DataFrame([[open_price, high_price, low_price, close_price_input, adj_close_price]],
                                      columns=['Open', 'High', 'Low', 'Close', 'Adj Close'])
            next_day_values = regressor2.predict(input_data)[0]

        
        elif 'predict_classifier' in request.form:
            open_price = float(request.form['open'])
            high_price = float(request.form['high'])
            low_price = float(request.form['low'])
            
            input_data = pd.DataFrame([[open_price, high_price, low_price]], columns=['Open', 'High', 'Low'])
            market_status = classifier.predict(input_data)[0]
            status_dict = {0: 'Down', 1: 'Up', 2: 'Neutral'}
            market_status = status_dict.get(market_status, 'Unknown')

    return render_template('index.html', close_price=close_price, next_day_values=next_day_values, market_status=market_status)

if __name__ == '__main__':
    app.run(debug=True)
