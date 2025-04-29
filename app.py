from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Load the pre-trained model
try:
    model = joblib.load('car_price_model.pkl')
except FileNotFoundError:
    model = None
    print("Warning: car_price_model.pkl not found. Model functionality will be limited until training.")

# Load the training data
try:
    df = pd.read_csv('car_purchasing.csv', encoding='latin1')
except FileNotFoundError:
    df = None
    print("Error: car_purchasing.csv not found. Training functionality will be unavailable.")

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', model_available=(model is not None), data_available=(df is not None))

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Please train the model first.'})

    try:
        annual_salary = float(request.form['annual_salary'])
        credit_card_debt = float(request.form['credit_card_debt'])
        net_worth = float(request.form['net_worth'])
        age = int(request.form['age'])

        prediction = model.predict([[annual_salary, credit_card_debt, net_worth, age]])[0]
        return jsonify({'prediction': f'${prediction:.2f}'})
    except ValueError:
        return jsonify({'error': 'Invalid input. Please enter numeric values.'})
    except Exception as e:
        return jsonify({'error': f'An error occurred during prediction: {str(e)}'})

@app.route('/train', methods=['POST'])
def train_model():
    global model, df
    if df is None:
        return jsonify({'error': 'Training data not loaded.'})

    try:
        X = df[['annual Salary', 'credit card debt', 'net worth', 'age']]
        y = df['car purchase amount']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        new_model = LinearRegression()
        new_model.fit(X_train, y_train)
        joblib.dump(new_model, 'car_price_model.pkl')
        model = new_model  # Update the loaded model
        return jsonify({'message': 'Model trained and saved successfully!'})
    except Exception as e:
        return jsonify({'error': f'An error occurred during training: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True)

