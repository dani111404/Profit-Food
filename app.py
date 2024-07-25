from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the model and label encoder
model = joblib.load('decision_tree_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        menu_category = request.form.get('MenuCategory')
        price = float(request.form.get('Price'))

        # Convert menu category to numeric
        menu_category_num = label_encoder.transform([menu_category])[0]

        # Validate input
        if price < 0:
            raise ValueError("Price cannot be negative")

        # Create dataframe for prediction
        df = pd.DataFrame([[menu_category_num, price]], columns=['MenuCategory', 'Price'])

        # Make prediction
        prediction = model.predict(df)

        # Decode the label (0 atau 1) menjadi hasil yang berarti
        result = 'Profit' if prediction[0] == 1 else 'Not Profit'
    except Exception as e:
        result = str(e)
    
    return render_template('index.html', prediction_text=f'Result: {result}')

if __name__ == '__main__':
    app.run(debug=True)
