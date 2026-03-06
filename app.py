from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load("final_model.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form.to_dict()  # form data from index.html
        input_df = pd.DataFrame([data])
        input_df = pd.get_dummies(input_df)

        model_columns = model.feature_names_in_
        input_df = input_df.reindex(columns=model_columns, fill_value=0)

        prediction = model.predict(input_df)[0]

        # Dynamic price range
        lower_bound = prediction - 25
        upper_bound = prediction + 25

        return render_template(
            'index.html',
            point_estimate=f"${prediction:.2f}",
            price_range=f"${lower_bound:.2f} - ${upper_bound:.2f}"
        )
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
