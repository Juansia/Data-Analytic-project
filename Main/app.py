from flask import Flask, render_template, request, jsonify
import prediction_module
import traceback

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the days parameter from the form, default to 3 if not provided
        days = int(request.form.get('days', 3))
        
        # Get commodity parameter
        commodity = request.form.get('commodity', 'brent').lower()
        
        # Validate that days is one of the expected values, if not default to the closest
        if days not in [3, 5, 22]:
            if days < 3:
                days = 3
            elif days < 5:
                days = 3
            elif days < 22:
                days = 5
            else:
                days = 22
        
        # Validate commodity
        if commodity not in ['brent', 'sugar']:
            commodity = 'brent'
        
        # Print debug information
        print(f"Received prediction request: days={days}, commodity={commodity}")
        
        # Run the prediction with the specified number of days and commodity
        result = prediction_module.run_prediction(days_ahead=days, commodity=commodity)
        
        # Debug the result
        print(f"Prediction successful for {commodity} {days}-day forecast")
        
        return jsonify(result)
    
    except Exception as e:
        # Log the full error
        print(f"Error in prediction: {str(e)}")
        print(traceback.format_exc())
        
        # Return error response
        return jsonify({
            "error": str(e),
            "message": "Failed to generate prediction. See server logs for details."
        }), 500

@app.after_request
def add_header(response):
    """Add headers to allow CORS."""
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'OPTIONS, POST, GET'
    return response

if __name__ == '__main__':
    app.run(debug=True)