# ============================================================
# CROP RECOMMENDATION SYSTEM - Flask Web App
# For 1st Year Engineering Students
# ============================================================

# IMPORTS: We need these libraries
from flask import Flask, request, render_template  # Flask = web framework
import numpy as np    # NumPy = for number arrays
import pickle         # Pickle = for loading saved ML model

# ============================================================
# STEP 1: CREATE THE FLASK APP
# ============================================================
app = Flask(__name__)   # Create our web application

# ============================================================
# STEP 2: LOAD THE TRAINED ML MODEL
# ============================================================
# We saved the model after training, now we load it
with open('crop_model.pkl', 'rb') as file:
    model = pickle.load(file)

print("✅ Model loaded successfully!")

# ============================================================
# STEP 3: LIST OF ALL 22 CROPS THE MODEL CAN PREDICT
# ============================================================
crop_names = [
    'apple', 'banana', 'blackgram', 'chickpea', 'coconut',
    'coffee', 'cotton', 'grapes', 'jute', 'kidneybeans',
    'lentil', 'maize', 'mango', 'mothbeans', 'mungbean',
    'muskmelon', 'orange', 'papaya', 'pigeonpeas', 'pomegranate',
    'rice', 'watermelon'
]

# ============================================================
# STEP 4: DEFINE ROUTES (PAGES OF OUR WEBSITE)
# ============================================================

# HOME PAGE - Just shows the form
@app.route('/')
def home():
    """Shows the main page with the input form"""
    return render_template('index.html')


# PREDICT PAGE - Gets the form data and makes prediction
@app.route('/predict', methods=['POST'])
def predict():
    """
    This function runs when user clicks the 'Recommend Crop' button.
    It:
    1. Gets the values from the form
    2. Puts them into an array
    3. Feeds the array into the ML model
    4. Shows the result to the user
    """

    # ----------------------------------------
    # GET VALUES FROM THE HTML FORM
    # ----------------------------------------
    N = float(request.form['Nitrogen'])        # Nitrogen content
    P = float(request.form['Phosphorus'])      # Phosphorus content
    K = float(request.form['Potassium'])       # Potassium content
    temperature = float(request.form['Temperature'])   # Temperature in Celsius
    humidity = float(request.form['Humidity'])         # Humidity percentage
    ph = float(request.form['pH'])                     # Soil pH value
    rainfall = float(request.form['Rainfall'])         # Rainfall in mm

    # ----------------------------------------
    # CREATE INPUT ARRAY FOR THE MODEL
    # The model expects a 2D array: [[N, P, K, temp, humidity, ph, rainfall]]
    # ----------------------------------------
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

    # ----------------------------------------
    # MAKE PREDICTION USING ML MODEL
    # The model returns a number (0-21), we map it to crop name
    # ----------------------------------------
    prediction = model.predict(input_data)     # Get prediction number
    predicted_crop = crop_names[prediction[0]] # Convert number to crop name

    # ----------------------------------------
    # SHOW RESULT TO USER
    # ----------------------------------------
    result = f"🌱 Best Crop for Your Soil: {predicted_crop.upper()}"

    print(f"📊 Input: N={N}, P={P}, K={K}, Temp={temperature}, Humidity={humidity}, pH={ph}, Rainfall={rainfall}")
    print(f"🌾 Predicted Crop: {predicted_crop}")

    return render_template('index.html', result=result, crop=predicted_crop)


# ============================================================
# STEP 5: RUN THE APP
# ============================================================
if __name__ == '__main__':
    # debug=True means it auto-reloads when you change code
    app.run(debug=True)
