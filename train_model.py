# ============================================================
#  CROP RECOMMENDATION SYSTEM — MODEL TRAINING
#  For 1st Year Engineering Students
#  Run this ONCE to train and save the model.
# ============================================================

# ---- STEP 0: Import Libraries ----
import pandas as pd          # for reading CSV files
import numpy as np           # for number arrays
import pickle                # for saving/loading models
from sklearn.ensemble import RandomForestClassifier   # our ML model
from sklearn.model_selection import train_test_split  # to split data
from sklearn.metrics import accuracy_score            # to check accuracy
import matplotlib.pyplot as plt  # for charts (optional)

print("📦 All libraries imported successfully!\n")

# ============================================================
# STEP 1: LOAD THE DATASET
# ============================================================
# Read the CSV file into a DataFrame (like an Excel table in Python)
df = pd.read_csv('Crop_recommendation.csv')

print("📊 Dataset loaded!")
print(f"   Rows: {df.shape[0]},  Columns: {df.shape[1]}")
print(f"   Columns: {list(df.columns)}")
print(f"\n   First 3 rows:\n{df.head(3)}\n")

# ============================================================
# STEP 2: UNDERSTAND THE DATA
# ============================================================
print("🌾 Crops in dataset:", df['label'].unique())
print(f"   Total unique crops: {df['label'].nunique()}\n")

print("📈 Basic statistics:")
print(df.describe().round(2))
print()

# ============================================================
# STEP 3: ENCODE CROP NAMES TO NUMBERS
# ============================================================
# ML models work with numbers, not text.
# So we convert "rice" → 0, "maize" → 1, etc.

crop_list = sorted(df['label'].unique())  # sorted list of all crops
crop_to_num = {crop: i for i, crop in enumerate(crop_list)}  # dictionary: crop → number

print("🔢 Crop name → Number mapping:")
for crop, num in crop_to_num.items():
    print(f"   {num:2d} = {crop}")
print()

# Add a new column 'crop_num' with the encoded numbers
df['crop_num'] = df['label'].map(crop_to_num)

# ============================================================
# STEP 4: SPLIT INTO FEATURES (X) AND TARGET (y)
# ============================================================
# X = inputs (N, P, K, temperature, humidity, ph, rainfall)
# y = output (crop number)

X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['crop_num']

print(f"✅ Features (X) shape: {X.shape}")   # Should be (2200, 7)
print(f"✅ Target (y) shape:   {y.shape}\n") # Should be (2200,)

# ============================================================
# STEP 5: SPLIT INTO TRAIN AND TEST DATA
# ============================================================
# We use 80% of data for training, 20% for testing
# test_size=0.2 means 20% goes to test set
# random_state=42 makes it reproducible (same split every time)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"📚 Training data:  {X_train.shape[0]} rows")
print(f"🧪 Testing data:   {X_test.shape[0]} rows\n")

# ============================================================
# STEP 6: TRAIN THE MACHINE LEARNING MODEL
# ============================================================
# Random Forest = like asking 100 experts and taking majority vote
# n_estimators=100 means we use 100 decision trees

print("🤖 Training the Random Forest model...")
print("   (This might take a few seconds...)\n")

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)  # THIS is where the model learns!

print("✅ Model training complete!\n")

# ============================================================
# STEP 7: TEST THE MODEL
# ============================================================
# Now we test on data the model has NEVER seen before (X_test)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 Model Accuracy: {accuracy * 100:.2f}%")
print("   (A score above 90% is excellent!)\n")

# ============================================================
# STEP 8: FEATURE IMPORTANCE (OPTIONAL - NICE TO SHOW IN PROJECT)
# ============================================================
features = ['N', 'P', 'K', 'Temperature', 'Humidity', 'pH', 'Rainfall']
importances = model.feature_importances_

print("📊 Which features matter most:")
for feat, imp in sorted(zip(features, importances), key=lambda x: -x[1]):
    bar = '█' * int(imp * 40)
    print(f"   {feat:12s} {bar} {imp:.3f}")
print()

# ============================================================
# STEP 9: SAVE THE TRAINED MODEL
# ============================================================
# We save the model so the Flask app can use it without retraining

with open('crop_model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Also save the crop list so Flask knows how to decode predictions
with open('crop_names.pkl', 'wb') as file:
    pickle.dump(crop_list, file)

print("💾 Model saved as: crop_model.pkl")
print("💾 Crop list saved as: crop_names.pkl\n")

# ============================================================
# STEP 10: TEST ONE PREDICTION MANUALLY
# ============================================================
print("🧪 Manual test — Rice conditions:")
test_input = np.array([[90, 42, 43, 20.8, 82.0, 6.5, 202.9]])
#                       N   P   K  temp  hum   ph  rainfall

pred_num = model.predict(test_input)[0]
pred_crop = crop_list[pred_num]
print(f"   Input:  N=90, P=42, K=43, Temp=20.8, Humidity=82, pH=6.5, Rainfall=202.9")
print(f"   Result: {pred_crop.upper()} (should be RICE)")
print()

print("=" * 55)
print("✅ All done! Your model is ready.")
print("   Next step: Run 'python app.py' to start the web app")
print("=" * 55)
