# 🌾 Crop Recommendation System
### A Machine Learning Project for 1st Year Engineering Students

---

## 📌 What Does This Project Do?

This project uses **Machine Learning** to recommend the best crop to grow based on:
- Soil nutrients (Nitrogen, Phosphorus, Potassium)
- Climate conditions (Temperature, Humidity, Rainfall)
- Soil quality (pH level)

A farmer enters 7 values → the ML model analyses them → it suggests the best crop!

---

## 📁 Project Files

```
crop-recommendation/
│
├── train_model.py          ← Run this FIRST to train the ML model
├── app.py                  ← Run this SECOND to start the web app
├── Crop_recommendation.csv ← Dataset with 2200 crop examples
│
├── templates/
│   └── index.html          ← The webpage (what users see)
│
├── crop_model.pkl          ← Saved ML model (created after training)
├── crop_names.pkl          ← Saved crop list (created after training)
│
├── requirements.txt        ← Python libraries needed
└── README.md               ← This file!
```

---

## 🚀 How to Run (Step by Step)

### Step 1: Install Python Libraries

Open terminal/command prompt and run:
```bash
pip install flask numpy pandas scikit-learn matplotlib
```

### Step 2: Train the ML Model

```bash
python train_model.py
```

You'll see output like:
```
✅ Model training complete!
🎯 Model Accuracy: 99.32%
💾 Model saved as: crop_model.pkl
```

### Step 3: Start the Web App

```bash
python app.py
```

You'll see:
```
 * Running on http://127.0.0.1:5000
```

### Step 4: Open in Browser

Go to: **http://127.0.0.1:5000**

Fill in the form and click **"Recommend Best Crop"**!

---

## 🧠 How Does Machine Learning Work Here?

### The Data
We have **2200 examples** of crops with their ideal conditions.
Each example has 7 numbers → 1 crop name.

Example rows from the dataset:
| N | P | K | Temp | Humidity | pH | Rainfall | Crop |
|---|---|---|------|----------|----|----------|------|
| 90 | 42 | 43 | 20.8 | 82 | 6.5 | 203 | rice |
| 85 | 58 | 41 | 21.8 | 80 | 7.0 | 227 | rice |
| 60 | 55 | 44 | 23.0 | 82 | 7.8 | 263 | jute |

### The Model: Random Forest
Think of it like asking **100 experts** for advice:
- Each "expert" is a Decision Tree
- All 100 trees vote for a crop
- The crop with the most votes wins!

### The Training Process
```
Raw Data → Split 80%/20% → Train on 80% → Test on 20% → Save Model
```

- **Training data** = what the model learns from
- **Test data** = data the model has never seen (used to measure accuracy)

---

## 📊 What is Accuracy?

After training, we test the model on new data.
- **99% accuracy** = out of 100 predictions, 99 are correct!
- This is excellent for a 1st year project 😊

---

## 🔢 Input Value Ranges

| Input | Unit | Min | Max | Example |
|-------|------|-----|-----|---------|
| Nitrogen (N) | kg/ha | 0 | 140 | 90 |
| Phosphorus (P) | kg/ha | 5 | 145 | 42 |
| Potassium (K) | kg/ha | 5 | 205 | 43 |
| Temperature | °C | 8 | 44 | 25 |
| Humidity | % | 14 | 100 | 80 |
| pH | 0-14 | 3.5 | 9.9 | 6.5 |
| Rainfall | mm | 20 | 300 | 200 |

---

## 🌱 Crops the System Can Predict (22 crops)

apple, banana, blackgram, chickpea, coconut, coffee, cotton, grapes, jute, kidneybeans, lentil, maize, mango, mothbeans, mungbean, muskmelon, orange, papaya, pigeonpeas, pomegranate, **rice**, watermelon

---

## 🧪 Test Cases (Try These!)

| N | P | K | Temp | Humidity | pH | Rainfall | Expected |
|---|---|---|------|----------|----|----------|---------|
| 90 | 42 | 43 | 20.8 | 82 | 6.5 | 203 | Rice |
| 0 | 20 | 10 | 27 | 85 | 5.7 | 220 | Coconut |
| 60 | 55 | 44 | 23 | 82 | 7.8 | 263 | Jute |
| 30 | 60 | 200 | 25 | 50 | 6.5 | 100 | Grapes |

---

## 📚 Key Concepts Learned

By completing this project, you learned:

1. **Python Basics** — Reading files, functions, loops
2. **Pandas** — Working with datasets (like Excel in Python)
3. **Machine Learning** — Training a model on data
4. **Flask** — Building a web application
5. **HTML/CSS** — Creating a webpage interface
6. **Pickle** — Saving and loading ML models

---

## 🛠️ Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError` | Run `pip install flask sklearn pandas numpy` |
| `FileNotFoundError: crop_model.pkl` | Run `python train_model.py` first! |
| `Port already in use` | Change port: `app.run(port=5001)` |
| Blank page | Check terminal for error messages |

---

## 🎓 For Your Presentation

**Explain these 3 things clearly:**

1. **The Problem** — Farmers don't always know which crop suits their soil
2. **The Solution** — ML model trained on 2200 soil/climate samples
3. **The Result** — 99%+ accuracy with a simple web interface

**Key talking points:**
- Used Random Forest (ensemble of 100 decision trees)
- Dataset has 22 different crops
- Built a web interface using Flask

---

Made with 🌱 Python + Flask + Scikit-Learn | 1st Year Engineering Project
