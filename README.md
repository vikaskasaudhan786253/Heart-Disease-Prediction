# Heart-Disease-Prediction


https://heart-disease-prediction-3am7z3vwumvnzt83xwqupu.streamlit.app/


# ❤️ Heart Disease Prediction System

## 📌 Overview

This project is a **Machine Learning-based Heart Disease Prediction System** that predicts whether a person is likely to have heart disease based on medical attributes.

The application is built using:

* Multiple ML models (Logistic Regression, Decision Tree, Random Forest)
* A Streamlit web app for user interaction
* Pre-trained `.pkl` models for real-time prediction

---

## 🚀 Features

* Predict heart disease risk instantly
* Multiple ML algorithms implemented:

  * Logistic Regression
  * Decision Tree
  * Random Forest
* Cleaned and processed dataset
* Model comparison using different parameters
* Interactive web interface using Streamlit

---

## 📂 Project Structure

```
├── app.py                         # Main Streamlit app
├── decisiontree.py               # Decision Tree model logic
├── logisticregression.py         # Logistic Regression model logic
├── randomforest.py               # Random Forest model logic

├── model_dt.pkl                  # Trained Decision Tree model
├── model_lr.pkl                  # Trained Logistic Regression model
├── model_rf.pkl                  # Trained Random Forest model

├── train.csv                     # Training dataset
├── test.csv                      # Test dataset
├── train_cleaned.csv             # Cleaned dataset
├── collected_data.csv            # Processed data

├── all_combination_param_decision_tree.csv
├── all_combinations_random_forest.csv

├── Decision_Tree.ipynb           # Notebook for Decision Tree
├── Logistic_Regression.ipynb     # Notebook for Logistic Regression
├── Random_Forest.ipynb           # Notebook for Random Forest

├── pages/                        # Additional Streamlit pages
├── requirements.txt              # Dependencies
├── README.md                    # Project documentation
```

---

## 🧠 Machine Learning Models

| Model               | Description                        |
| ------------------- | ---------------------------------- |
| Logistic Regression | Baseline linear model              |
| Decision Tree       | Rule-based classification          |
| Random Forest       | Ensemble model for better accuracy |

---

## 📊 Dataset Features

Typical features used:

* Age
* Sex
* Chest Pain Type
* Resting Blood Pressure
* Cholesterol
* Fasting Blood Sugar
* Resting ECG
* Max Heart Rate
* Exercise Induced Angina
* Oldpeak
* Slope
* Number of Major Vessels
* Thal

---

## ⚙️ Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/heart-disease-prediction.git
cd heart-disease-prediction
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

---

## 📈 Model Training

You can train models using:

* `Decision_Tree.ipynb`
* `Logistic_Regression.ipynb`
* `Random_Forest.ipynb`

---

## 📌 How It Works

1. User inputs medical data through the UI
2. Data is preprocessed
3. Selected model predicts heart disease risk
4. Output is displayed instantly

---

## 🛠️ Future Improvements

* Add Deep Learning models
* Improve feature engineering
* Deploy on cloud (AWS / Streamlit Cloud)
* Add SHAP explainability
* Real-time health report generation

---

## 🤝 Contributing

Contributions are welcome! Feel free to fork and improve the project.

---

## 📜 License

This project is open-source and available under the MIT License.

---

## 👨‍💻 Author

**Vikas Kasaudhan**

---

## ⭐ Support

If you like this project, give it a ⭐ on GitHub!
