# 📊 Student Exam Performance Indicator

## 📌 Project Overview
This project is an **end-to-end Machine Learning web application** that predicts a student's **math score** based on academic and background-related inputs such as gender, race/ethnicity, parental level of education, lunch type, test preparation course, reading score, and writing score.

The project includes:
- a complete training pipeline
- data preprocessing
- model prediction pipeline
- Flask-based web application for real-time predictions

---

## 🌐 Live Demo
👉 Deployed App: **https://mlproject-5-yfln.onrender.com**

Example:
`https://your-app-link.com`

---

## 🚀 Features
- Predicts student **math performance**
- End-to-end **ML pipeline**
- Flask-based **web interface**
- Uses saved **model** and **preprocessor**
- User-friendly input form
- Real-time prediction output

---

## 🧠 Tech Stack
- Python 🐍
- Flask
- Pandas
- NumPy
- Scikit-learn
- CatBoost
- XGBoost
- Matplotlib
- Seaborn

---

## 📂 Project Structure
mlproject/
│
├── .ebxtensions/                 # Deployment-related config
├── .vscode/                      # VS Code settings
├── artifacts/                    # Saved model and preprocessing objects
├── catboost_info/                # CatBoost training info
├── notebook/                     # Dataset and notebooks
├── src/                          # Source code for pipeline and components
├── templates/                    # HTML templates
├── .gitignore
├── README.md
├── application.py                # Flask app entry point
├── main.py                       # Data ingestion / training-related script
├── requirements.txt
└── setup.py

---

## ❓ Problem Statement
Student performance prediction helps in understanding how different factors influence academic outcomes.  
This project predicts the **math score** of a student using other available features.

### Example:
If a student has:
- strong reading score 📘
- strong writing score ✍️
- completed test preparation ✅

the model can estimate the expected math score.

---

## 📥 Input Features
The application takes the following inputs:

- Gender
- Race or Ethnicity
- Parental Level of Education
- Lunch Type
- Test Preparation Course
- Reading Score
- Writing Score

### Example Input
- Gender: Female
- Race/Ethnicity: Group C
- Parental Education: Bachelor's Degree
- Lunch: Standard
- Test Preparation Course: Completed
- Reading Score: 78
- Writing Score: 80

---

## 📤 Output
The model predicts:

- **Math Score**

### Example Output
- Predicted Math Score: **82.4**

---
## 📤 screenshot
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/06fd10c0-5d3d-4dbe-a868-f7ab45207f98" />

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/c3e008c8-877f-4ba8-b50c-fd776c2e5571" />


## ⚙️ How It Works
1. User enters student details in the web form
2. Input data is converted into a DataFrame
3. Saved preprocessing object transforms the input
4. Trained model makes prediction
5. Predicted math score is shown on the webpage

---

## 🧪 Machine Learning Workflow
This project follows a typical ML workflow:

- Data Ingestion
- Data Transformation
- Model Training
- Model Evaluation
- Prediction Pipeline
- Web App Deployment

---

## 🖥️ Web Application
The Flask app provides two main routes:

- `/` → Home page
- `/predictdata` → Prediction page

Users can fill in the form and get the prediction instantly.

---

## 📁 Important Files
- `application.py` → runs the Flask application
- `main.py` → handles data ingestion / training flow
- `src/pipeline/predict_pipeline.py` → prediction logic
- `artifacts/model.pkl` → trained model
- `artifacts/preprocessor.pkl` → preprocessing object
- `templates/home.html` → frontend prediction form

---

## ▶️ Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/divyasreevemula918/mlproject.git
cd mlproject
