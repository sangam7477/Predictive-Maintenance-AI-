# 🔧 Predictive Maintenance System using AI

## 📌 Project Overview

This project is an **AI-based Predictive Maintenance System** that predicts machine failures using **Machine Learning**.

The system analyzes important machine parameters such as:

* Air Temperature
* Process Temperature
* Rotational Speed
* Torque
* Tool Wear

Based on these parameters, the system predicts whether a machine is likely to fail.

---

## 🚀 Features

* 🤖 Machine Failure Prediction
* 📊 Data Analysis
* 📈 Feature Importance Visualization
* 🔍 Confusion Matrix
* 🧠 Machine Learning Model Training
* 🌐 Flask Web Application
* 💻 User-Friendly Interface

---

## 🛠️ Technologies Used

* Python
* Machine Learning
* Scikit-learn
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Flask
* Joblib

---

## 📂 Project Structure

```text
Predictive-Maintenance-AI/
│
├── dataset/
│   └── ai4i2020.csv
│
├── models/
│   └── predictive_maintenance_model.pkl
│
├── static/
│   ├── confusion_matrix.png
│   └── feature_importance.png
│
├── templates/
│   └── index.html
│
├── app.py
├── train_model.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📊 Input Features

The Machine Learning model uses the following features:

1. Air Temperature **[K]**
2. Process Temperature **[K]**
3. Rotational Speed **[rpm]**
4. Torque **[Nm]**
5. Tool Wear **[min]**

---

## 🤖 Prediction

The model predicts:

* **0 → No Machine Failure**
* **1 → Machine Failure**

---

## ⚙️ Installation

### Clone the Repository

```bash
git clone https://github.com/sangam7477/Predictive-Maintenance-AI-.git
```

### Go to the Project Folder

```bash
cd Predictive-Maintenance-AI-
```

### Create a Virtual Environment

```bash
python -m venv .venv
```

### Activate the Virtual Environment

**Windows:**

```bash
.venv\Scripts\activate
```

### Install Required Libraries

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Project

### Train the Machine Learning Model

```bash
python train_model.py
```

### Start the Flask Application

```bash
python app.py
```

Open your browser and visit:

```text
http://127.0.0.1:5000
```

---

## 📈 Model Performance

The model achieved approximately:

* **Accuracy: 98.3%**

The project also provides:

* Confusion Matrix
* Feature Importance Analysis
* Machine Failure Prediction

---

## 👨‍💻 Author

**Sangam Saini**

BCA — AI/ML Specialization

---

## ⭐ Support

If you like this project, please give it a **⭐ Star** on GitHub!
