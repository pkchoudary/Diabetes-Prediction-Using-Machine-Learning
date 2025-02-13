# Diabetes Prediction Using Machine Learning

## Overview
This project aims to predict whether a person has diabetes based on medical diagnostic measurements. The model is trained using machine learning algorithms on a dataset containing relevant health indicators.

## Features
- Data preprocessing and feature selection
- Multiple machine learning models for prediction
- Model evaluation and performance comparison
- User-friendly interface for making predictions

## Dataset
The dataset used for training and evaluation comes from the **Pima Indians Diabetes Database**. It includes medical parameters such as:
- Number of pregnancies
- Glucose level
- Blood pressure
- Skin thickness
- Insulin level
- BMI (Body Mass Index)
- Diabetes Pedigree Function
- Age

## Technologies Used
- Python
- Pandas & NumPy (Data Handling)
- Scikit-learn (Machine Learning)
- Matplotlib & Seaborn (Data Visualization)
- Flask (Web Application - optional)

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/your-username/Diabetes-Prediction-Using-Machine-Learning.git
   ```
2. Navigate to the project directory:
   ```sh
   cd Diabetes-Prediction-Using-Machine-Learning
   ```
3. Install the required dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage
1. Run the Jupyter Notebook to train and evaluate models:
   ```sh
   jupyter notebook
   ```
2. If using a web application, start the Flask server:
   ```sh
   python app.py
   ```
3. Open the browser and go to:
   ```
   http://127.0.0.1:5000/
   ```

## Model Performance
The following machine learning models were trained and evaluated:
- Logistic Regression
- Random Forest Classifier
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)

Performance metrics such as accuracy, precision, recall, and F1-score are used to compare model effectiveness.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for improvements.

## License
This project is licensed under the MIT License.

## Contact
For questions or collaboration, create an issue in this repository.

