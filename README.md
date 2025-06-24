# Mental-health-and-Disease-Prediction-Chatbot-
MedChain+ Chatbot is a beginner-friendly Streamlit application that predicts mental health conditions or diseases based on user input. It uses machine learning to analyze emotional or symptom-based text and provides helpful descriptions, precautions, and collects user feedback to improve future predictions.

MedChain+ Chatbot

Overview:
---------
MedChain+ is a beginner-friendly Streamlit-based chatbot application that offers two essential features:
1. Mental Health Prediction based on user input (feelings or emotional states).
2. Disease Prediction based on symptoms.

It uses machine learning models (Random Forest) trained on custom datasets to provide accurate predictions, along with helpful descriptions and precautions. It also collects user feedback to improve the model in the future.

Features:
---------
 Predict Mental Health Conditions from emotional text.
 Predict Diseases from symptom descriptions.
 View Top 3 most probable predictions with confidence score.
 Get disease descriptions and precautionary advice.
 Collect user feedback (Was this helpful? Yes/No).
 Save feedback into separate CSV files for retraining.

How It Works:
-------------
1. The user selects whether they want a Mental Health or Disease prediction.
2. They enter their emotional state or symptoms.
3. The app processes the input and shows the top 3 predictions.
4. The user gives feedback, which is saved locally for future improvement.

Files Required:
---------------
- `mental_health_prediction_dataset.csv`: Training data for mental health model.
- `118medchain_chatbot_expanded_dataset.csv`: Dataset for disease prediction model.
- `mental_health_feedback.csv`: Feedback storage for mental health.
- `disease_prediction_feedback.csv`: Feedback storage for disease prediction.

Technologies Used:
------------------
- Python
- Streamlit
- Pandas, NumPy
- Scikit-learn (RandomForestClassifier, TfidfVectorizer, LabelEncoder)
- Regex
- OS

How to Run:
-----------
1. Install requirements:
   pip install streamlit pandas scikit-learn

2. Run the app:
   streamlit run your_script_name.py

3. Make sure your CSV files are in the same directory.

Author:
-------
Developed by Mohammed Rayyan Shaikh

