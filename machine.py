import streamlit as st
import pandas as pd
import numpy as np
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# ✅ Streamlit Page Config
st.set_page_config(page_title="MedChain+ Chatbot", layout="centered")

# ✅ Load Datasets
@st.cache_data
def load_data():
    mh_df = pd.read_csv("mental_health_prediction_dataset.csv")
    dp_df = pd.read_csv("118medchain_chatbot_expanded_dataset.csv")
    return mh_df, dp_df

mh_df, dp_df = load_data()

# ✅ Text Preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text

# ✅ Train Mental Health Model
@st.cache_resource
def train_mental_health_model(df):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['User_Input'])
    y = df['Predicted_Condition']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, vectorizer

mh_model, mh_vectorizer = train_mental_health_model(mh_df)

# ✅ Train Disease Prediction Model
@st.cache_resource
def train_disease_model(df):
    df['Cleaned_Symptoms'] = df['Symptoms'].apply(preprocess_text)
    vectorizer = TfidfVectorizer(max_features=2000)
    X = vectorizer.fit_transform(df['Cleaned_Symptoms'])
    le = LabelEncoder()
    y = le.fit_transform(df['Disease'])

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, y)

    info = df[['Disease', 'Description', 'Precautions']].drop_duplicates(subset=['Disease']).set_index('Disease')
    return model, vectorizer, le, info

dp_model, dp_vectorizer, label_encoder, disease_info = train_disease_model(dp_df)

# ✅ Initialize Session State
if "feedback_data" not in st.session_state:
    st.session_state.feedback_data = {}

# ✅ Save Feedback Separately
def save_user_input(data, file_name):
    df = pd.DataFrame([data])
    if os.path.exists(file_name):
        df.to_csv(file_name, mode='a', header=False, index=False)
    else:
        df.to_csv(file_name, index=False)

# ✅ UI Layout
st.title("MedChain+ Chatbot")
st.write("Analyze your **mental health** or get **disease predictions** based on symptoms.")

choice = st.radio("What would you like to check?", ["Mental Health", "Disease Prediction"])

# ✅ Mental Health Prediction
if choice == "Mental Health":
    user_input = st.text_area("How are you feeling today?", placeholder="e.g., I feel anxious and sad...")

    if st.button("Predict Mental Health Condition"):
        if user_input.strip():
            vec = mh_vectorizer.transform([user_input])
            probs = mh_model.predict_proba(vec)[0]
            top3 = np.argsort(probs)[-3:][::-1]

            st.subheader(" Top 3 Possible Conditions")
            for i in range(3):
                condition = mh_model.classes_[top3[i]]
                desc = mh_df[mh_df['Predicted_Condition'] == condition]['Description'].iloc[0]
                precaution = mh_df[mh_df['Predicted_Condition'] == condition]['Precautions'].iloc[0]
                confidence = probs[top3[i]] * 100
                st.markdown(f"**{i+1}. {condition} ({confidence:.2f}%)**")
                st.markdown(f"- *Description:* {desc}")
                st.markdown(f"- *Precautions:* {precaution}")

            # ✅ Store for feedback
            st.session_state.feedback_data = {
                "User_Input": user_input,
                "Prediction": mh_model.classes_[top3[0]],
                "Helpful": ""
            }
        else:
            st.warning("Please enter how you're feeling.")

# ✅ Disease Prediction
elif choice == "Disease Prediction":
    symptom_input = st.text_area("Enter your symptoms", placeholder="e.g., fever cough fatigue")

    if st.button("Predict Disease"):
        if symptom_input.strip():
            clean_symptoms = preprocess_text(symptom_input)
            vectorized = dp_vectorizer.transform([clean_symptoms])
            probs = dp_model.predict_proba(vectorized)[0]
            top3 = np.argsort(probs)[-3:][::-1]

            st.subheader(" Top 3 Possible Diseases")
            for i in top3:
                disease = label_encoder.inverse_transform([i])[0]
                desc = disease_info.loc[disease, 'Description']
                precaution = disease_info.loc[disease, 'Precautions']
                confidence = probs[i] * 100

                st.markdown(f"**- {disease} ({confidence:.2f}%)**")
                st.markdown(f"  - *Description:* {desc}")
                st.markdown(f"  - *Precautions:* {precaution}")

            # ✅ Store for feedback
            st.session_state.feedback_data = {
                "User_Input": symptom_input,
                "Prediction": label_encoder.inverse_transform([top3[0]])[0],
                "Helpful": ""
            }
        else:
            st.warning("Please enter your symptoms.")

# ✅ Feedback Section
if st.session_state.feedback_data:
    feedback_data = st.session_state.feedback_data
    st.markdown("Was this prediction helpful?")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Yes"):
            feedback_data["Helpful"] = "Yes"
            if choice == "Mental Health":
                save_user_input(feedback_data, "mental_health_feedback.csv")
            else:
                save_user_input(feedback_data, "disease_prediction_feedback.csv")
            st.success("Thanks for your feedback!")
            st.session_state.feedback_data = {}

    with col2:
        if st.button("No"):
            feedback_data["Helpful"] = "No"
            if choice == "Mental Health":
                save_user_input(feedback_data, "mental_health_feedback.csv")
            else:
                save_user_input(feedback_data, "disease_prediction_feedback.csv")
            st.info("Thanks! We'll use this to improve.")
            st.session_state.feedback_data = {}
