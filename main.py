import os
# ===============================
# Step 1: Import Libraries
# ===============================
import numpy as np
import tensorflow as tf
import re
import streamlit as st
import time

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model


# ===============================
# Step 2: Page Config
# ===============================
st.set_page_config(
    page_title="IMDB Review Analyzer",
    page_icon="🎬",
    layout="centered"
)

# ===============================
# Step 3: Custom CSS Styling
# ===============================
st.markdown("""
    <style>
    body {
        background-color: #0f172a;
    }
    .main {
        background-color: #0f172a;
        color: white;
    }
    .title {
        text-align: center;
        font-size: 42px;
        font-weight: bold;
        color: #38bdf8;
        margin-bottom: 10px;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #94a3b8;
        margin-bottom: 30px;
    }
    .card {
        background-color: #1e293b;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0px 4px 20px rgba(0,0,0,0.3);
    }
    .result {
        font-size: 24px;
        font-weight: bold;
        text-align: center;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)


# ===============================
# Step 4: Load Model & Word Index
# ===============================
max_features = 10000

word_index = imdb.get_word_index()
model = load_model('imdbRNN.h5')


# ===============================
# Step 5: Preprocessing
# ===============================
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z ]', '', text)
    words = text.lower().split()

    encoded_review = []

    for word in words:
        idx = word_index.get(word, 2) + 3
        if idx >= max_features:
            idx = 2
        encoded_review.append(idx)

    return sequence.pad_sequences([encoded_review], maxlen=500)


# ===============================
# Step 6: UI Layout
# ===============================
st.markdown('<div class="title">🎬 IMDB Sentiment Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Analyze movie reviews using Deep Learning</div>', unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)

    user_input = st.text_area("✍️ Enter your movie review:", height=150)

    col1, col2 = st.columns([1,1])
    with col1:
        analyze = st.button("🚀 Analyze")
    with col2:
        clear = st.button("🧹 Clear")

    st.markdown('</div>', unsafe_allow_html=True)


# ===============================
# Step 7: Prediction + Animation
# ===============================
if analyze:

    if not user_input.strip():
        st.warning("⚠️ Please enter a valid review.")
    else:
        with st.spinner("🔍 Analyzing sentiment..."):
            time.sleep(1.5)  # animation feel

            preprocessed_input = preprocess_text(user_input)
            prediction = model.predict(preprocessed_input)

            score = float(prediction[0][0])
            sentiment = "Positive 😊" if score > 0.5 else "Negative 😞"

        # 🎯 Result display
        st.markdown(f'<div class="result">Sentiment: {sentiment}</div>', unsafe_allow_html=True)

        # 📊 Confidence bar
        st.progress(score)

        st.write(f"Confidence Score: **{score:.4f}**")


# ===============================
# Step 8: Footer
# ===============================
st.markdown("---")
