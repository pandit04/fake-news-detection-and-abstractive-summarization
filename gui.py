import os
import torch
import streamlit as st
import torch.nn.functional as F
import numpy as np
import re
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForSeq2SeqLM
from lime.lime_text import LimeTextExplainer

# =======================
# ✅ 1. Set Device to CPU (For Models Trained on GPU)
# =======================
device = torch.device("cpu")  # Force CPU usage

# =======================
# ✅ 2. Define Model Paths
# =======================
fake_news_model_path = "FAKE_NEWS/best_fake_news_model"
summarization_model_path = "FAKE_NEWS/best_summarization_model"

# =======================
# ✅ 3. Check if Models Exist
# =======================
if not os.path.exists(fake_news_model_path) or not os.path.exists(summarization_model_path):
    st.error("❌ Extracted models not found! Ensure they are placed inside 'FAKE_NEWS/'")
    st.stop()

# =======================
# ✅ 4. Load Fake News Detection Model (Ensure CPU Compatibility)
# =======================
st.info("🔄 Loading Fake News Detection Model...")
fake_news_tokenizer = AutoTokenizer.from_pretrained(fake_news_model_path, use_fast=True)  # Use fast tokenizer
fake_news_model = AutoModelForSequenceClassification.from_pretrained(
    fake_news_model_path, 
    torch_dtype=torch.float32  # Convert GPU-trained model to CPU
).to(device)

# =======================
# ✅ 5. Load Summarization Model (Ensure CPU Compatibility)
# =======================
st.info("🔄 Loading Summarization Model...")
summarization_tokenizer = AutoTokenizer.from_pretrained(summarization_model_path, use_fast=True)  # Use fast tokenizer
summarization_model = AutoModelForSeq2SeqLM.from_pretrained(
    summarization_model_path, 
    torch_dtype=torch.float32  # Convert GPU-trained model to CPU
).to(device)

st.success("✅ Models Loaded on CPU Successfully!")

# =======================
# ✅ 6. Initialize LIME Explainer
# =======================
explainer = LimeTextExplainer(class_names=['Real News', 'Fake News'])

# =======================
# ✅ 7. Streamlit UI Setup
# =======================
st.title("📰 Fake News Detection & Summarization (Hindi Supported)")

# Text Input (Supports Hindi)
news_text = st.text_area("Enter News Text (in Hindi or English):", height=200)

# =======================
# ✅ 8. Analyze Button
# =======================
if st.button("Analyze News"):
    if news_text.strip() == "":
        st.warning("⚠ Please enter a news article to analyze!")
    else:
        # -------------------
        # 🟢 8.1 Fake News Detection
        # -------------------
        st.subheader("🔍 Fake News Prediction")

        inputs = fake_news_tokenizer(
            news_text,
            return_tensors="pt",
            max_length=256,
            truncation=True,
            padding="max_length"
        ).to(device)

        with torch.no_grad():
            outputs = fake_news_model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=1).item()
            probabilities = F.softmax(logits, dim=1).detach().cpu().numpy()

        label_map = {0: "🟢 Real News", 1: "🔴 Fake News"}
        st.write(f"*Prediction:* {label_map[prediction]}")
        st.write(f"*Confidence Scores:* {probabilities}")

        # -------------------
        # 🟡 8.2 Generate LIME Explanation (Fix for Hindi Text & Show Probabilities Instead of Image)
        # -------------------
        st.subheader("📊 Word Contribution Probabilities for Fake News & Real News")

        def preprocess_text_for_lime(text):
            """Preprocess Hindi text to ensure LIME treats full words correctly."""
            text = text.lower()  # Convert to lowercase
            text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
            words = text.split()  # Split text into words
            return " ".join(words)  # Join words with spaces for LIME

        def predictor_lime(texts):
            """LIME predictor function - properly processes Hindi text."""
            preprocessed_texts = [preprocess_text_for_lime(text) for text in texts]  # Preprocess text

            inputs = fake_news_tokenizer(
                preprocessed_texts,
                return_tensors="pt",
                max_length=256,
                truncation=True,
                padding="max_length"
            ).to(device)

            with torch.no_grad():
                logits = fake_news_model(inputs["input_ids"], attention_mask=inputs["attention_mask"]).logits
                probabilities = F.softmax(logits, dim=1).detach().cpu().numpy()

            return probabilities

        # Preprocess input text before passing to LIME
        preprocessed_news_text = preprocess_text_for_lime(news_text)

        # Generate LIME explanation
        explanation = explainer.explain_instance(preprocessed_news_text, predictor_lime, num_features=5, num_samples=100)

        # Get word importance scores
        word_scores = explanation.as_list()

        # Convert word scores into a structured format
        probability_table = []
        for word, score in word_scores:
            probability_table.append({"Word": word, "Fake News Score": round(score, 4)})

        # Convert to Pandas DataFrame for better readability
        df_probabilities = pd.DataFrame(probability_table)

        # Display in Streamlit
        st.write("Below are the words contributing to Fake News detection along with their probabilities:")
        st.dataframe(df_probabilities)

        # -------------------
        # 🔵 8.3 Generate Summary
        # -------------------
        st.subheader("📖 Generated Summary")

        summary_inputs = summarization_tokenizer(
            news_text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding="max_length"
        ).to(device)

        summary_ids = summarization_model.generate(
            summary_inputs["input_ids"],
            max_length=70,
            min_length=30,
            num_beams=4,
            early_stopping=True
        )

        generated_summary = summarization_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        st.write(generated_summary)