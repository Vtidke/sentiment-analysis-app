import streamlit as st
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

# App Title
st.title("💬 Sentiment Analysis App (Amazon Reviews)")
st.write("Analyze customer reviews using Machine Learning")
st.caption("⚡ GenAI-inspired features (No external API used)")

# ✅ Cache function
@st.cache_resource
def load_model():

    folder_path = "data/csv_parts"
    df_list = []

    if not os.path.exists(folder_path):
        st.error("❌ Folder 'data/csv_parts' not found!")
        return None, None, None, 0, 0

    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            file_path = os.path.join(folder_path, file)
            try:
                df_temp = pd.read_csv(file_path)
                if not df_temp.empty:
                    df_list.append(df_temp)
            except:
                continue

    if len(df_list) == 0:
        st.error("❌ No valid CSV files found!")
        return None, None, None, 0, 0

    df = pd.concat(df_list, ignore_index=True)

    if 'Text' not in df.columns or 'Score' not in df.columns:
        st.error("❌ CSV must contain 'Text' and 'Score' columns!")
        return None, None, None, 0, 0

    if len(df) > 10000:
        df = df.sample(10000, random_state=42)

    df = df[['Text', 'Score']].dropna()
    df['Text'] = df['Text'].astype(str).str.lower()

    def get_sentiment(score):
        if score >= 4:
            return "positive"
        elif score == 3:
            return "neutral"
        else:
            return "negative"

    df['sentiment'] = df['Score'].apply(get_sentiment)

    X_train, X_test, y_train, y_test = train_test_split(
        df['Text'], df['sentiment'], test_size=0.2, random_state=42
    )

    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=5000,
        ngram_range=(1, 2)
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model_lr = LogisticRegression(max_iter=200)
    model_nb = MultinomialNB()

    model_lr.fit(X_train_vec, y_train)
    model_nb.fit(X_train_vec, y_train)

    acc_lr = accuracy_score(y_test, model_lr.predict(X_test_vec))
    acc_nb = accuracy_score(y_test, model_nb.predict(X_test_vec))

    return vectorizer, model_lr, model_nb, acc_lr, acc_nb


# Load model
vectorizer, model_lr, model_nb, acc_lr, acc_nb = load_model()

if vectorizer is None:
    st.stop()

# Accuracy
st.subheader("📊 Model Accuracy")
st.write(f"👉 Logistic Regression: {acc_lr:.2f}")
st.write(f"👉 Naive Bayes: {acc_nb:.2f}")

# Input
user_input = st.text_area("✍️ Enter your review here:")

# 🔍 Prediction
if st.button("🔍 Analyze Sentiment"):

    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text")
    else:
        user_input = user_input.lower()
        input_data = vectorizer.transform([user_input])

        # Predictions
        pred1 = model_lr.predict(input_data)[0]
        pred2 = model_nb.predict(input_data)[0]

        # Confidence-based neutral
        probs = model_lr.predict_proba(input_data)[0]
        confidence = max(probs)

        if confidence < 0.6:
            final_pred = "neutral"
        else:
            final_pred = pred1

        # Display
        st.subheader("📊 Prediction Results")
        st.write("👉 Logistic Regression:", pred1)
        st.write("👉 Naive Bayes:", pred2)
        st.write(f"🔥 Confidence: {confidence:.2f}")

        if final_pred == "positive":
            st.success("😊 Positive Review")
        elif final_pred == "negative":
            st.error("😠 Negative Review")
        else:
            st.info("😐 Neutral Review")

        # =========================
        # 🤖 GenAI-like Features
        # =========================

        st.subheader("🤖 AI Features")

        def explain_sentiment(text, prediction):
            positive_words = ["good", "great", "excellent", "amazing", "love"]
            negative_words = ["bad", "poor", "worst", "late", "slow"]

            pos = sum(word in text for word in positive_words)
            neg = sum(word in text for word in negative_words)

            if prediction == "positive":
                return f"This review is positive due to {pos} positive indicators."
            elif prediction == "negative":
                return f"This review is negative due to {neg} negative indicators."
            else:
                return "This review has mixed or neutral sentiment."

        def improve_review(text):
            text = text.strip().capitalize()
            return f"{text}. The experience can be described more clearly with additional details."

        def summarize_review(text):
            words = text.split()
            return " ".join(words[:10]) + "..." if len(words) > 10 else text

        def generate_reply(prediction):
            if prediction == "positive":
                return "Thank you for your positive feedback! We're glad you had a great experience."
            elif prediction == "negative":
                return "We’re sorry for your experience. We will work on improving our service."
            else:
                return "Thank you for your feedback. We appreciate your suggestions."

        if st.button("🧠 Explain Sentiment"):
            st.write(explain_sentiment(user_input, final_pred))

        if st.button("✨ Improve Review"):
            st.write(improve_review(user_input))

        if st.button("📄 Summarize Review"):
            st.write(summarize_review(user_input))

        if st.button("💌 Generate Reply"):
            st.write(generate_reply(final_pred))
