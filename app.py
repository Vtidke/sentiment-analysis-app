import streamlit as st
import pandas as pd
import glob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

# App Title
st.title("💬 Sentiment Analysis App (Amazon Reviews)")
st.write("Analyze customer reviews using Machine Learning")

# ✅ Cache function
@st.cache_resource
def load_model():
    
    # Load split CSV files
    files = glob.glob("data/csv_parts/*.csv")

    df_list = []
    for file in files:
        df_list.append(pd.read_csv(file))

    df = pd.concat(df_list, ignore_index=True)

    # Take sample for speed
    df = df.sample(10000, random_state=42)

    # Keep only required columns
    df = df[['Text', 'Score']]
    df = df.dropna()

    # Lowercase
    df['Text'] = df['Text'].str.lower()

    # Sentiment mapping
    def get_sentiment(score):
        if score >= 4:
            return "positive"
        elif score == 3:
            return "neutral"
        else:
            return "negative"

    df['sentiment'] = df['Score'].apply(get_sentiment)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        df['Text'], df['sentiment'], test_size=0.2, random_state=42
    )

    # Vectorizer
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1,2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Models
    model_lr = LogisticRegression(max_iter=200)
    model_nb = MultinomialNB()

    # Train
    model_lr.fit(X_train_vec, y_train)
    model_nb.fit(X_train_vec, y_train)

    # Accuracy
    acc_lr = accuracy_score(y_test, model_lr.predict(X_test_vec))
    acc_nb = accuracy_score(y_test, model_nb.predict(X_test_vec))

    return vectorizer, model_lr, model_nb, acc_lr, acc_nb


# Load model
vectorizer, model_lr, model_nb, acc_lr, acc_nb = load_model()

# Show accuracy
st.subheader("📊 Model Accuracy")
st.write(f"👉 Logistic Regression: {acc_lr:.2f}")
st.write(f"👉 Naive Bayes: {acc_nb:.2f}")

# Input
user_input = st.text_area("✍️ Enter your review here:")

# Prediction
if st.button("🔍 Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text")
    else:
        user_input = user_input.lower()
        input_data = vectorizer.transform([user_input])

        pred1 = model_lr.predict(input_data)[0]
        pred2 = model_nb.predict(input_data)[0]

        st.subheader("📊 Prediction Results")
        st.write("👉 Logistic Regression:", pred1)
        st.write("👉 Naive Bayes:", pred2)

        if pred1 == "positive":
            st.success("😊 Positive Review")
        elif pred1 == "negative":
            st.error("😠 Negative Review")
        else:
            st.info("😐 Neutral Review")
