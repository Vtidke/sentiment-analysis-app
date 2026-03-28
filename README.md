# 💬 Sentiment Analysis App (Amazon Reviews)

A Machine Learning web application built using **Streamlit** that analyzes customer reviews and predicts sentiment as **Positive, Negative, or Neutral**.

---

## 🚀 Live Demo

🔗 https://sentiment-analysis-app-ien2borjxkjws9jkrewz7p.streamlit.app/

---

## 📌 Features

* Analyze real-time user input reviews
* Predict sentiment using:

  * Logistic Regression
  * Naive Bayes
* Displays model accuracy
* Clean and interactive UI using Streamlit
* Handles multiple CSV files as dataset input
* Robust error handling for missing or invalid data

---

## 🧠 Machine Learning Workflow

1. Data Collection (Amazon Reviews Dataset)
2. Data Preprocessing:

   * Lowercasing text
   * Removing null values
3. Sentiment Mapping:

   * Score ≥ 4 → Positive
   * Score = 3 → Neutral
   * Score ≤ 2 → Negative
4. Feature Engineering:

   * TF-IDF Vectorization (unigrams + bigrams)
5. Model Training:

   * Logistic Regression
   * Multinomial Naive Bayes
6. Evaluation:

   * Accuracy Score

---

## 🛠️ Tech Stack

* Python 🐍
* Streamlit 🌐
* Pandas 📊
* Scikit-learn 🤖

---

## 📂 Project Structure

```
sentiment-analysis-app/
│
├── app.py
├── requirements.txt
├── data/
│   └── csv_parts/
│       ├── part1.csv
│       ├── part2.csv
│       └── part3.csv
```

---

## ⚠️ Challenges Faced

* Handling large CSV files on Streamlit Cloud
* Fixing file path issues during deployment
* Managing empty/corrupt CSV files
* Optimizing model training for limited resources

---

## 💡 Learnings

* Real-world ML deployment challenges
* Importance of data preprocessing
* Handling imbalanced datasets (neutral class issue)
* Difference between training vs inference in production

---

## 🔮 Future Improvements

* Use pre-trained models instead of training in app
* Improve neutral sentiment detection
* Add confidence score for predictions
* Deploy using optimized ML pipeline
* Enhance UI/UX

---

## 🙋‍♀️ Author

**Vaishnavi Tidke**

* Ex-TCS | Ex-Amazon
* Aspiring Data Scientist

---

## ⭐ If you like this project

Give it a ⭐ on GitHub and share your feedback!

---
