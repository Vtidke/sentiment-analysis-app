💬 Sentiment Analysis App (Amazon Reviews)

A Machine Learning + Streamlit web application that analyzes customer reviews and classifies them into Positive 😊, Negative 😠, or Neutral 😐 sentiments.

This project is enhanced with confidence-based prediction and GenAI-inspired features, making it more practical and closer to real-world AI systems.

🚀 Live Demo

👉 https://sentiment-analysis-app-ien2borjxkjws9jkrewz7p.streamlit.app/

📌 Features
✅ Sentiment prediction using Logistic Regression and Naive Bayes
✅ Displays model accuracy
✅ NLP using TF-IDF Vectorization
✅ Confidence-based neutral classification
✅ Interactive UI built with Streamlit
✅ Handles real-world ambiguous reviews
🤖 GenAI-Inspired Features (No API Used)
🧠 Explain sentiment reasoning
✨ Improve user-written reviews
📄 Summarize long reviews
💌 Generate automated replies
🧠 How It Works
Loads Amazon review dataset from CSV files
Cleans and preprocesses text data
Converts text into numerical format using TF-IDF
Trains two models:
Logistic Regression
Multinomial Naive Bayes
Predicts sentiment for user input
🔥 Confidence-Based Prediction Logic

The model calculates probability for each class:

If confidence ≥ 0.6 → Show predicted sentiment
If confidence < 0.6 → Classify as Neutral

👉 This helps handle uncertain or mixed reviews more effectively.

🛠 Tech Stack
Python
Streamlit
Scikit-learn
Pandas
NLP (TF-IDF)
📂 Project Structure
sentiment-analysis-app/
│
├── app.py
├── requirements.txt
├── README.md
└── data/
    └── csv_parts/
        ├── file1.csv
        ├── file2.csv
▶️ How to Run Locally
git clone https://github.com/your-username/sentiment-analysis-app.git
cd sentiment-analysis-app
pip install -r requirements.txt
streamlit run app.py
📊 Sample Test Reviews

Positive:
"This product is amazing, I absolutely love it!"

Negative:
"Worst product ever, very disappointed."

Neutral:
"It is okay, not too good not too bad."

🎯 Future Improvements
Voice input support 🎤
Multi-language sentiment analysis 🌐
Data visualization dashboard 📊
Integration with real GenAI APIs 🤖
👩‍💻 Author

Vaishnavi Tidke

Ex-TCS | Ex-Amazon
Aspiring Data Scientist
⭐ Support

If you like this project, consider giving it a ⭐ on GitHub!
