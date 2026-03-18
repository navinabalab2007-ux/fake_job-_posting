import streamlit as st
import pickle
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# download stopwords (first time only)
nltk.download('stopwords')

# load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# stopwords
stop_words = set(stopwords.words('english'))

# text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# UI
st.title("🧠 Fake Job Detection System")

st.write("Enter a job description to check whether it is Fake or Real")

user_input = st.text_area("Enter Job Description")

# prediction
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text")
    else:
        clean = clean_text(user_input)
        vec = vectorizer.transform([clean])
        pred = model.predict(vec)[0]
        prob = model.predict_proba(vec)[0]

        if pred == 1:
            st.error(f"🚨 Fake Job (Confidence: {round(prob[1]*100,2)}%)")
        else:
            st.success(f"✅ Real Job (Confidence: {round(prob[0]*100,2)}%)")