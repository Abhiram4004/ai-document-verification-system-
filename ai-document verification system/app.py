import streamlit as st
import pickle
from ocr.ocr_utils import extract_text_from_image
from preprocessing.text_preprocessing import clean_text

# Load model & encoder
model = pickle.load(open("models/model.pkl", "rb"))
encoder = pickle.load(open("models/label_encoder.pkl", "rb"))

st.set_page_config(page_title="AI Document Verification", layout="centered")

st.title("📄 AI-Powered Document Verification System")

uploaded_file = st.file_uploader("Upload Document Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Document", use_column_width=True)

    extracted_text = extract_text_from_image(uploaded_file)
    processed_text = clean_text(extracted_text)

    prediction = model.predict([processed_text])[0]
    probability = model.predict_proba([processed_text]).max() * 100

    result = encoder.inverse_transform([prediction])[0]

    st.subheader("Prediction Result")
    st.success(f"Document Status: {result}")
    st.info(f"Confidence Score: {probability:.2f}%")
