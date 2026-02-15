import streamlit as st
import pickle
import os
from utils import preprocess_text, get_evidence

# Configuration & Constants
MODEL_PATH = 'model/model.pkl'
TFIDF_PATH = 'model/tfidf.pkl'

# Helper Functions
@st.cache_resource
def load_models():
    """
    Load the trained Logistic Regression model and TF-IDF vectorizer.
    
    Returns:
        tuple: (model, tfidf) or (None, None) if files not found.
    """
    try:
        if not os.path.exists(MODEL_PATH) or not os.path.exists(TFIDF_PATH):
            st.error("Model files not found. Please run `python train_model.py` first.")
            return None, None
            
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(TFIDF_PATH, 'rb') as f:
            tfidf = pickle.load(f)
        return model, tfidf
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

def get_verdict(prediction_label, confidence_pct, evidence_summary):
    """
    Generate the final verdict and color code based on prediction and evidence.
    
    Args:
        prediction_label (str): 'Fake' or 'Real'.
        confidence_pct (float): Model confidence percentage.
        evidence_summary (str): Wikipedia summary or None.
        
    Returns:
        tuple: (verdict_message, verdict_color)
    """
    verdict_message = ""
    verdict_color = "info" 
    
    # evidence exists
    if evidence_summary:
        if confidence_pct < 60:
            verdict_message = "Uncertain: The model is unsure, but external evidence suggests this topic is well-documented. Please verify manually."
            verdict_color = "warning"
        elif prediction_label == "Fake":
            verdict_message = "Likely Factual but Stylistically Unusual: The text is flagged as Fake by the model, but we found relevant Wikipedia content. It might be true information written in a non-standard way."
            verdict_color = "warning"
        elif prediction_label == "Real":
            verdict_message = "Credible: The model predicts this is Real news, and we found supporting references in Wikipedia."
            verdict_color = "success"
    
    else:
        if prediction_label == "Fake":
            verdict_message = "Potentially Misleading: Use caution. The model predicts this is Fake, and we could not find reliable verification on Wikipedia."
            verdict_color = "error"
        elif prediction_label == "Real":
            verdict_message = "Credible but Not Verified: The model predicts Real, but we could not cross-reference it with Wikipedia. Proceed with mild caution."
            verdict_color = "info"
            
    return verdict_message, verdict_color

def main():
    st.set_page_config(page_title="Fake News Detection", page_icon="📰", layout="wide")

    st.title("📰 Fake News Detection with Evidence")
    st.markdown("""
    **Hybrid AI Analysis System**  
    This tool combines machine learning linguistic analysis with fact-based retrieval from Wikipedia to assess news credibility.
    """)

    # Load Models
    model, tfidf = load_models()

    if model and tfidf:

        st.subheader("Analyze Text")
        news_text = st.text_area("Enter a news article or claim:", height=200, placeholder="Paste text here to detect fake news patterns...")

        if st.button("Analyze Credibility", type="primary"):
            if not news_text.strip():
                st.warning("Please enter some text to analyze.")
            else:
                with st.spinner("Processing..."):
                    # 1. Preprocessing & Prediction
                    clean_text = preprocess_text(news_text)
                    text_vector = tfidf.transform([clean_text])
                    
                    prediction_label = model.predict(text_vector)[0]
                    prediction_prob = model.predict_proba(text_vector).max()
                    confidence_pct = round(prediction_prob * 100, 2)
                    
                    # 2. Evidence Retrieval
                    # Use the first 20 words for the search query for better relevance
                    search_query = " ".join(news_text.split()[:20])
                    evidence_summary = get_evidence(search_query)
                    
                    # 3. Verdict
                    verdict_msg, verdict_color = get_verdict(prediction_label, confidence_pct, evidence_summary)
                    
                    # 4. Display Results
                    st.divider()
                    
                    col1, col2, col3 = st.columns([1, 2, 1])
                    
                    with col1:
                        st.subheader("Assessment")
                        st.metric("Prediction", prediction_label)
                        st.metric("Confidence", f"{confidence_pct}%")
                        st.progress(confidence_pct / 100)
                    
                    with col2:
                        st.subheader("Verdict")
                        if verdict_color == "success":
                            st.success(verdict_msg)
                        elif verdict_color == "warning":
                            st.warning(verdict_msg)
                        elif verdict_color == "error":
                            st.error(verdict_msg)
                        else:
                            st.info(verdict_msg)
                            
                        st.subheader("Evidence Context")
                        if evidence_summary:
                            st.info(f"**Wikipedia Summary:** {evidence_summary}")
                        else:
                            st.markdown("*No direct Wikipedia match found.*")

    # Footer / Sidebar
    with st.sidebar:
        st.header("About the Project")
        st.write("This application demonstrates a hybrid approach to fake news detection.")
        st.markdown("""
        **Tech Stack:**
        - Python
        - Streamlit
        - Scikit-Learn (Logistic Regression)
        - NLTK (Preprocessing)
        - Wikipedia API
        """)

if __name__ == "__main__":
    main()
