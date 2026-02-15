# 📰 Fake News Detection with Evidence

**A Hybrid AI Approach for News Credibility Assessment**

This project is a **Fake News Detection Web Application** built with Python and Streamlit. It goes beyond simple classification by combining **Machine Learning (Logistic Regression)** with **Fact-Based Evidence Retrieval (Wikipedia)** to provide explainable and trustworthy results.

---

## 🚀 Features

*   **Hybrid Analysis**: Combines linguistic pattern recognition with factual cross-referencing.
*   **ML Classification**: Predicts if text is "Fake" or "Real" using a TF-IDF + Logistic Regression pipeline.
*   **Evidence Retrieval**: Automatically searches Wikipedia to find context related to the claim.
*   **Explainable Verdict**: Provides a human-readable interpretation (e.g., "Credible", "Uncertain", "Potentially Misleading") based on model confidence and evidence availability.
*   **Confidence Scoring**: Displays the model's certainty percentage.

## 🛠️ Tech Stack

*   **Python**: Core programming language.
*   **Streamlit**: For the interactive web interface.
*   **Scikit-Learn**: For the Machine Learning pipeline (TF-IDF, Logistic Regression).
*   **NLTK**: For Natural Language Processing (Tokenization, Stopwords, Lemmatization).
*   **Wikipedia-API**: For real-time evidence fetching.

---

## 📦 Installation

Clone the repository and install the dependencies.

```bash
git clone <repository-url>
cd fake-news-detection-app
pip install -r requirements.txt
```

## ⚙️ Setup

The project includes a training script to generate the model artifacts. Run this once before starting the app:

```bash
python train_model.py
```
*This will create the `model/` directory containing `model.pkl` and `tfidf.pkl`.*

## ▶️ Usage

1.  Run the application:
    ```bash
    streamlit run app.py
    ```

2.  Open your browser (usually at `http://localhost:8501`).
3.  Paste a news article or claim into the text area.
4.  Click **"Analyze Credibility"**.
5.  View the **Prediction**, **Confidence Score**, and **Final Verdict**.

---

## 🧩 Project Architecture

```mermaid
graph TD
    A[User Input] --> B(Preprocessing);
    B --> C{Analysis};
    C -->|Linguistic Features| D[ML Model (Logistic Regression)];
    C -->|Keywords| E[Evidence Retrieval (Wikipedia)];
    D --> F[Prediction & Confidence];
    E --> G[Context Summary];
    F --> H[Verdict Logic];
    G --> H;
    H --> I[Final Output Display];
```

## 🔮 Future Improvements

*   **Knowledge Graph Integration**: To map entities and relationships for deeper fact-checking.
*   **Transformer Models**: upgrading to BERT/RoBERTa for better semantic understanding.
*   **Real-time News API**: Checking against live news sources instead of static Wikipedia.

---

## 📄 License

This project is open-source and available for educational purposes.
