import pandas as pd
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from utils import preprocess_text

def train_and_save_model():
    """
    Trains a Logistic Regression model on a dummy dataset and saves the 
    Model and TF-IDF vectorizer to the 'model/' directory.
    
    In a production environment, this would load a real dataset.
    """
    
    # 1. Create a dummy dataset
    data = {
        'text': [
            "The earth is flat and the government is lying to us.",
            "Scientists confirm that the earth is round based on satellite imagery.",
            "Aliens have landed in New York City and are meeting with the President.",
            "NASA launches a new rover to explore the surface of Mars.",
            "Drinking bleach cures all known diseases instantly.",
            "The CDC recommends vaccination to prevent the spread of the virus.",
            "A secret cabal of elites controls the world's economy.",
            "The stock market closed higher today driven by tech gains.",
            "Celebrity X was replaced by a clone in 2010.",
            "The movie won three Academy Awards mainly for its screenplay.",
            "Water turns into wine if you stare at it long enough.",
            "H2O is the chemical formula for water."
        ],
        'label': [
            'Fake', 'Real', 'Fake', 'Real', 'Fake', 'Real', 
            'Fake', 'Real', 'Fake', 'Real', 'Fake', 'Real'
        ]
    }

    df = pd.DataFrame(data)

    # 2. Preprocess the text
    print("Preprocessing text...")
    df['clean_text'] = df['text'].apply(preprocess_text)

    # 3. Vectorize
    print("Vectorizing...")
    tfidf = TfidfVectorizer(max_features=5000)
    X = tfidf.fit_transform(df['clean_text'])
    y = df['label']

    # 4. Train Model
    print("Training model...")
    model = LogisticRegression()
    model.fit(X, y)

    # 5. Save Artifacts
    print("Saving artifacts...")
    os.makedirs('model', exist_ok=True)
    
    with open('model/tfidf.pkl', 'wb') as f:
        pickle.dump(tfidf, f)
    with open('model/model.pkl', 'wb') as f:
        pickle.dump(model, f)

    print("Done! Artifacts saved in 'model/' directory.")

if __name__ == "__main__":
    train_and_save_model()
