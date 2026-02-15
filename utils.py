import re
import string
import nltk
import wikipedia
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure NLTK resources are downloaded
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('punkt_tab')

def preprocess_text(text):
    """
    Preprocess the input text for the machine learning model.
    
    Steps:
    1. Lowercase text.
    2. Remove punctuation and non-alphabetic characters.
    3. Tokenize text.
    4. Remove stopwords (English).
    5. Lemmatize tokens.
    
    Args:
        text (str): Raw input text string.
        
    Returns:
        str: Cleaned and preprocessed text string.
    """
    if not isinstance(text, str):
        return ""
    
    # Helper for cleaning
    def clean(x):
        # Lowercasing
        x = x.lower()
        # Removing punctuation and numbers (keeping only alphabetic)
        x = re.sub(r'[^a-z\s]', '', x)
        return x

    text = clean(text)
    
    # Tokenization
    tokens = nltk.word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return " ".join(tokens)

def get_evidence(query):
    """
    Retrieve a short summary from Wikipedia for the given query.
    
    Args:
        query (str): The search term to look up on Wikipedia.
        
    Returns:
        str or None: A 2-sentence summary if found, else None.
    """
    try:
        # Search for the page most relevant to the query
        search_results = wikipedia.search(query)
        if not search_results:
            return None
        
        # Get the summary of the first result
        summary = wikipedia.summary(search_results[0], sentences=2)
        return summary
    except wikipedia.exceptions.DisambiguationError as e:
        # If ambiguous, try the first option
        try:
            summary = wikipedia.summary(e.options[0], sentences=2)
            return summary
        except:
            return None
    except wikipedia.exceptions.PageError:
        return None
    except Exception as e:
        return None
