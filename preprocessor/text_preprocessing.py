import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def preprocess(description):
    tokens = word_tokenize(description.lower())
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    return ' '.join(filtered_tokens)
