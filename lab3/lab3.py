# Import necessary libraries for text processing and analysis
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import requests
import re
import matplotlib.pyplot as plt
import numpy as np

# Ensure required NLTK resources (tokenizers and stopwords) are downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Load the text from the Gutenberg Project (Alice's Adventures in Wonderland)
url = "http://www.gutenberg.org/files/11/11-0.txt"
response = requests.get(url)
text = response.text  # Full raw text of the book

# Preprocessing function to clean and normalize text
def preprocess_text(text):
    # Remove numbers from the text
    text = re.sub(r'\d+', '', text)
    # Remove punctuation and non-alphanumeric symbols
    text = re.sub(r'[^\w\s]', '', text)
    # Replace multiple whitespaces with a single space and strip leading/trailing spaces
    text = re.sub(r'\s+', ' ', text).strip()
    # Convert the text to lowercase for consistency
    text = text.lower()
    # Tokenize the text into individual words
    tokens = word_tokenize(text)
    # Load the English stopword list
    stop_words = set(stopwords.words('english'))
    # Remove stopwords from the tokenized words
    tokens = [word for word in tokens if word not in stop_words]
    # Reconstruct cleaned text from filtered tokens
    return ' '.join(tokens)

# Apply preprocessing to the entire book text
clean_text = preprocess_text(text)

# Split the book into chapters using the "CHAPTER" delimiter
# Skip the preamble before the first chapter
chapters = text.split("CHAPTER")[1:]
# Preprocess each chapter individually
chapter_texts = [preprocess_text(chapter) for chapter in chapters]

# TF-IDF analysis: Convert chapters into a TF-IDF matrix
vectorizer = TfidfVectorizer(max_features=5000)  # Limit vocabulary to the 5000 most important words
tfidf_matrix = vectorizer.fit_transform(chapter_texts)

# Extract feature names (vocabulary) from the TF-IDF vectorizer
feature_names = vectorizer.get_feature_names_out()

# Extract the top 20 words with the highest TF-IDF scores per chapter
top_words_per_chapter = {}
for i, chapter in enumerate(tfidf_matrix):
    # Sort words by their TF-IDF scores in descending order and select the top 20
    sorted_indices = chapter.toarray().argsort()[0, -20:]
    top_words_per_chapter[f"Chapter {i+1}"] = [feature_names[idx] for idx in sorted_indices]

# Print the top 20 words for each chapter
print("Top 20 words per chapter (TF-IDF):")
for chapter, words in top_words_per_chapter.items():
    print(f"{chapter}: {', '.join(words)}")

# Latent Dirichlet Allocation (LDA) for topic modeling
# Set the number of topics to extract (n_components)
lda = LatentDirichletAllocation(n_components=5, random_state=42)
# Fit the LDA model on the TF-IDF matrix and transform the data
lda_matrix = lda.fit_transform(tfidf_matrix)

# Print the top 10 words associated with each topic
print("\nTopics extracted by LDA:")
for idx, topic in enumerate(lda.components_):
    # Sort the words by their importance in the topic and select the top 10
    print(f"Topic {idx+1}: {[feature_names[i] for i in topic.argsort()[-10:]]}")
