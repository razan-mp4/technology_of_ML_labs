# Computer Practicum 3: Text Preprocessing, TF-IDF, and Topic Modeling with LDA

## Overview

This repository contains the implementation of a computer practicum focused on text preprocessing, feature extraction, and topic modeling. The goal is to analyze the text of "Alice's Adventures in Wonderland" using techniques like TF-IDF and Latent Dirichlet Allocation (LDA) to identify key words and topics.

The practicum involves preprocessing text data, converting it into numerical representations, and using machine learning methods for topic extraction and visualization.

---

## Objectives

1. **Preprocess Raw Text:**
   - Clean text by removing numbers, punctuation, and stopwords.
   - Normalize text to lowercase and tokenize it into individual words.

2. **Extract Features Using TF-IDF:**
   - Represent the text numerically using Term Frequency-Inverse Document Frequency (TF-IDF).
   - Identify the most significant words for each chapter.

3. **Perform Topic Modeling with LDA:**
   - Extract hidden topics from the text using Latent Dirichlet Allocation (LDA).
   - Visualize the most important words for each topic.

---

## Dataset

The dataset used in this practicum is the text of **"Alice's Adventures in Wonderland"**, obtained from the [Gutenberg Project](https://www.gutenberg.org/). The text is split into chapters for analysis.

---

## Steps

### 1. Text Preprocessing
- Load the full text of the book using the `requests` library.
- Clean and normalize the text:
  - Remove numbers, punctuation, and non-alphanumeric symbols.
  - Tokenize text into individual words.
  - Remove stopwords using NLTK's predefined English stopword list.

### 2. TF-IDF Analysis
- Convert the preprocessed text into a TF-IDF matrix, limiting the vocabulary size to 5000 words.
- Extract the top 20 words with the highest TF-IDF scores for each chapter.

### 3. Topic Modeling with LDA
- Perform Latent Dirichlet Allocation (LDA) to extract 5 topics from the text.
- Identify the top 10 words associated with each topic.
- Visualize the topics with horizontal bar plots.

---

## Results

The results include:
1. **Top Words per Chapter:**  
   - A list of the 20 most significant words for each chapter based on TF-IDF scores.
   
2. **LDA Topics:**  
   - Key topics extracted from the text with their associated words.

3. **Visualizations:**  
   - Horizontal bar plots displaying the top words for each LDA topic.

---

## Technologies Used

- **Programming Language:** Python
- **Libraries:**
  - `nltk` for text preprocessing (tokenization and stopwords)
  - `scikit-learn` for TF-IDF vectorization and LDA
  - `requests` for fetching the text dataset
  - `matplotlib` for data visualization
  - `re` for regular expression-based text cleaning

---

## Conclusion

This practicum demonstrates the application of NLP techniques to analyze a literary text. The TF-IDF analysis effectively highlights significant words for each chapter, while LDA provides insights into the overarching themes of the text. The combination of preprocessing, feature extraction, and topic modeling showcases the power of NLP for text analysis.

---
