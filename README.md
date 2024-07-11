# Lexa
Lexa is a Flask-based web application designed to perform topic modeling on text data extracted from PDF files or input directly via a text box. Using advanced natural language processing (NLP) techniques, Lexa leverages Latent Dirichlet Allocation (LDA) to uncover hidden topics within the provided text. This tool is useful for researchers, data scientists, and anyone interested in understanding the underlying themes in their textual data.

# Features
* Upload PDFs or Enter Text Directly: Users can upload PDF files or enter text directly into a text box for analysis.
* Text Extraction: Automatically extracts text from PDF files.
* Text Preprocessing: Includes text normalization, punctuation removal, and stopwords removal.
* Topic Modeling: Applies LDA to identify and display key topics within the text.
* Interactive Results: Displays topics with the most significant words and their weights.

# Prerequisites
Ensure you have the following installed:

Python 3.6+
Flask
PyMuPDF
NLTK
scikit-learn
