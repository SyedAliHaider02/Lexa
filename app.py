from flask import Flask, request, render_template
import fitz  # PyMuPDF
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'  # Directory to save uploaded files

@app.route('/')
def upload_file():
    return render_template('upload.html')

@app.route('/uploader', methods=['POST'])
def uploader_file():
    if request.method == 'POST':
        input_type = request.form['input_type']
        if input_type == 'pdf':
            if 'file' not in request.files:
                return 'No file part'
            file = request.files['file']
            if file.filename == '':
                return 'No selected file'
            if file:
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(filepath)
                text = extract_text_from_pdf(filepath)
        elif input_type == 'text':
            text = request.form['text_input']
        
        # Text preprocessing
        processed_text = preprocess_text(text)
        
        # Apply LDA
        if processed_text:
            topics = apply_lda_sklearn(processed_text)
            return render_template('result.html', topics=topics)
        else:
            return 'Error: No text to process'

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def preprocess_text(text):
    text = text.lower()
    text = removeTags(text)
    text = removePunct(text)
    text = removeStopwords(text)
    return text

def removeTags(text):
    tags = ['\n\n', '\n', '\'']
    for tag in tags:
        if tag in text:
            text = text.replace(tag, '')
    return text

import string

def removePunct(text):
    for char in text:
        if char in string.punctuation:
            text = text.replace(char, '')
    return text

def removeStopwords(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words = [word for word in tokens if word.lower() not in stop_words]
    return ' '.join(words)

def apply_lda_sklearn(text):

    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()
    lda = LatentDirichletAllocation(n_components=2, max_iter=10, learning_method='online', random_state=42)
    lda.fit(X)
    topics = get_top_words_with_weights(lda, feature_names, 4)
    return topics

def get_top_words_with_weights(model, feature_names, n_top_words):
    topics = {}
    for topic_idx, topic in enumerate(model.components_):
        top_words_idx = topic.argsort()[:-n_top_words - 1:-1]
        topic_words = [(feature_names[i], topic[i]) for i in top_words_idx]
        topics[f'Topic {topic_idx}'] = topic_words
    return topics

if __name__ == '__main__':
    app.run(debug=True)
