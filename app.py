import nltk
import spacy
from flask import Flask, render_template, request
import numpy as np

# Download required NLTK data files
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Initialize NLP tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
nlp = spacy.load('en_core_web_sm')

# Preprocessing function
def preprocess_text(text):
    words = word_tokenize(text.lower())
    words = [w for w in words if w.isalpha() and w not in stop_words]
    words = [lemmatizer.lemmatize(w) for w in words]
    return ' '.join(words)

# Feature extraction function
def extract_features(text):
    doc = nlp(text)
    num_words = len([token for token in doc if token.is_alpha])
    num_sentences = len(list(doc.sents))
    grammar_errors = sum([1 for token in doc if token.tag_ in ('NN', 'VB') and token.dep_ == 'advmod'])
    return [num_words, num_sentences, grammar_errors]

# Generate feedback
def generate_feedback(text):
    doc = nlp(text)
    feedback = []
    # Example feedback
    if len(list(doc.sents)) < 3:
        feedback.append("Your essay could benefit from more sentences.")
    if any(token.tag_ == 'NN' and token.dep_ == 'advmod' for token in doc):
        feedback.append("Consider improving the grammatical structure of your essay.")
    if len([token for token in doc if token.is_alpha]) < 100:
        feedback.append("Try to elaborate more on your ideas to increase the word count.")
    return feedback

# Example pre-trained model (Linear Regression)
class PreTrainedModel:
    def __init__(self):
        self.coefficients = [0.1, 0.5, -0.2]
        self.intercept = 2.0
    
    def predict(self, features):
        return np.dot(features, self.coefficients) + self.intercept

# Instantiate the model
model = PreTrainedModel()

# Create Flask app
app = Flask(__name__)

# Feature extraction function with detailed feedback
def extract_features_with_feedback(text):
    doc = nlp(text)
    num_words = len([token for token in doc if token.is_alpha])
    num_sentences = len(list(doc.sents))
    grammar_errors = sum([1 for token in doc if token.tag_ in ('NN', 'VB') and token.dep_ == 'advmod'])
    
    feedback = []
    if num_words < 50:
        feedback.append("Your essay is too short. Try to elaborate more on your ideas.")
    if num_sentences < 3:
        feedback.append("Your essay needs more sentences. Aim for more comprehensive paragraphs.")
    if grammar_errors > 5:
        feedback.append("Your essay has several grammar errors. Consider revising your grammar.")
    
    return [num_words, num_sentences, grammar_errors], feedback

# Update the index route
@app.route('/', methods=['GET', 'POST'])
def index():
    score = None
    feedback = []
    if request.method == 'POST':
        user_essay = request.form['essay']
        processed_essay = preprocess_text(user_essay)
        features, feedback = extract_features_with_feedback(processed_essay)
        score = model.predict(features)
    return render_template('index.html', score=score, feedback=feedback)


if __name__ == '__main__':
    app.run(debug=True)
