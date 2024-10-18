from flask import Flask, render_template, request
import spacy
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

app = Flask(__name__)

nlp = spacy.load("en_core_web_sm")

# Load the dataset
dataset = pd.read_csv('answers.csv')

def check_grammar(answer):
    blob = TextBlob(answer)
    return str(blob.correct())

def check_coherence(answer):
    sentences = answer.split('.')
    return len(sentences) > 1  # Basic check for more than one sentence

def check_plagiarism(student_answer, correct_answer):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([student_answer, correct_answer])
    plagiarism_score = (vectors * vectors.T).A[0, 1]
    return plagiarism_score

@app.route('/')
def home():
    questions = dataset[['question_id', 'question']].to_dict(orient='records')
    return render_template('index.html', questions=questions)

@app.route('/evaluate', methods=['POST'])
def evaluate():
    student_answer = request.form['answer']
    question_id = int(request.form['question_id'])

    # Get the correct answer and question from the dataset based on the question ID
    correct_answer = dataset[dataset['question_id'] == question_id]['correct_answer'].values[0]
    question_text = dataset[dataset['question_id'] == question_id]['question'].values[0]

    # Evaluate grammar and coherence
    corrected_answer = check_grammar(student_answer)
    coherence = check_coherence(corrected_answer)
    
    # Check plagiarism score by comparing with the correct answer
    plagiarism_score = check_plagiarism(corrected_answer, correct_answer)

    result = {
        'corrected_answer': corrected_answer,
        'coherence': coherence,
        'plagiarism_score': plagiarism_score,
        'correct_answer': correct_answer,
        'question_text': question_text
    }
    
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)