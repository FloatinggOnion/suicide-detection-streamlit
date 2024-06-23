import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from utils import show_code, send_assessment_emails

# Define questions and their multiple-choice options
QUESTIONS = {
    # 'q1': {
    #     'text': "What level are you?",
    #     'options': ["1: 100", "2: 200", "3: 300", "4: 400", "5: 500"]
    # },
    'q2': {
        'text': "What is your gender?",
        'options': ["1: Male", "2: Female"]
    },
    'q3': {
        'text': "Have you ever felt so bad that you wished you were dead?",
        'options': ["1: No", "2: Yes, but I don't have these thoughts anymore", "3: Yes, I still have these thoughts"]
    },
    'q4': {
        'text': "Have you ever had thoughts of killing yourself but you wouldn't do it?",
        'options': ["1: No", "2: Yes, but I don't have these thoughts anymore", "3: Yes, I still have these thoughts"]
    },
    'q5': {
        'text': "Have you ever thought of killing yourself, and you had a plan for how you would do it?",
        'options': ["1: No", "2: Yes, but I don't have these thoughts anymore", "3: Yes, I still have these thoughts"]
    },
    'q6': {
        'text': "Have you ever made an attempt to kill yourself?",
        'options': ["1: No", "2: Yes, but it didn't work", "3: Yes, and it did work"]
    },
    'q7': {
        'text': "Have you ever done anything to prepare to kill yourself (such as collecting pills, obtaining a weapon, writing a suicide note)?",
        'options': ["1: No", "2: Yes, but I didn't go through with it", "3: Yes, and I went through with it"]
    },
    'q8': {
        'text': "How often have you been bothered by thoughts of suicide in the past week/month?",
        'options': ["1: Not at all", "2: Several days", "3: More than half the days", "4: Nearly every day"]
    },
    'q9': {
        'text': "Have you ever deliberately hurt yourself without intending to kill yourself (e.g., cutting, burning)?",
        'options': ["1: No", "2: Yes, but I don't do it anymore", "3: Yes, I still do it"]
    },
    'q10': {
        'text': "Wish to live",
        'options': ["1: I don't have any thoughts of killing myself", "2: I have a weak desire to kill myself", "3: I have a moderate desire to kill myself", "4: I have a strong desire to kill myself", "5: I have a very strong desire to kill myself."]
    },
    'q11': {
        'text': "Wish to die",
        'options': ["1: I have a strong desire to live", "2: I have a weak desire to live", "3: I don't care whether I live or die", "4: I would like to be dead", "5: I would like to be dead right now"]
    },
    'q12': {
        'text': "Suicide ideation",
        'options': ["1: I don't have thoughts of killing myself", "2: I have occasional thoughts of killing myself", "3: I have frequent thoughts of killing myself", "4: I have consistent thoughts of killing myself"]
    },
    'q13': {
        'text': "Intensity of Suicidal Thoughts",
        'options': ["1: I don't think about killing myself", "2: I have a weak desire to kill myself", "3: I have a moderate desire to kill myself", "4: I have a strong desire to kill myself", "5: I have a very strong desire to kill myself"]
    },
    'q14': {
        'text': "Specific Plans for Suicide",
        'options': ["1: I don't have a plan to kill myself", "2: I  have a plan to kill myself, but I won’t carry it out", "3: I have a plan to kill myself, and I might carry it out", "4: I have a plan to kill myself, and I will carry it out if I get the chance"]
    },
    'q15': {
        'text': "Suicide intent",
        'options': ["1: I don't intend to kill myself", "2: I might kill myself if the opportunity arises", "3: I intend to kill myself if I can figure out how", "4: I intend to kill myself"]
    },
    'q16': {
        'text': "Have you ever thought about or attempted to kill yourself?",
        'options': ["1: Not at all", "2: A few times", "3: More than half the time", "4: Nearly everyday"]
    },
    'q17': {
        'text': "Have you told someone that you were going to commit suicide, or that you might, do it?",
        'options': ["1: No", "2: Yes, but I didn't mean it", "3: Yes, and I meant it at the time", "4: Yes, but I didn't tell them directly"]
    },
    'q18': {
        'text': "Have you ever made a plan for how you would kill yourself?",
        'options': ["1: No", "2: Yes, but I didn't go through with it", "3: Yes, and I did go through with it", "4: Yes, but I don't remember what it was"]
    },
    'q19': {
        'text': "Have you ever tried to kill yourself?",
        'options': ["1: No", "2: Yes, but I didn't really want to die", "3: Yes, and I really wanted to die", "4: Yes, but I don't remember it very well"]
    },
    'q20': {
        'text': "Have you ever deliberately harmed yourself in any way but not with the intention of killing yourself?",
        'options': ["1: No", "2: Yes, once", "3: Yes, a few times", "4: Yes, many times"]
    },
    
}

# Load and preprocess data
@st.cache_data
def load_data():
    data = pd.read_csv('suiciderisk.csv')

    # Drop the first column
    data = data.iloc[:, 1:]
    
    numeric_data = data.select_dtypes(include='number')
    data['response'] = numeric_data.mean(axis=1).round().astype(int)
    
    def map_risk(response):
        if response in [1, 2]:
            return 'High_risk'  # Changed from Low_risk
        elif response == 3:
            return 'Medium_risk'
        elif response in [4, 5]:
            return 'Low_risk'  # Changed from High_risk
    
    data['risk_category'] = data['response'].apply(map_risk)
    
    le = LabelEncoder()
    for column in data.select_dtypes(include=['object']).columns:
        data[column] = le.fit_transform(data[column])
    
    return data

# Train models
@st.cache_resource
def train_models(X, y):
    models = {
        'Naive Bayes': GaussianNB(),
        'SVM': SVC(probability=True),
        'Random Forest': RandomForestClassifier(),
        'Logistic Regression': LogisticRegression(max_iter=500)
    }
    
    results = {}
    for name, model in models.items():
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        results[name] = {
            'accuracy': np.mean(cv_scores),
            'precision': np.mean(cross_val_score(model, X, y, cv=5, scoring='precision_weighted')),
            'recall': np.mean(cross_val_score(model, X, y, cv=5, scoring='recall_weighted')),
            'f1': np.mean(cross_val_score(model, X, y, cv=5, scoring='f1_weighted')),
            'roc_auc': np.mean(cross_val_score(model, X, y, cv=5, scoring='roc_auc_ovr_weighted'))
        }
    
    average_scores = {name: sum(scores.values()) / len(scores) for name, scores in results.items()}
    best_model_name = max(average_scores, key=average_scores.get)
    best_model = models[best_model_name].fit(X, y)  # Train the best model on all data
    
    return best_model, best_model_name, results, average_scores

# Visualization function for model results
def plot_results(results):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    metrics = list(next(iter(results.values())).keys())
    x = np.arange(len(metrics))
    width = 0.2
    
    for i, (model, scores) in enumerate(results.items()):
        ax.bar(x + i*width, list(scores.values()), width, label=model)
    
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(metrics)
    ax.legend(loc='best')
    
    return fig

# Interpret risk function
def interpret_risk(risk_category, probabilities):
    risk_levels = {
        0: 'Low',  # Changed from Low
        1: 'Moderate',
        2: 'High',  # Changed from High
        'High_risk': 'High',
        'Medium_risk': 'Moderate',
        'Low_risk': 'Low'
    }
    
    interpreted_risk = risk_levels.get(risk_category, 'Unknown')
    highest_prob = max(probabilities.values())
    confidence = 'Low' if highest_prob < 0.5 else 'Moderate' if highest_prob < 0.75 else 'High'
    
    return interpreted_risk, confidence

# Plot risk probabilities function
def plot_risk_probabilities(probabilities):
    fig, ax = plt.subplots(figsize=(8, 6))
    categories = list(probabilities.keys())
    values = list(probabilities.values())
    
    colors = ['green', 'yellow', 'red']
    bars = ax.bar(categories, values, color=colors)
    
    ax.set_ylabel('Probability')
    ax.set_title('Risk Category Probabilities')
    ax.set_ylim(0, 1)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom')
    
    return fig

# Main app
def main():
    st.title('Suicide Ideation Risk Assessment Model')

    # Add email inputs to sidebar
    st.sidebar.header("Contact Information")
    user_email = st.sidebar.text_input("Your email address")
    kin_email = st.sidebar.text_input("Next of kin's email address (optional)")
    
    data = load_data()
    X = data.drop(['response', 'risk_category'], axis=1)
    y = data['risk_category']
    
    best_model, best_model_name, results, average_scores = train_models(X, y)
    
    page = st.sidebar.radio('Choose a page', ['Model Results', 'Risk Assessment'])
    
    if page == 'Model Results':
        st.header('Model Training Results')
        st.pyplot(plot_results(results))
        
        st.subheader('Detailed Results')
        st.write(pd.DataFrame(results).T)
        
        st.subheader('Average Scores')
        st.write(pd.DataFrame.from_dict(average_scores, orient='index', columns=['Average Score']))
        
        st.subheader('Best Model')
        st.write(f"The best performing model is: {best_model_name}")
        st.write(f"Average score: {average_scores[best_model_name]:.4f}")
        
    elif page == 'Risk Assessment':
        st.header('Suicide Ideation Risk Assessment')
        st.write(f"Using the best model: {best_model_name}")
        
        input_data = {}
        for column, question in zip(X.columns, list(QUESTIONS.values())):
            st.write(question['text'])
            with st.expander("Options"):
                for option in question['options']:
                    st.write(option)
            input_data[column] = st.number_input(f"Enter value for {column}", min_value=1, max_value=5, value=1, step=1)
        
        if st.button('Assess Risk'):
            input_df = pd.DataFrame([input_data])
            
            prediction = best_model.predict(input_df)[0]
            probabilities = dict(zip(best_model.classes_, best_model.predict_proba(input_df)[0]))
            
            interpreted_risk, confidence = interpret_risk(prediction, probabilities)
            
            st.subheader('Risk Assessment Results')
            st.write(f'Predicted Risk Level: **{interpreted_risk}**')
            st.write(f'Assessment Confidence: **{confidence}**')
            
            st.write('### Risk Probability Breakdown')
            st.pyplot(plot_risk_probabilities(probabilities))
            
            st.write('### Interpretation')
            if interpreted_risk == 'Low':
                interpretation = "The model suggests a low risk of suicide ideation. However, any level of risk should be taken seriously."
            elif interpreted_risk == 'Moderate':
                interpretation = "The model indicates a moderate risk of suicide ideation. It's advisable to seek professional help or support."
            else:
                interpretation = "The model indicates a high risk of suicide ideation. Immediate professional help and support are strongly recommended."
            
            st.write(interpretation)

            st.write("**Important**: This assessment is based on a machine learning model and should not be considered as a substitute for professional medical advice, diagnosis, or treatment. If you or someone you know is experiencing thoughts of suicide, please seek immediate help from a qualified mental health professional or contact a suicide prevention hotline.")
            
            # Send emails if provided
            if user_email or kin_email:
                send_assessment_emails(user_email=user_email, kin_email=kin_email, interpreted_risk=interpreted_risk, confidence=confidence, interpretation=interpretation)

            

show_code(main)

if __name__ == '__main__':
    main()
