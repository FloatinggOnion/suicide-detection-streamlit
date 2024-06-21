import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Page configuration
st.set_page_config(page_title="Suicide Ideation Prediction", layout="wide")

# Define questions and options
questions = {
    'q1': {
        'text': "What level are you?",
        'options': ["1: 100", "2: 200", "3: 300", "4: 400", "5: 500"]
    },
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

# Define metrics
metrics = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'roc_auc_ovr']

# Load and preprocess data
@st.cache_data
def load_data():
    data = pd.read_csv('suiciderisk.csv')
    numeric_data = data.select_dtypes(include='number')
    data['total_score'] = numeric_data.sum(axis=1)
    
    def map_risk(total_score):
        low_risk_range = range(20, 41)
        medium_risk_range = range(41, 61)
        high_risk_range = range(61, 101)
        
        if total_score in low_risk_range:
            return 'Low_risk'
        elif total_score in medium_risk_range:
            return 'Medium_risk'
        elif total_score in high_risk_range:
            return 'High_risk'
        else:
            return 'Unknown_risk'
    
    data['risk_category'] = data['total_score'].apply(map_risk)
    
    le = LabelEncoder()
    for column in data.select_dtypes(include=['object']).columns:
        data[column] = le.fit_transform(data[column])
    
    X = data.drop(['total_score', 'risk_category'], axis=1)
    y = data['risk_category']
    
    return X, y, data

X, y, data = load_data()

# Train models
@st.cache_resource
def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        'Naive Bayes': GaussianNB(),
        'SVM': SVC(probability=True),
        'Random Forest': RandomForestClassifier(),
        'Logistic Regression': LogisticRegression(max_iter=500)
    }

    results = {}
    fitted_models = {}
    
    for model_name, model in models.items():
        print(f"Training {model_name}...")
        model.fit(X_train, y_train)
        fitted_models[model_name] = model
        
        model_results = {}
        for metric in metrics:
            cv_scores = cross_val_score(model, X, y, cv=3, scoring=metric)
            model_results[metric] = cv_scores.mean()

        results[model_name] = model_results

    average_scores = {model_name: sum(scores.values()) / len(scores) for model_name, scores in results.items()}
    best_model_name = max(average_scores, key=average_scores.get)

    return results, average_scores, best_model_name, fitted_models

results, average_scores, best_model_name, fitted_models = train_models(X, y)

# Interpret risk function
def interpret_risk(risk_category, probabilities):
    risk_levels = {
        'Low_risk': 'Low',
        'Medium_risk': 'Moderate',
        'High_risk': 'High'
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

# Sidebar for navigation
page = st.sidebar.radio("Navigate", ["Model Results", "Test Best Model"])

if page == "Model Results":
    st.header("Suicide Ideation Prediction Model Results")

    # Visualization options
    viz_type = st.selectbox("Select Visualization", ["Vertical Bar Chart", "Horizontal Bar Chart", "Individual Metrics"])

    if viz_type == "Vertical Bar Chart":
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(results))
        width = 0.15
        
        for i, metric in enumerate(metrics):
            scores = [results[model][metric] for model in results]
            ax.bar(x + i*width, scores, width, label=metric)
        
        ax.set_xlabel('Machine Learning Models')
        ax.set_ylabel('Score')
        ax.set_title('Evaluation Metrics for Suicide Ideation Prediction Models')
        ax.set_xticks(x + 2*width)
        ax.set_xticklabels(results.keys())
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)

    elif viz_type == "Horizontal Bar Chart":
        fig, ax = plt.subplots(figsize=(10, 6))
        y_pos = np.arange(len(results))
        metrics_scores = {metric: [results[model][metric] for model in results] for metric in metrics}
        
        bar_width = 0.15
        for i, metric in enumerate(metrics):
            ax.barh(y_pos + i*bar_width, metrics_scores[metric], bar_width, label=metric)
        
        ax.set_xlabel('Score')
        ax.set_ylabel('Machine Learning Models')
        ax.set_title('Evaluation Metrics for Suicide Ideation Prediction Models')
        ax.set_yticks(y_pos + 2*bar_width)
        ax.set_yticklabels(results.keys())
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)

    else:  # Individual Metrics
        fig, axs = plt.subplots(nrows=len(metrics), ncols=1, figsize=(10, 15))
        for i, metric in enumerate(metrics):
            ax = axs[i]
            ax.barh(list(results.keys()), [results[model][metric] for model in results], color=plt.cm.Paired(np.arange(len(results))))
            ax.set_xlabel(metric)
            ax.set_xlim([0, 1.05])
            ax.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig)

    st.subheader('Detailed Results')
    st.write(pd.DataFrame(results).T)
    
    st.subheader('Average Scores')
    st.write(pd.DataFrame.from_dict(average_scores, orient='index', columns=['Average Score']))
    
    st.subheader('Best Model')
    st.write(f"The best performing model is: {best_model_name}")
    st.write(f"Average score: {average_scores[best_model_name]:.4f}")

else:  # Test Best Model page
    st.header(f"Test Best Model: {best_model_name}")

    # Create input fields for each feature
    st.subheader("Enter test data:")
    input_data = {}
    for column, question in zip(X.columns, questions.values()):
        st.write(question['text'])
        with st.expander("Options"):
            for option in question['options']:
                st.write(option)
        input_data[column] = st.number_input(f"Enter value for {column}", min_value=1, max_value=5, value=1, step=1)

    if st.button("Predict"):
        input_df = pd.DataFrame([input_data])
        total_score = input_df.sum(axis=1).values[0]
        
        st.subheader("Model Prediction:")
        best_model = fitted_models[best_model_name]
        
        # Map the total score to a risk category
        if total_score in range(20, 41):
            prediction = 'Low_risk'
        elif total_score in range(41, 61):
            prediction = 'Medium_risk'
        elif total_score in range(61, 101):
            prediction = 'High_risk'
        else:
            prediction = 'Unknown_risk'
        
        probabilities = dict(zip(best_model.classes_, best_model.predict_proba(input_df)[0]))
        interpreted_risk, confidence = interpret_risk(prediction, probabilities)
        
        st.subheader('Risk Assessment Results')
        st.write(f'Total Score: **{total_score}**')
        st.write(f'Predicted Risk Level: **{interpreted_risk}**')
        st.write(f'Assessment Confidence: **{confidence}**')
        
        st.write('### Risk Probability Breakdown')
        st.pyplot(plot_risk_probabilities(probabilities))
        
        st.write('### Interpretation')
        if interpreted_risk == 'Low':
            st.write("The model suggests a low risk of suicide ideation. However, any level of risk should be taken seriously.")
        elif interpreted_risk == 'Moderate':
            st.write("The model indicates a moderate risk of suicide ideation. It's advisable to seek professional help or support.")
        else:
            st.write("The model indicates a high risk of suicide ideation. Immediate professional help and support are strongly recommended.")
        
        st.write("**Important**: This assessment is based on a machine learning model and should not be considered as a substitute for professional medical advice, diagnosis, or treatment. If you or someone you know is experiencing thoughts of suicide, please seek immediate help from a qualified mental health professional or contact a suicide prevention hotline.")

st.sidebar.write("Note: This is a demonstration app. The model results are simulated and may not reflect actual performance.")