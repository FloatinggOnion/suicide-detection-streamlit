# Copyright 2018-2022 Streamlit Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import streamlit as st
import inspect
import textwrap

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import ssl

from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

import os
import base64

def show_code(demo):
    """Showing the code of the demo."""
    show_code = st.sidebar.checkbox("Show code", False)
    if show_code:
        # Showing the code of the demo.
        st.markdown("## Code")
        sourcelines, _ = inspect.getsourcelines(demo)
        st.code(textwrap.dedent("".join(sourcelines[1:])))



def send_email(recipient, subject, body):
    # Email configuration
    sender_email = os.environ.get('SENDER_EMAIL')
    sender_password = os.environ.get('SENDER_PASSWORD')
    smtp_server = "smtp.gmail.com"
    smtp_port = 587  # SSL port for Gmail


    # Create message
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = recipient
    message["Subject"] = subject

    # Add body to email
    message.attach(MIMEText(body, "plain"))

    try:
        # Create secure SSL context
        context = ssl.create_default_context()

        # Create SMTP_SSL session
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls(context=context)
            server.login(sender_email, sender_password)
            
            # Send email
            server.sendmail(sender_email, recipient, message.as_string())
        
        print(f"Email sent successfully to {recipient}")
        return True
    except Exception as e:
        print(f"Failed to send email to {recipient}. Error: {str(e)}")
        return False

# Usage in your Streamlit app
def send_assessment_emails(user_email, kin_email, interpreted_risk, confidence, interpretation):
    if user_email:
        subject = "Your Suicide Ideation Risk Assessment Results"
        body = f"""
        Dear User,

        Here are your Suicide Ideation Risk Assessment results:

        Risk Level: {interpreted_risk}
        Confidence: {confidence}

        Interpretation:
        {interpretation}

        IMPORTANT: This assessment is based on a machine learning model and should not be considered as a substitute for professional medical advice, diagnosis, or treatment. If you are experiencing thoughts of suicide, please seek immediate help from a qualified mental health professional or contact a suicide prevention hotline.

        Take care,
        Your Assessment Team
        """
        send_email(user_email, subject, body)
    
    if kin_email:
        subject = "Suicide Ideation Risk Assessment Results for Your Loved One"
        body = f"""
        Dear Next of Kin,

        A Suicide Ideation Risk Assessment was conducted for your loved one. Here are the results:

        Risk Level: {interpreted_risk}
        Confidence: {confidence}

        Interpretation:
        {interpretation}

        IMPORTANT: This assessment is based on a machine learning model and should not be considered as a substitute for professional medical advice, diagnosis, or treatment. If you are concerned about your loved one, please encourage them to seek help from a qualified mental health professional or contact a suicide prevention hotline.

        Take care,
        The Assessment Team
        """
        send_email(kin_email, subject, body)

# In your main Streamlit app, you can call this function after performing the assessment:
# send_assessment_emails(user_email, kin_email, interpreted_risk, confidence, interpretation)