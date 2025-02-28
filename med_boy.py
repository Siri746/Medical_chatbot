#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import streamlit as st
import pandas as pd
import google.generativeai as genai
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

st.set_page_config(page_title="Medical Chatbot 🩺", page_icon="🩺", layout="centered")

if "conversation" not in st.session_state:
    st.session_state.conversation = []

def load_knowledge_base(file_path):
    try:
        if not os.path.exists(file_path):
            st.error(f"File not found: {file_path}")
            return None
        df = pd.read_csv(file_path)
        if df.empty:
            st.error(f"The file {file_path} is empty.")
            return None
        return df
    except Exception as e:
        st.error(f"Error loading the knowledge base: {e}")
        return None

def preprocess_data(df):
    df = df.fillna("")
    df['short_question'] = df['short_question'].str.lower()
    df['short_answer'] = df['short_answer'].str.lower()
    return df

def create_vectorizer(df):
    vectorizer = TfidfVectorizer()
    question_vectors = vectorizer.fit_transform(df['short_question'])
    return vectorizer, question_vectors

def find_closest_question(user_query, vectorizer, question_vectors, df):
    query_vector = vectorizer.transform([user_query.lower()])
    similarities = cosine_similarity(query_vector, question_vectors).flatten()
    best_match_index = similarities.argmax()
    best_match_score = similarities[best_match_index]
    if best_match_score > 0.3:
        return df.iloc[best_match_index]['short_answer']
    else:
        return None

def configure_generative_model(api_key):
    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel('gemini-1.5-flash')
    except Exception as e:
        st.error(f"Error configuring the generative model: {e}")
        return None

def refine_answer_with_gemini(generative_model, user_query, closest_answer):
    try:
        context = """
        You are a medical chatbot. Refine the following answer to make it more professional, clear, and actionable.
        Ensure the response is detailed, well-structured, and includes bullet points for clarity.
        """
        prompt = f"{context}\n\nUser Query: {user_query}\nClosest Answer: {closest_answer}\nRefined Answer:"
        response = generative_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error refining the answer: {e}"

def medical_chatbot(df, vectorizer, question_vectors, generative_model):
    st.title("Medical Chatbot 🩺")
    st.write("Welcome to the Medical Chatbot! Ask me anything about medical topics.")

    st.markdown("### Conversation History")
    for message in st.session_state.conversation:
        if message["role"] == "User":
            st.markdown(f"<div style='background-color: #e6f7ff; padding: 10px; border-radius: 10px; margin: 5px 0;'><strong>You:</strong> {message['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='background-color: #f0f0f0; padding: 10px; border-radius: 10px; margin: 5px 0;'><strong>Bot:</strong> {message['content']}</div>", unsafe_allow_html=True)

    user_query = st.text_input("User:", placeholder="Type your question here...", key="user_input")

    if user_query:
        st.session_state.conversation.append({"role": "You", "content": user_query})
        closest_answer = find_closest_question(user_query, vectorizer, question_vectors, df)

        if closest_answer:
            with st.spinner("Refining the answer..."):
                refined_answer = refine_answer_with_gemini(generative_model, user_query, closest_answer)
                st.session_state.conversation.append({"role": "Bot", "content": refined_answer})
                st.markdown(f"<div style='background-color: #f0f0f0; padding: 10px; border-radius: 10px; margin: 5px 0;'><strong>Bot (refined answer):</strong> {refined_answer}</div>", unsafe_allow_html=True)
        else:
            try:
                context = """
                You are a medical chatbot. Provide accurate, detailed, and well-structured answers to medical questions.
                If the user describes symptoms, suggest possible causes and recommend consulting a doctor for a proper diagnosis.
                Use a professional tone and format the response clearly with bullet points.
                """
                prompt = f"{context}\n\nUser: {user_query}\nBot:"
                response = generative_model.generate_content(prompt)
                st.session_state.conversation.append({"role": "Bot", "content": response.text})
                st.markdown(f"<div style='background-color: #f0f0f0; padding: 10px; border-radius: 10px; margin: 5px 0;'><strong>Bot (AI-generated):</strong> {response.text}</div>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Sorry, I couldn't generate a response. Error: {e}")

def main():
    file_path = "med_bot_data.csv"
    df = load_knowledge_base(file_path)
    if df is None:
        return
    df = preprocess_data(df)
    vectorizer, question_vectors = create_vectorizer(df)
    API_KEY = "AIzaSyA-9-lTQTWdNM43YdOXMQwGKDy0SrMwo6c"
    if not API_KEY:
        st.error("API key not found. Please set the GOOGLE_API_KEY environment variable.")
        return
    generative_model = configure_generative_model(API_KEY)
    if generative_model is None:
        return
    medical_chatbot(df, vectorizer, question_vectors, generative_model)

if __name__ == "__main__":
    main()
