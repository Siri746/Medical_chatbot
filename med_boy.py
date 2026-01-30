import streamlit as st
import pandas as pd
import google.generativeai as genai
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

st.set_page_config(page_title="Medical Chatbot 🩺", page_icon="🩺", layout="centered")

# Use Environment Variables for security
API_KEY = os.getenv("GOOGLE_API_KEY")

if "conversation" not in st.session_state:
    st.session_state.conversation = []

@st.cache_data # Cache data so it doesn't reload on every click
def load_knowledge_base(file_path):
    if not os.path.exists(file_path):
        return None
    df = pd.read_csv(file_path)
    df = df.fillna("")
    df['short_question'] = df['short_question'].str.lower()
    return df

@st.cache_resource # Cache the vectorizer
def get_vectors(df):
    vectorizer = TfidfVectorizer()
    question_vectors = vectorizer.fit_transform(df['short_question'])
    return vectorizer, question_vectors

def find_closest_question(user_query, vectorizer, question_vectors, df):
    query_vector = vectorizer.transform([user_query.lower()])
    similarities = cosine_similarity(query_vector, question_vectors).flatten()
    best_match_index = similarities.argmax()
    if similarities[best_match_index] > 0.3:
        return df.iloc[best_match_index]['short_answer']
    return None

def main():
    st.title("Medical Chatbot 🩺")
    
    if not API_KEY:
        st.error("Please set the GOOGLE_API_KEY in your Environment Variables.")
        st.stop()

    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')

    df = load_knowledge_base("med_bot_data.csv")
    if df is not None:
        vectorizer, question_vectors = get_vectors(df)

    # Display Chat History
    for chat in st.session_state.conversation:
        with st.chat_message(chat["role"]):
            st.markdown(chat["content"])

    # Chat Input
    if prompt := st.chat_input("How can I help you today?"):
        st.session_state.conversation.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            
            # Logic: Check CSV first, then Gemini
            closest_answer = find_closest_question(prompt, vectorizer, question_vectors, df) if df is not None else None
            
            if closest_answer:
                full_prompt = f"Refine this medical answer for clarity: {closest_answer}"
            else:
                full_prompt = f"System: You are a professional medical assistant. User Question: {prompt}"

            try:
                response = model.generate_content(full_prompt)
                assistant_response = response.text
                response_placeholder.markdown(assistant_response)
                st.session_state.conversation.append({"role": "assistant", "content": assistant_response})
            except Exception as e:
                st.error("Model Error: Check your API Key or Quota.")

if __name__ == "__main__":
    main()
