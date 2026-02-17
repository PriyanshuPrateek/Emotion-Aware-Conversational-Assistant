import os
import streamlit as st
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

if "Grok_API_KEY" in st.secrets:
    groq_api_key = st.secrets("Grok_API_KEY")
else:
     groq_api_key = os.getenv("Grok_API_KEY")    



llm = ChatGroq(
    api_key=groq_api_key,
    model_name= "meta-llama/llama-4-scout-17b-16e-instruct"
)

def generate_support_response(text, emotion_result):

    prompt = f"""
        You are an empathetic emotional support assistant.

        Your task is to generate a supportive, understanding, and compassionate response to the user.

     User message:
     \"\"\"{text}\"\"\"

     Detected emotions: {emotion_result}

        Instructions:
        Acknowledge the user's feelings in a natural and validating way.
        Show empathy, care, and emotional understanding.
        Base the response on the top-3 detected emotions, giving highest priority to the emotion with the highest confidence score, while also considering the emotional context of the other two.
        Ensure the tone and wording primarily reflect the most dominant emotion, with subtle support for secondary emotions if relevant.
        Do NOT invalidate, dismiss, or judge their emotions.
        Do NOT give overly clinical, robotic, or generic responses.
        Avoid toxic positivity (do not say things like "just be happy").
        If the dominant emotion is negative (sadness, anger, fear, anxiety, loneliness), offer gentle reassurance and emotional support.
        If the dominant emotion is positive (happy, excited, proud), encourage and celebrate with them.
        Keep the tone warm, human-like, and supportive.
        Keep the response concise (2–4 sentences).
        Do NOT mention emotion detection explicitly.

        Supportive response:
        """

    response = llm.invoke(prompt)    

    return response.content