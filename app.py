import streamlit as st
import plotly.graph_objects as go
from prediction import prediction_emotion
from llm_response import generate_support_response



# Page Config

st.set_page_config(
    page_title="AI Mental Health Assistant",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 AI Mental Health Assistant")
st.title("Aapka apna sathi😊")



# Session State Initialization

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "emotion_history" not in st.session_state:
    st.session_state.emotion_history = []

if "show_analysis" not in st.session_state:
    st.session_state.show_analysis = False



# Emotion Analysis Function

def calculate_overall_emotion(emotion_history):

    emotion_sum = {
        "Sadness": 0,
        "Anxiety": 0,
        "Anger": 0,
        "Burnout": 0,
        "Positive": 0,
        "Neutral": 0
    }

    count = len(emotion_history)

    if count == 0:
        return emotion_sum

    for emotion in emotion_history:
        for key in emotion_sum:
            emotion_sum[key] += float(emotion[key])

    # convert to percentage
    for key in emotion_sum:
        emotion_sum[key] = (emotion_sum[key] / count) * 100

    return emotion_sum



# Plot Function

def plot_emotion_chart(emotion_percentages):

    emotions = list(emotion_percentages.keys())
    percentages = list(emotion_percentages.values())

    fig = go.Figure(
        data=[
            go.Bar(
                x=emotions,
                y=percentages,
                text=[f"{p:.1f}%" for p in percentages],
                textposition="auto",
                marker_color="skyblue"
            )
        ]
    )

    fig.update_layout(
        title="Session Emotional Analysis",
        yaxis=dict(range=[0, 100]),
        yaxis_title="Percentage %",
        xaxis_title="Emotion",
        height=400
    )

    return fig



# Chat Input

user_input = st.chat_input("Type your message...")


if user_input:

    # Save user message
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_input
    })

   
    emotion_result = prediction_emotion(user_input)

    # Save emotion in background
    st.session_state.emotion_history.append(emotion_result)

   
    llm_response = generate_support_response(user_input, emotion_result)

    # Save assistant message
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": llm_response
    })



# Display Chat History

for message in st.session_state.chat_history:

    if message["role"] == "user":

        with st.chat_message("user"):
            st.write(message["content"])

    else:

        with st.chat_message("assistant"):
            st.write(message["content"])



# Analysis Toggle Buttons

st.divider()

col1, col2 = st.columns(2)

with col1:
    if st.button("Show Emotion Analysis"):
        st.session_state.show_analysis = True

with col2:
    if st.button("Hide Emotion Analysis"):
        st.session_state.show_analysis = False



# Display Emotion Analysis

if st.session_state.show_analysis and len(st.session_state.emotion_history) > 0:

    st.subheader("📊 Emotional Analysis (Session)")

    overall_emotion = calculate_overall_emotion(
        st.session_state.emotion_history
    )

    fig = plot_emotion_chart(overall_emotion)

    st.plotly_chart(fig, use_container_width=True)

    dominant = max(overall_emotion, key=overall_emotion.get)

    st.success(
        f"Dominant Emotion: {dominant} ({overall_emotion[dominant]:.1f}%)"
    )
