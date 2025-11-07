import json
import requests
import streamlit as st
from streamlit_lottie import st_lottie
import pandas as pd
import pickle
import os

# Load the model
with open(r"C:\Users\svsas\Desktop\Fitness Tracker\src\random_forest.pickle", "rb") as f:
    rf = pickle.load(f)

# Define the feature columns
feature_columns = [
    "acc_r_freq_2.5_Hz_ws_14",
    "acc_x_freq_1.071_Hz_ws_14",
    "acc_y_freq_1.429_Hz_ws_14",
    "gyr_r_freq_1.786_Hz_ws_14",
    "gyr_z_freq_0.357_Hz_ws_14",
    "acc_z_freq_weighted",
    "gyr_y_freq_0.714_Hz_ws_14",
    "acc_y_freq_0.357_Hz_ws_14",
    "acc_z_max_freq",
    "acc_r",
    "acc_r_max_freq",
    "acc_r_freq_1.786_Hz_ws_14",
    "gyr_x_temp_std_ws_5",
    "acc_z",
    "gyr_x_freq_1.786_Hz_ws_14",
    "gyr_r_freq_weighted",
    "gyr_r_freq_2.143_Hz_ws_14",
    "gyr_r_freq_0.357_Hz_ws_14",
    "acc_r_freq_0.357_Hz_ws_14",
    "acc_x_freq_0.714_Hz_ws_14",
    "gyr_x_pse",
    "acc_y_freq_0.714_Hz_ws_14",
    "acc_x_max_freq",
    "acc_z_temp_std_ws_5",
    "acc_z_freq_2.143_Hz_ws_14",
    "acc_y_freq_2.5_Hz_ws_14",
    "acc_x_freq_2.5_Hz_ws_14",
    "gyr_x_max_freq",
    "gyr_x",
    "gyr_z_temp_mean_ws_5",
]


def predict_workout_type(features):
    """Make a prediction using the loaded model."""
    features = pd.DataFrame([features], columns=feature_columns)
    prediction = rf.predict(features)
    return prediction[0]


def load_lottieurl(url: str):
    """Fetch Lottie animation from a URL."""
    r = requests.get(url)
    if r.status_code != 200:
        st.error("Failed to load Lottie animation from the URL provided.")
        return None
    return r.json()


# Streamlit app UI
st.set_page_config(
    page_title="Workout Type Prediction", page_icon="💪", layout="centered"
)

# Add CSS for a modern, soothing design with light animations
st.markdown(
    """
    <style>
    /* General reset */
    *, *::before, *::after {
        padding: 0;
        margin: 0;
        box-sizing: border-box;
    }

    body {
        font-family: Raleway, sans-serif;
        background-color: #000;
        color: #fff;
        font-size: 24px;
    }

    .content {
        width: 95%;
        max-width: 40ch;
        padding: 3em 1em;
    }

    .marquee {
        position: relative;
        width: 100%;
        height: 2em;
        font-size: 5em;
        display: grid;
        place-items: center;
        overflow: hidden;
    }

    .marquee_text {
        position: absolute;
        min-width: 100%;
        white-space: nowrap;
        animation: marquee 16s infinite linear;
    }

    @keyframes marquee {
        from { transform: translateX(100%); }
        to { transform: translateX(-100%); }
    }

    .marquee_blur {
        position: absolute;
        inset: 0;
        display: grid;
        place-items: center;
        background-color: black;
        background-image:
            linear-gradient(to right, white, 1rem, transparent 50%),
            linear-gradient(to left, white, 1rem, transparent 50%);
        filter: contrast(15);
    }

    .marquee_clear {
        position: absolute;
        inset: 0;
        display: grid;
        place-items: center;
    }

    .text {
        margin-block: 2em;
    }

    .block-container {
        background-color: #FFFFFF;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        max-width: 800px;
        margin: auto;
        margin-top: 30px;
        transition: all 0.3s ease-in-out;
    }

    .block-container:hover {
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.2);
    }

    .main-header {
        font-size: 34px;
        text-align: center;
        color: #3E4C59;
        background-color: #9CA3AF;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 20px;
        animation: fadeIn 1.5s ease-in-out;
    }

    .instructions {
        font-size: 18px;
        margin-top: 20px;
        color: #6B7280;
    }

    .stSlider div {
        background-color: #F9FAFB;
        color: #111827;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 10px;
        transition: border 0.3s ease;
    }

    .stSlider div:focus {
        border-color: #60A5FA;
        box-shadow: 0 0 5px rgba(96, 165, 250, 0.3);
    }

    .stSlider label {
        color: #60A5FA;
    }

    .stButton button {
        background-color: #4F46E5;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
        border: none;
        transition: background-color 0.3s ease, transform 0.2s ease;
    }

    .stButton button:hover {
        background-color: #6D28D9;
        transform: translateY(-3px);
    }

    .animation-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-top: 20px;
    }

    .typewriter-container {
        max-width: 100%;
        overflow: hidden;
        text-align: center;
        position: relative;
        margin: 0 auto;
    }

    .typewriter-text {
        display: inline-block;
        white-space: nowrap;
        font-size: 30px;
        animation: typewriter 4s steps(44) 1s 1 normal both,
                   blinkTextCursor 500ms steps(44) infinite normal;
        font-weight: bold;
        text-transform: uppercase;
    }

    @keyframes typewriter {
        from { width: 0; }
        to { width: 100%; }
    }

    @keyframes blinkTextCursor {
        from { border-right-color: rgba(255,255,255,.75); }
        to { border-right-color: transparent; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# App Header
st.markdown(
    '<div class="main-header">💪 Workout Type Prediction</div>', unsafe_allow_html=True
)

# Load Lottie animations
initial_lottie_url = (
    "https://lottie.host/62913c06-adbe-4f4c-83c1-8c9b3c10dea9/VIXwaGjBPk.json"
)
initial_lottie_animation = load_lottieurl(initial_lottie_url)

result_lottie_url = (
    "https://lottie.host/63337eda-24a2-4c2e-b3e4-b7872c5fd855/hnh0Sv5t4N.json"
)
result_lottie_animation = load_lottieurl(result_lottie_url)

new_lottie_url = (
    "https://lottie.host/8d982cda-7c49-4b1e-87c7-959b1a0b6f9a/E1OUUceyHQ.json"
)
new_lottie_animation = load_lottieurl(new_lottie_url)

# Show the initial animation
if initial_lottie_animation:
    st.markdown('<div class="animation-container">', unsafe_allow_html=True)
    st_lottie(
        initial_lottie_animation,
        speed=1,
        reverse=False,
        loop=True,
        quality="low",  # Options: low, medium, high
        height=400,
        width=600,
        key="initial_animation",
    )
    st.markdown("</div>", unsafe_allow_html=True)

# Instructions for the user
st.markdown(
    "<p class='instructions'>Adjust the sliders to set feature values, and the model will predict your workout type (bench, deadlift, OHP, rest, row, or squat):</p>",
    unsafe_allow_html=True,
)

# Show the new Lottie animation
if new_lottie_animation:
    st.markdown('<div class="animation-container">', unsafe_allow_html=True)
    st_lottie(
        new_lottie_animation,
        speed=1,
        reverse=False,
        loop=True,
        quality="low",  # Options: low, medium, high
        height=400,
        width=600,
        key="new_animation",
    )
    st.markdown("</div>", unsafe_allow_html=True)

# Create slider inputs for all features in a two-column layout
with st.expander("Adjust Feature Values"):
    col1, col2 = st.columns(2)

    with col1:
        for feature in feature_columns[: len(feature_columns) // 2]:
            st.slider(
                f"{feature}",
                min_value=-15.00,
                max_value=15.00,
                step=0.01,
                value=0.0,
                key=f"input_{feature}",
            )

    with col2:
        for feature in feature_columns[len(feature_columns) // 2 :]:
            st.slider(
                f"{feature}",
                min_value=-15.00,
                max_value=15.00,
                step=0.01,
                value=0.0,
                key=f"input_{feature}",
            )

# Placeholder for the result animation
result_animation_placeholder = st.empty()

# Predict button
if st.button("Predict"):
    # Collect feature values
    feature_values = [
        st.session_state[f"input_{feature}"] for feature in feature_columns
    ]

    # Predict the workout type
    prediction = predict_workout_type(feature_values)

    # Define workout type colors
    workout_type_colors = {
        "bench": "#A5B4FC",
        "deadlift": "#FDBA74",
        "OHP": "#FCA5A5",
        "rest": "#F4DE1A",
        "row": "#6EE7B7",
        "squat": "#93C5FD",
    }

    prediction_color = workout_type_colors.get(prediction, "#2E549E")

    # Display the result with enhanced styling and typewriter effect
    st.markdown(
        f"""
        <div class="typewriter-container">
            <div class="typewriter-text">
                Predicted Workout Type: <span style="color: {prediction_color};">{prediction}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Show the result animation
    result_animation_placeholder.markdown(
        '<div class="animation-container">', unsafe_allow_html=True
    )
    st_lottie(
        result_lottie_animation,
        speed=1,
        reverse=False,
        loop=True,
        quality="low",  # Options: low, medium, high
        height=400,
        width=600,
        key="result_animation2",
    )
    result_animation_placeholder.markdown("</div>", unsafe_allow_html=True)

    
    
    # # # Map prediction label to image file path
    # workout_type_images = {
    #     "bench": "images/benchpress.jpg",
    #     "deadlift": "images/deadlift.png",
    #     "OHP": "images/ohp.jpg",
    #     "rest": "images/rest.jpg",
    #     "row": "images/row.jpg",
    #     "squat": "src/images/squat.jpg",
    # }

    # # Get image path corresponding to predicted workout
    # display_image_path = workout_type_images.get(prediction)  # fallback image

    # if display_image_path and os.path.exists(display_image_path):
    #     st.image(display_image_path, caption=f"{prediction} exercise", use_container_width=True)
    # else:
    #     st.warning(f"Image not found: {display_image_path}")
        
    # # Show the image in the Streamlit app
    # st.image(display_image_path, caption=f"{prediction} exercise", use_container_width=True)

    # # # Show the prediction image
    # st.image(display_image_path, caption=f"This is a {prediction} exercise")


    


