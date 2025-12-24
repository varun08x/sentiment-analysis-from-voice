import streamlit as st
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import speech_recognition as sr
from transformers import pipeline

st.set_page_config(page_title="Sentiment Analysis from Voice")

st.title("🎤 Sentiment Analysis from Voice")
st.write("Click the button and speak clearly")

fs = 16000
duration = 7


def get_default_input_device():
    device = sd.query_devices(kind="input")
    return device


if st.button("🎙 Record Voice"):
    try:
        device = get_default_input_device()
        channels = device["max_input_channels"]

        st.info(f"Using microphone: {device['name']}")
        st.info("Recording... Speak now")

        audio = sd.rec(
            int(duration * fs),
            samplerate=fs,
            channels=channels,   # AUTO FIX
            dtype="int16"
        )
        sd.wait()

        # Convert stereo → mono if needed
        if channels > 1:
            audio = np.mean(audio, axis=1, dtype=np.int16)

        write("temp.wav", fs, audio)
        st.success("Recording finished")

        recognizer = sr.Recognizer()
        recognizer.energy_threshold = 300
        recognizer.dynamic_energy_threshold = True

        with sr.AudioFile("temp.wav") as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)
            audio_data = recognizer.record(source)

        text = recognizer.recognize_google(audio_data)

        st.subheader("📝 Recognized Text")
        st.write(text)

        sentiment_model = pipeline("sentiment-analysis")
        result = sentiment_model(text)[0]

        st.subheader("📊 Sentiment Result")
        st.write(f"**Sentiment:** {result['label']}")
        st.write(f"**Confidence:** {round(result['score'] * 100, 2)}%")

    except sr.UnknownValueError:
        st.error("Speech not clear. Please speak slowly and clearly.")

    except Exception as e:
        st.error(f"Error: {e}")
