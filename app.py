import cv2
import numpy as np
import av
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer
import mediapipe as mp
import paho.mqtt.client as mqtt
import atexit

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
if "hands" not in st.session_state:
    st.session_state.hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7)

# MQTT
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
MQTT_TOPIC = "streamlit/gesto"
if "mqtt_client" not in st.session_state:
    st.session_state.mqtt_client = mqtt.Client()
    st.session_state.mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)

# Clean up resources on exit
atexit.register(st.session_state.hands.close)
atexit.register(st.session_state.mqtt_client.disconnect)

def detectar_gesto(landmarks):
    dedos_estirados = []
    tips_ids = [4, 8, 12, 16, 20]

    # Pulgar (eje X)
    dedos_estirados.append(landmarks[4][0] > landmarks[3][0])

    # Otros dedos (eje Y)
    for i in [8, 12, 16, 20]:
        dedos_estirados.append(landmarks[i][1] < landmarks[i - 2][1])

    # Distance calculation for OK gesture
    dist = np.sqrt((landmarks[4][0] - landmarks[8][0])**2 + (landmarks[4][1] - landmarks[8][1])**2)
    if dedos_estirados == [False] * 5:
        return "Puño cerrado ✊"
    elif dedos_estirados == [True] * 5:
        return "Palma abierta 🖐"
    elif dist < 0.05 and all(dedos_estirados[i] for i in [2, 3, 4]):
        return "Gesto OK 👌"
    return None

frame_counter = 0
last_gesture = None
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    global frame_counter, last_gesture
    frame_counter += 1
    if frame_counter % 2 != 0:  # Process every 2nd frame
        return frame

    image = frame.to_ndarray(format="bgr24")
    image = cv2.resize(image, (320, 240))  # Lower resolution
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = st.session_state.hands.process(image_rgb)

    gesture_text = ""
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
            gesto = detectar_gesto(landmarks)
            if gesto and gesto != last_gesture:
                gesture_text = gesto
                st.session_state.mqtt_client.publish(MQTT_TOPIC, gesto)
                last_gesture = gesto

    if gesture_text:
        cv2.putText(image, gesture_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    return av.VideoFrame.from_ndarray(image, format="bgr24")

# Streamlit UI
st.title("Detector de Gestos con MQTT 🖐✊👌")
webrtc_streamer(
    key="gesture",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": {"frameRate": 15}, "audio": False},
    async_processing=False,
)
