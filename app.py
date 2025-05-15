import cv2
import numpy as np
import av
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer
import mediapipe as mp
import paho.mqtt.client as mqtt
import atexit

# Title
st.title("Detector de Gestos con MQTT 🖐✊👌")

# Initialize MediaPipe Hands with dynamic mode for video
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

if "hands" not in st.session_state:
    # Use static_image_mode=False for continuous video frames
    st.session_state.hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )

# MQTT setup
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
MQTT_TOPIC = "streamlit/gesto"
if "mqtt_client" not in st.session_state:
    client = mqtt.Client()
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    st.session_state.mqtt_client = client
    # Ensure MQTT disconnect on exit
    atexit.register(client.disconnect)

# Gesture detection

def detectar_gesto(landmarks):
    dedos_estirados = []
    # IDs of landmark tips
    # Thumb (x-axis), Others (y-axis)
    dedos_estirados.append(landmarks[4][0] > landmarks[3][0])
    for idx in [8, 12, 16, 20]:
        dedos_estirados.append(landmarks[idx][1] < landmarks[idx - 2][1])

    # Check for OK gesture by distance thumb-index
    dist = np.linalg.norm(
        np.array(landmarks[4][:2]) - np.array(landmarks[8][:2])
    )

    # Determine gesture
    if dedos_estirados == [False] * 5:
        return "Puño cerrado ✊"
    elif dedos_estirados == [True] * 5:
        return "Palma abierta 🖐"
    elif dist < 0.05 and all(dedos_estirados[i] for i in [2, 3, 4]):
        return "Gesto OK 👌"
    return None

# Callback for each video frame
frame_counter = 0
last_gesture = None

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    global frame_counter, last_gesture
    frame_counter += 1
    # Process every other frame to save resources
    if frame_counter % 2 != 0:
        return frame

    img = frame.to_ndarray(format="bgr24")
    img = cv2.resize(img, (320, 240))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = st.session_state.hands.process(img_rgb)
    gesture_text = None

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
            gesto = detectar_gesto(landmarks)
            if gesto and gesto != last_gesture:
                gesture_text = gesto
                st.session_state.mqtt_client.publish(MQTT_TOPIC, gesto)
                last_gesture = gesto

    if gesture_text:
        cv2.putText(
            img,
            gesture_text,
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            3
        )
    # Return modified frame
    return av.VideoFrame.from_ndarray(img, format="bgr24")

# WebRTC streamer with STUN config and async processing
webrtc_streamer(
    key="gesture",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": {"frameRate": 15}, "audio": False},
    async_processing=True,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
)
