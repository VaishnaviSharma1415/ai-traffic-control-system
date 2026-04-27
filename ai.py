import streamlit as st
import cv2
import numpy as np
import tempfile
from ultralytics import YOLO

# --- Page Configuration ---
st.set_page_config(page_title="AI Traffic Control", page_icon="🚦", layout="wide")

# --- Load YOLOv8 Model ---
model = YOLO("yolov8n.pt")

# --- Vehicle Classes (car, motorcycle, bus, truck) ---
vehicle_classes = [2, 3, 5, 7]

# --- Count Vehicles ---
def count_vehicles(results):
    count = 0
    for r in results:
        for c in r.boxes.cls:
            if int(c) in vehicle_classes:
                count += 1
    return count

# --- Calculate Green Light Time ---
def calculate_green_light_time(vehicle_count):
    if vehicle_count == 0:
        return 0
    time_per_vehicle = 3
    min_time = 15
    max_time = 90
    return min(max(vehicle_count * time_per_vehicle, min_time), max_time)

# --- Sidebar ---
with st.sidebar:
    st.title("🚦 Smart Traffic Control")
    st.markdown("An AI system to detect vehicles and recommend green light duration.")
    input_mode = st.radio("Select Input Mode", ["Image", "Video", "Live Camera"])
    uploaded_file = None
    if input_mode in ["Image", "Video"]:
        uploaded_file = st.file_uploader("📤 Upload File", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])
    st.markdown("---")
    st.caption("Powered by YOLOv8 🚀 and Streamlit ❤")

# --- Main UI ---
st.title("🧠 AI-Powered Traffic Signal Optimization")
st.subheader("Real-time vehicle detection and green light time recommendation")

# --- Image Mode ---
if input_mode == "Image" and uploaded_file:
    st.info("📷 Processing Image...")
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="📥 Uploaded Image", channels="BGR", use_container_width=True)

    with st.spinner("Detecting vehicles..."):
        results = model(image)
        vehicle_count = count_vehicles(results)
        green_time = calculate_green_light_time(vehicle_count)
        result_img = results[0].plot()

    with col2:
        st.image(result_img, caption="📤 Detection Result", channels="BGR", use_container_width=True)

    if vehicle_count == 0:
        st.warning("🚫 No vehicles detected.")
    else:
        st.success(f"🚗 Vehicles Detected: {vehicle_count}")
        st.info(f"🟢 Recommended Green Light Time: {green_time} seconds")

# --- Video Mode ---
elif input_mode == "Video" and uploaded_file:
    st.info("🎥 Processing Video...")
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)

    stframe = st.empty()
    vehicle_total = 0
    frame_count = 0

    with st.spinner("Running vehicle detection on video..."):
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            vehicle_count = count_vehicles(results)
            vehicle_total += vehicle_count
            frame_count += 1

            result_frame = results[0].plot()
            stframe.image(result_frame, channels="BGR", use_container_width=True)

    cap.release()

    avg_vehicle_count = vehicle_total // max(frame_count, 1)
    green_time = calculate_green_light_time(avg_vehicle_count)

    if avg_vehicle_count == 0:
        st.warning("🚫 No vehicles detected in video.")
    else:
        st.success(f"📊 Avg Vehicles per Frame: {avg_vehicle_count}")
        st.info(f"🟢 Recommended Green Light Time: {green_time} seconds")

# --- Live Camera Mode ---
elif input_mode == "Live Camera":
    st.info("📡 Live Camera Detection - Turn on webcam and click below to start.")
    run_live = st.checkbox("📍 Start Live Detection")

    if run_live:
        cap = cv2.VideoCapture(0)
        stframe = st.empty()

        vehicle_total = 0
        frame_count = 0

        with st.spinner("Starting live vehicle detection... Press Stop to end."):
            while run_live:
                ret, frame = cap.read()
                if not ret:
                    st.error("❌ Failed to access camera.")
                    break

                results = model(frame)
                vehicle_count = count_vehicles(results)
                vehicle_total += vehicle_count
                frame_count += 1

                result_frame = results[0].plot()
                stframe.image(result_frame, channels="BGR", use_container_width=True)

                st.markdown(f"🚗 Vehicles in Current Frame: {vehicle_count}")
                st.markdown(f"🟢 Green Time (Current): {calculate_green_light_time(vehicle_count)} seconds")

        cap.release()