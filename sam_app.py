import streamlit as st
import cv2
import json
import os
import shutil
from streamlit_image_coordinates import streamlit_image_coordinates
from ultralytics.models.sam import SAM2DynamicInteractivePredictor

st.set_page_config(layout="wide")

st.title("SAM 2 Human-in-the-Loop Tracker")
st.write("Annotate and track a specific player across the video using SAM2.1.")

# Extract frames to a temporary directory so we can iterate efficiently
@st.cache_data
def extract_frames(video_path, output_dir):
    if os.path.exists(output_dir):
        # We assume they are already extracted if the dir exists and contains files
        if len(os.listdir(output_dir)) > 0:
            return sorted([os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith('.jpg')])
        
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_paths = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        path = os.path.join(output_dir, f"frame_{idx:04d}.jpg")
        cv2.imwrite(path, frame)
        frame_paths.append(path)
        idx += 1
    cap.release()
    return sorted(frame_paths)

# User inputs
col1, col2, col3, col4 = st.columns(4)
video_path = col1.text_input("Video Path", "data/sample.mp4")
output_json = col2.text_input("Output JSON", "track_sample.json")
track_id_input = col3.number_input("Assign Track ID", min_value=1, value=99)
model_variant = col4.selectbox("SAM 2.1 Model", ["sam2.1_t.pt", "sam2.1_s.pt", "sam2.1_b.pt", "sam2.1_l.pt"], index=1)

if not os.path.exists(video_path):
    st.error(f"Video {video_path} not found.")
    st.stop()

frames_dir = os.path.join(os.path.dirname(video_path), "frames_cache")

# Extracted frame references
with st.spinner("Extracting frames or loading from cache..."):
    frame_paths = extract_frames(video_path, frames_dir)
num_frames = len(frame_paths)

if num_frames == 0:
    st.error("No frames extracted.")
    st.stop()

# Application state
if "predictor" not in st.session_state or st.session_state.get("model_variant") != model_variant:
    st.session_state.model_variant = model_variant
    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    overrides = dict(conf=0.1, task="segment", mode="predict", imgsz=1024, model=model_variant, save=False, device=device)
    # The default cache size or model loading params can be adjusted via overrides
    st.session_state.predictor = SAM2DynamicInteractivePredictor(overrides=overrides, max_obj_num=10)
    st.session_state.tracking_data = {}
    st.session_state.last_click = None
    st.session_state.last_clicked_frame = None

# UI for scrubbing through frames
col_scrub, col_actions = st.columns([3, 1])
current_frame_idx = col_scrub.slider("Select Frame", 0, max(0, num_frames - 1), 0)
current_frame_path = frame_paths[current_frame_idx]

st.write(f"Click directly on the player in Frame {current_frame_idx} below to refine the SAM 2 tracking memory.")

# Fetch coordinate clicks natively natively via library component
value = streamlit_image_coordinates(current_frame_path, key=f"img_{current_frame_idx}")

# Handle click event
if value is not None and (value != st.session_state.get("last_click") or current_frame_idx != st.session_state.get("last_clicked_frame")):
    st.session_state.last_click = value
    st.session_state.last_clicked_frame = current_frame_idx
    x, y = value["x"], value["y"]
    
    with st.spinner(f"Updating memory with prompt at ({x}, {y}) for track ID {track_id_input}..."):
        results = st.session_state.predictor(
            source=current_frame_path, 
            points=[[x, y]], 
            labels=[1], 
            obj_ids=[0], # Internal SAM2 memory uses max_obj_num=10, so we just use 0 internally 
            update_memory=True
        )
        
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            box = boxes.xyxy[0].cpu().tolist()
            st.session_state.tracking_data[f"frame_{current_frame_idx}"] = [int(b) for b in box]
            st.success(f"Added positive point at ({x},{y}).")
        else:
            st.warning("SAM 2 could not find an object bounding box around this point.")

with col_actions:
    if st.button("▶ Track Forward From Here", type="primary"):
        with st.spinner("Tracking forward..."):
            progress_bar = st.progress(0)
            
            track_end = num_frames
            for i in range(current_frame_idx + 1, track_end):
                res = st.session_state.predictor(source=frame_paths[i])
                
                boxes = res[0].boxes
                if boxes is not None and len(boxes) > 0:
                    try:
                        box = boxes.xyxy[0].cpu().tolist()
                        st.session_state.tracking_data[f"frame_{i}"] = [int(b) for b in box]
                    except Exception:
                        pass
                
                progress_bar.progress((i - current_frame_idx) / (track_end - current_frame_idx - 1))
                
            progress_bar.empty()
            st.success("Forward tracking completed.")

    if st.button("💾 Replace JSON Output"):
        with st.spinner("Writing to JSON..."):
            final_json = {}
            for f_idx in range(num_frames):
                fk = f"frame_{f_idx}"
                if fk in st.session_state.tracking_data:
                    final_json[fk] = [{
                        "track_id": track_id_input,
                        "bbox": st.session_state.tracking_data[fk]
                    }]
                else:
                    final_json[fk] = []
                    
            with open(output_json, "w") as f:
                json.dump(final_json, f, indent=4)
            st.success(f"Successfully replaced mapped tracking points to {output_json}!")

st.divider()

# Result visualization logic within the GUI
st.subheader("Tracking Data Preview")
col_img, col_info = st.columns([3, 1])

with col_img:
    if f"frame_{current_frame_idx}" in st.session_state.tracking_data:
        box = st.session_state.tracking_data[f"frame_{current_frame_idx}"]
        img_bgr = cv2.imread(current_frame_path)
        # Draw bounding box and ID label onto the image
        cv2.rectangle(img_bgr, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 3)
        cv2.putText(img_bgr, f"ID {track_id_input}", (int(box[0]), max(0, int(box[1]) - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        st.image(img_rgb, caption=f"Tracked Output for Frame {current_frame_idx}")
    else:
        st.write("No tracking data available for this frame.")

with col_info:
    frames_tracked = len(st.session_state.tracking_data)
    st.metric("Total Frames Tracked", f"{frames_tracked} / {num_frames}")
