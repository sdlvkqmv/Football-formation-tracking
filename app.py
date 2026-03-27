import streamlit as st
import json
import os
import cv2

st.set_page_config(layout="wide")

def load_data(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_unique_ids(tracking_data):
    unique_ids = set()
    for frame, objects in tracking_data.items():
        for obj in objects:
            unique_ids.add(obj["track_id"])
    return sorted(list(unique_ids))

def get_representative_boxes(tracking_data):
    rep_boxes = {}
    for frame_key, objects in tracking_data.items():
        frame_idx = int(frame_key.split('_')[1])
        for obj in objects:
            tid = obj["track_id"]
            if tid not in rep_boxes:
                rep_boxes[tid] = {"frame_idx": frame_idx, "bbox": obj["bbox"]}
    return rep_boxes


st.title("Football Position Mapping & AI Aliasing App")
st.write("Review the tracked video, inspect the IDs, and group different IDs into a specific Player Identity.")

col_path1, col_path2, col_path3, col_path4 = st.columns(4)
with col_path1:
    tracking_data_path = st.text_input("Tracking Data JSON", "track_sample.json")
with col_path2:
    video_path = st.text_input("Raw Video Path", "data/sample.mp4")
with col_path3:
    tracked_video_path = st.text_input("Tracked Output Video", "track_output.mp4")
with col_path4:
    mapping_data_path = st.text_input("Mapping Output JSON", "mapped_positions.json")

# Initialize session state for players
if "players" not in st.session_state:
    st.session_state.players = []
    # Try loading existing
    existing = load_data(mapping_data_path)
    if existing and "players" in existing:
        st.session_state.players = existing["players"]

# Play the tracked video
if os.path.exists(tracked_video_path):
    with st.expander("Watch Tracked Video", expanded=True):
        st.video(tracked_video_path)

tracking_data = load_data(tracking_data_path)

if tracking_data is None:
    st.error(f"Could not find {tracking_data_path}.")
else:
    unique_ids = extract_unique_ids(tracking_data)
    rep_boxes = get_representative_boxes(tracking_data)
    
    st.success(f"Loaded {len(unique_ids)} unique tracked IDs.")
    
    col1, col2 = st.columns([1.2, 1])
    
    with col1:
        st.subheader("📋 Define Players (ID Aliasing)")
        
        # Determine which IDs have already been mapped so we can filter or warn
        mapped_ids = []
        for p in st.session_state.players:
            mapped_ids.extend(p["track_ids"])
            
        available_ids = [uid for uid in unique_ids if uid not in mapped_ids]
        
        with st.form("create_player_form"):
            p_name = st.text_input("Player Name (e.g. 'Goalkeeper A')")
            p_ids = st.multiselect("Select Track IDs (You can choose multiple if tracker lost them)", unique_ids, default=None)
            p_pos = st.selectbox("Position", ["GK", "DF", "MF", "FW", "Ref/Other"])
            submitted = st.form_submit_button("Add to Roster")
            
            if submitted and p_name != "" and len(p_ids) > 0:
                st.session_state.players.append({
                    "name": p_name,
                    "track_ids": p_ids,
                    "position": p_pos
                })
                st.rerun()

        st.markdown("### Current Team Roster")
        for i, p in enumerate(st.session_state.players):
            with st.container():
                c1, c2, c3, c4 = st.columns([2, 1, 3, 1])
                c1.write(f"**{p['name']}**")
                c2.write(f"_{p['position']}_")
                c3.write(f"IDs: {p['track_ids']}")
                if c4.button("Remove", key=f"del_{i}"):
                    st.session_state.players.pop(i)
                    st.rerun()
                    
        if st.button("💾 Save Pipeline Roster", type="primary"):
            with open(mapping_data_path, "w") as f:
                json.dump({"players": st.session_state.players}, f, indent=4)
            st.success(f"Saved {len(st.session_state.players)} players to {mapping_data_path}!")

    with col2:
        st.subheader("🔍 ID Inspector")
        st.info("Unsure who an ID belongs to? Preview them below.")
        selected_id_to_view = st.selectbox("Select ID", ["None"] + [str(u) for u in unique_ids])
        
        if selected_id_to_view != "None":
            tid = int(selected_id_to_view)
            if tid in rep_boxes:
                info = rep_boxes[tid]
                
                if not os.path.exists(video_path):
                    st.error(f"Video file not found at {video_path}")
                else:
                    cap = cv2.VideoCapture(video_path)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, info["frame_idx"])
                    ret, frame = cap.read()
                    cap.release()
                    
                    if ret:
                        x1, y1, x2, y2 = map(int, info["bbox"])
                        
                        drawn_frame = frame.copy()
                        cv2.rectangle(drawn_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        cv2.putText(drawn_frame, f"ID {tid}", (x1, max(0, y1 - 10)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                        drawn_frame = cv2.cvtColor(drawn_frame, cv2.COLOR_BGR2RGB)
                        
                        h, w = frame.shape[:2]
                        cx1 = max(0, x1 - 20)
                        cy1 = max(0, y1 - 20)
                        cx2 = min(w, x2 + 20)
                        cy2 = min(h, y2 + 20)
                        
                        crop = frame[cy1:cy2, cx1:cx2]
                        if crop.size > 0:
                            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                            st.image(crop, caption=f"Cropped ID {tid}")
                        
                        st.image(drawn_frame, caption=f"Full Context (Frame {info['frame_idx']})")
                    else:
                        st.error("Could not read frame from video.")
