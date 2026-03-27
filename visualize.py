import cv2
import json
import argparse
import numpy as np
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize tracked players and networks.")
    parser.add_argument("--video", type=str, required=True, help="Input video")
    parser.add_argument("--tracking", type=str, default="track_sample.json", help="Tracking data JSON")
    parser.add_argument("--mapping", type=str, default="mapped_positions.json", help="Mapped positions JSON")
    parser.add_argument("--output", type=str, default="final_output.mp4", help="Output video")
    return parser.parse_args()

# BGR Colors
COLORS = {
    "GK": (0, 255, 255),    # Yellow
    "DF": (255, 0, 0),      # Blue
    "MF": (0, 255, 0),      # Green
    "FW": (0, 0, 255),      # Red
    "Ref/Other": (255, 255, 255) # White
}

def get_player_info(track_id, mapping_data):
    players = mapping_data.get("players", [])
    for p in players:
        if track_id in p.get("track_ids", []):
            return p.get("name", f"ID {track_id}"), p.get("position")
    return None, None

def main():
    args = parse_args()
    
    if not os.path.exists(args.tracking) or not os.path.exists(args.mapping):
        print("Missing tracking or mapping data.")
        return
        
    with open(args.tracking, "r") as f:
        tracking_data = json.load(f)
        
    with open(args.mapping, "r") as f:
        mapping_data = json.load(f)
        
    cap = cv2.VideoCapture(args.video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Attempting to use a mobile/web friendly codec just in case
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_key = f"frame_{frame_idx}"
        current_objects = tracking_data.get(frame_key, [])
        
        # Dictionary to store center points for network topology per position
        pos_centers = {pos: [] for pos in COLORS.keys()}
        
        for obj in current_objects:
            track_id = obj["track_id"]
            x1, y1, x2, y2 = map(int, obj["bbox"])
            
            # Bottom center for Ellipse
            center_x = (x1 + x2) // 2
            bottom_y = y2
            
            name, pos = get_player_info(track_id, mapping_data)
            
            if pos:
                color = COLORS[pos]
                
                # Draw Ellipse on the ground
                cv2.ellipse(frame, (center_x, bottom_y), (int((x2-x1)/2), int((x2-x1)*0.2)),
                            0, 0, 360, color, 2)
                cv2.ellipse(frame, (center_x, bottom_y), (int((x2-x1)/2), int((x2-x1)*0.2)),
                            0, 0, 360, color, -1)
                            
                # Draw Custom Name Badge
                (text_width, text_height), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
                
                # Use black text for yellow/white backgrounds specifically for contrast
                text_color = (0, 0, 0) if pos in ["GK", "Ref/Other"] else (255, 255, 255)
                cv2.putText(frame, name, (x1, y1 - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
                            
                # Collect centers for topology (using body center for shape drawing)
                body_center_y = (y1 + y2) // 2
                pos_centers[pos].append((center_x, body_center_y))
            else:
                # Unmapped
                cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 128, 128), 1)

        # Draw networks (Formation lines)
        for pos, centers in pos_centers.items():
            # Only draw lines for groups of 3 or more to avoid awkward 2-man connections (like LDM to RDM)
            if len(centers) >= 2 and pos != "Ref/Other":
                color = COLORS.get(pos, (255, 255, 255))
                
                # Sort by Y-axis (top to bottom of the screen) 
                # This naturally connects Far-side -> Center -> Near-side without crossing back!
                sorted_centers = sorted(centers, key=lambda pt: pt[1])
                pts = np.array(sorted_centers, np.int32)
                
                # isClosed=False guarantees LB never connects to RB, and LW never connects to RW
                cv2.polylines(frame, [pts], isClosed=False, color=color, thickness=2)
                    
        out.write(frame)
        frame_idx += 1
        
    cap.release()
    out.release()
    print(f"Saved visualization to {args.output}")

if __name__ == "__main__":
    main()
