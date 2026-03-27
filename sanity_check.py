import cv2
import json
import argparse
import os
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Sanity check tracking data.")
    parser.add_argument("--video", type=str, required=True, help="Input video path")
    parser.add_argument("--tracking", type=str, default="track_sample.json", help="Tracking JSON path")
    parser.add_argument("--id", type=int, required=False, default=None, help="Track ID to visualize (omit to visualize all)")
    parser.add_argument("--output", type=str, default="sanity_check_output.mp4", help="Output video path")
    return parser.parse_args()

def main():
    args = parse_args()
    
    if not os.path.exists(args.tracking):
        print(f"Tracking data not found at {args.tracking}")
        return
        
    with open(args.tracking, "r") as f:
        tracking_data = json.load(f)
        
    cap = cv2.VideoCapture(args.video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    
    desc_label = f"Visualizing ID {args.id}" if args.id is not None else "Visualizing All IDs"
    
    frame_idx = 0
    with tqdm(total=total_frames, desc=desc_label) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_key = f"frame_{frame_idx}"
            current_objects = tracking_data.get(frame_key, [])
            
            # Find objects to draw
            if args.id is not None:
                objects_to_draw = [obj for obj in current_objects if obj["track_id"] == args.id]
            else:
                objects_to_draw = current_objects
                
            for obj in objects_to_draw:
                x1, y1, x2, y2 = map(int, obj["bbox"])
                tid = obj["track_id"]
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
                # Draw ID badge
                label = f"ID {tid}"
                (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, max(0, y1 - text_height - 10)), (x1 + text_width, y1), (0, 0, 255), -1)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
            out.write(frame)
            frame_idx += 1
            pbar.update(1)
            
    cap.release()
    out.release()
    print(f"Sanity check video saved to {args.output}")

if __name__ == "__main__":
    main()
