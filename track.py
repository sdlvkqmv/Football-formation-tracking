import argparse
import json
import cv2
from ultralytics import YOLO
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Track players in football video using YOLO26")
    parser.add_argument("--source", type=str, required=True, help="Path to input video")
    parser.add_argument("--model", type=str, default="yolo26x.pt", help="Path to YOLO26 model")
    parser.add_argument("--tracker", type=str, default="botsort.yaml", help="Tracker config")
    parser.add_argument("--output", type=str, default="track_sample.json", help="Output JSON path")
    parser.add_argument("--device", type=str, default="mps", help="Device to run on (e.g., mps, cuda, cpu)")
    parser.add_argument("--out_video", type=str, default="track_output.mp4", help="Output tracked video path")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    return parser.parse_args()

def main():
    args = parse_args()
    model = YOLO(args.model)
    
    # Get FPS
    cap = cv2.VideoCapture(args.source)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps == 0 or fps != fps:
        fps = 30.0
        
    print(f"Starting tracking on {args.source} using {args.model} with {args.tracker}")
    results = model.track(
        source=args.source,
        tracker=args.tracker,
        classes=[0], # 0 is usually person
        stream=True,
        device=args.device,
        imgsz=args.imgsz
    )
    
    tracking_data = {}
    video_writer = None
    
    for frame_idx, result in enumerate(results):
        boxes = result.boxes
        plotted_frame = result.orig_img.copy()
        
        if boxes is not None and boxes.id is not None:
            track_ids = boxes.id.int().cpu().tolist()
            coords = boxes.xyxy.cpu().tolist()
            
            tracking_data[f"frame_{frame_idx}"] = []
            for i, track_id in enumerate(track_ids):
                x1, y1, x2, y2 = map(int, coords[i])
                tracking_data[f"frame_{frame_idx}"].append({
                    "track_id": track_id,
                    "bbox": [x1, y1, x2, y2]
                })
                
                color = (int((track_id * 80) % 255), int((track_id * 150) % 255), int((track_id * 220) % 255))
                cv2.rectangle(plotted_frame, (x1, y1), (x2, y2), color, 2)
                
                label = str(track_id)
                (w_txt, h_txt), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(plotted_frame, (x1, max(0, y1 - h_txt - 8)), (x1 + w_txt + 6, y1), color, -1)
                cv2.putText(plotted_frame, label, (x1 + 3, max(0, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if video_writer is None:
            h, w = plotted_frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            video_writer = cv2.VideoWriter(args.out_video, fourcc, fps, (w, h))
            
        video_writer.write(plotted_frame)
        
    with open(args.output, "w") as f:
        json.dump(tracking_data, f, indent=4)
        
    if video_writer is not None:
        video_writer.release()
        
    print(f"Tracking data saved to {args.output}")
    print(f"Tracked video saved to {args.out_video}")

if __name__ == "__main__":
    main()
