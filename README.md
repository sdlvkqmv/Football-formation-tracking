# Football Position Tracking & Visualization Pipeline

A fully-featured pipeline to track football players using YOLO, intelligently alias and group fragmented tracking IDs into custom "Player Identities" via a Streamlit UI, and generate a final visualization video featuring elegant labels and open formation networking.

## Environment Setup
Make sure you have created the Conda environment and installed dependencies:

```bash
conda create -n football_tracking python=3.11 -y
conda activate football_tracking
pip install -r requirements.txt
```

## How to Run the Pipeline

### Step 1: Track Players in Video
Run the tracker on your input video. This uses BoT-SORT to assign IDs across frames. It will output both a `tracking_data.json` file and a new `track_output.mp4` video with clean, custom-sized ID labels on every player so you can quickly review the tracks without YOLO clutter.

```bash
python track.py --source /path/to/your/football_video.mp4 --output tracking_data.json --out_video track_output.mp4
```

*(By default, it uses `yolo26x.pt` which will download the weights automatically via Ultralytics if available.)*

### Step 1.5: (Optional) Sanity Check Tracker IDs
If you want to isolate and verify specific tracking IDs, run the sanity check utility:

```bash
# Preview all IDs to map who is who:
python sanity_check.py --video /path/to/your/football_video.mp4 --tracking tracking_data.json --output all_ids_sanity.mp4

# Or target exactly one ID (e.g. ID 5) to see if the tracker held onto them:
python sanity_check.py --video /path/to/your/football_video.mp4 --tracking tracking_data.json --id 5 --output id5_sanity.mp4
```

### Step 2: Create Player Identities & Aliasing (Streamlit UI)
Trackers frequently drop and create new IDs for the same physical person. Run the Streamlit app to watch the tracked video and group multiple IDs into custom "Player Profiles".

```bash
streamlit run app.py
```
- Play the **Tracked Video** natively inside your browser.
- Use the **ID Inspector** on the right to isolate and crop exactly where a specific ID appeared so you can verify their identity.
- Under **Player Roster**, assign a Custom Name (e.g. "Son Heung-min"), select all relevant Track IDs for that person from the multi-select dropdown, and assign a Position. 
- Click **Save Pipeline Roster** to seamlessly output your `mapped_positions.json`.

### Step 3: Generate Final Visualization
Now that we have tracking data and your custom player aliases, render the final broadcast-style video:

```bash
python visualize.py \
    --video /path/to/your/football_video.mp4 \
    --tracking tracking_data.json \
    --mapping mapped_positions.json \
    --output final_output.mp4
```

## Visualization Details
- **Overhead Labels:** Uses your beautifully aliased Player Names with bounding ellipses colored dynamically by position.
- **Formation Networking:** Automatically sorts players in the same position group by pitch width and draws **open formation lines** connecting them (e.g., LB -> LCB -> RCB -> RB). It strategically prevents closed-loop connections, ensuring opposite sides of the pitch (like Left vs Right) don't mistakenly connect across the center. 
- **Colors:**
  - Goalkeeper (`GK`): Yellow
  - Defender (`DF`): Blue
  - Midfielder (`MF`): Green
  - Forward (`FW`): Red
  - Ref/Other: White (Excluded from network drawing)
