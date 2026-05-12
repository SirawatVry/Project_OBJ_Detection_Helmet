import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tempfile
import os
import csv
from datetime import datetime
from ultralytics import YOLO
from norfair import Tracker, Detection
from collections import defaultdict
from media_pipeline import FalsePositiveReducer, create_reducer, VideoAnalyzer

st.markdown("""
<style>

/* ===== Background ===== */
.main {
    background-color: #0F172A;
}

/* spacing */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* ===== Text ===== */
h1, h2, h3 {
    color: #E5E7EB;
}

p, span, label {
    color: #CBD5F5;
}

/* ===== Buttons ===== */
div.stButton > button {
    background-color: #3B82F6;
    color: white;
    border-radius: 10px;
    border: none;
    height: 45px;
    font-weight: 600;
}

div.stButton > button:hover {
    background-color: #2563EB;
}

/* ===== Metric Cards ===== */
div[data-testid="metric-container"] {
    background: #1E293B;
    border-radius: 14px;
    padding: 15px;
    border: 1px solid #334155;
}

/* ===== Dataframe ===== */
div[data-testid="stDataFrame"] {
    border-radius: 12px;
    border: 1px solid #334155;
}

/* ===== Images ===== */
img {
    border-radius: 12px;
    border: 1px solid #334155;
}

/* ===== Sidebar ===== */
section[data-testid="stSidebar"] {
    background-color: #020617;
    border-right: 1px solid #334155;
}

/* ===== Progress bar ===== */
div[data-testid="stProgressBar"] > div > div {
    background-color: #3B82F6;
}

/* ===== File uploader ===== */
section[data-testid="stFileUploader"] {
    background-color: #1E293B;
    border-radius: 12px;
    padding: 15px;
    border: 1px dashed #3B82F6;
}

/* ===== Divider ===== */
hr {
    border-color: #334155;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------
# 1. CONFIG (อ้างอิงจาก norfair_points.py)
# ---------------------------
MODEL_PATH = "../all_model_compare/finetune.pt"
VIOLATION_DIR = "./violations/no_helmet_captures"
LOG_FILE = "./violations/no_helmet_log.csv"

CONF_THRES = 0.4
IOU_THRES = 0.5
VOTING_WINDOW = 5  # Reduced from 10 for faster convergence
VOTING_THRESHOLD = 0.7
CAPTURE_INTERVAL = 200
# Detection Thresholds
st.subheader("🔧 Threshold Settings")
PIPELINE_CONFIG = {
    'min_brightness': 30.0,
    'max_brightness': 220.0,
    'min_blur_threshold': 100.0,
    'confidence_thresholds': {0: 0.5, 1: 0.4, 2: 0.65},
    'min_detection_area': 100.0,
    'max_area_ratio': 0.8,
    'voting_threshold': 0.7,
    'min_track_age': 3
}
min_brightness = st.slider(
        "Min Brightness",
        min_value=0,
        max_value=100,
        value=int(PIPELINE_CONFIG['min_brightness']),
        help="Frames darker than this are skipped"
    )
    
max_brightness = st.slider(
        "Max Brightness",
        min_value=150,
        max_value=255,
        value=int(PIPELINE_CONFIG['max_brightness']),
        help="Frames brighter than this are skipped"
    )
    
min_blur = st.slider(
        "Min Blur Threshold",
        min_value=0,
        max_value=300,
        value=int(PIPELINE_CONFIG['min_blur_threshold']),
        help="Frames blurrier than this are skipped (Laplacian variance)"
    )
    
no_helmet_conf = st.slider(
        "No-Helmet Confidence",
        min_value=0.1,
        max_value=1.0,
        value=PIPELINE_CONFIG['confidence_thresholds'][2],
        step=0.05,
        help="Minimum confidence to classify as no-helmet violation"
    )
# Media Pipeline Configuration
PIPELINE_PRESET = 'balanced'  # Options: 'strict', 'balanced', 'lenient'
ENABLE_PREPROCESSING = False  # ✅ Toggle preprocessing - ปิดเป็นค่าเริ่มต้นเพื่อความเร็ว
ENABLE_FRAME_FILTERING = False  # ✅ Toggle frame quality filtering
FRAME_SKIP_INTERVAL = 1  # ✅ Process every N frames (1 = all frames, 2 = half, 3 = third)
VIDEO_SCALE_FACTOR = 1.0  # ✅ Scale video 1.0=full, 0.75=75%, 0.5=50% (ลดเวลา)
UPDATE_UI_INTERVAL = 1  # ✅ Update UI every N frames (1=all, 3=every 3rd frame)

PIPELINE_CONFIG = {
    'min_brightness': 30.0,
    'max_brightness': 220.0,
    'min_blur_threshold': 100.0,
    'confidence_thresholds': {0: 0.5, 1: 0.4, 2: 0.65},  # Helmet, Motorcycle, No Helmet
    'min_detection_area': 100.0,
    'max_area_ratio': 0.8,
    'voting_threshold': 0.7,
    'min_track_age': 3
}

# กำหนดชื่อคอลัมน์มาตรฐาน
HEADER_NAMES = ['timestamp', 'frame_number', 'track_id', 'confidence', 'x1', 'y1', 'x2', 'y2', 'image_filename', 'vote_score', 'stability_score']

CLASS_NAMES = ['helmet', 'motorcycle', 'no_helmet']
COLORS = {
    0: (0, 255, 0),      # Helmet - Green
    1: (255, 0, 0),      # Motorcycle - Blue
    2: (0, 0, 255),      # No helmet - Red
}

if not os.path.exists(VIOLATION_DIR):
    os.makedirs(VIOLATION_DIR)

# ---------------------------
# 2. CORE FUNCTIONS WITH MEDIA PIPELINE
# ---------------------------

def process_video_with_pipeline(video_path, model, pipeline_config=None, enable_preprocessing=True, 
                               frame_skip=1, video_scale=1.0, ui_update_interval=1, enable_filtering=True):
    """
    Process video with media pipeline for false positive reduction
    
    Parameters:
    - enable_preprocessing: Apply denoise + contrast enhancement (slower)
    - frame_skip: Process every N frames (1=all, 2=half, 3=third)
    - video_scale: Scale video (0.5=50%, 0.75=75%, 1.0=full)
    - ui_update_interval: Update UI every N frames
    - enable_filtering: Apply detection filtering
    """
    if pipeline_config is None:
        pipeline_config = PIPELINE_CONFIG
    
    # Initialize media pipeline
    reducer = FalsePositiveReducer(**pipeline_config)
    analyzer = VideoAnalyzer(reducer)
    
    cap = cv2.VideoCapture(video_path)
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Apply video scaling
    width = int(orig_width * video_scale)
    height = int(orig_height * video_scale)
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    tracker = Tracker(
        distance_function="euclidean",
        distance_threshold=250,
        initialization_delay=2,
        hit_counter_max=5,
        past_detections_length=7
    )

    last_capture_time = {}
    active_track_ids = set()
    motorcycle_ids = set()
    violation_ids = set()
    frame_stats = defaultdict(int)

    frame_placeholder = st.empty()
    progress_bar = st.progress(0)
    frame_count = 0
    processed_frame_count = 0
    
    # Performance tracking
    import time
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        
        # 🚀 Frame Skipping - ข้าม frame ตามที่ตั้ง
        if frame_count % frame_skip != 0:
            progress_bar.progress(frame_count / total_frames if total_frames > 0 else 0)
            continue
        
        processed_frame_count += 1
        
        # 🚀 Video Scaling - ลดขนาด frame เพื่อความเร็ว
        if video_scale != 1.0:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)

        # 🚀 Preprocessing - ปิด/เปิดได้
        if enable_preprocessing:
            processed_frame, quality_metrics = analyzer.analyze_frame(frame)
            # Skip frames with poor quality (optional - can be disabled)
            if quality_metrics['blur_score'] < reducer.min_blur_threshold:
                progress_bar.progress(frame_count / total_frames if total_frames > 0 else 0)
                continue
        else:
            processed_frame = frame

        # Run inference on processed frame
        results = model.predict(
            processed_frame, 
            conf=CONF_THRES, 
            iou=IOU_THRES, 
            verbose=False, 
            agnostic_nms=True
        )[0]

        detections = []
        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy().astype(int)
            
            for box, conf, cls in zip(boxes, confs, classes):
                # 🚀 Detection Filtering - ปิด/เปิดได้
                if enable_filtering:
                    is_valid, filter_results = reducer.filter_detection(
                        box, conf, cls, width, height
                    )
                    if not is_valid:
                        frame_stats['filtered_detections'] += 1
                        continue
                
                # Add to detection list
                cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                detections.append(Detection(
                    points=np.array([[cx, cy]]),
                    scores=np.array([conf]),
                    data={
                        'box': box,
                        'conf': conf,
                        'cls': cls,
                        'visibility': 1.0
                    }
                ))

        # Update tracker
        tracked_objects = tracker.update(detections=detections)
        
        current_ids = {obj.id for obj in tracked_objects}
        for tid in (active_track_ids - current_ids):
            if enable_filtering:
                reducer.cleanup_track(tid)
            for d in [last_capture_time]:
                if tid in d:
                    del d[tid]
        active_track_ids = current_ids

        # Process tracked objects
        for obj in tracked_objects:
            if obj.last_detection is None or obj.age < reducer.min_track_age:
                continue
            
            tid = obj.id
            box = obj.last_detection.data['box']
            conf = obj.last_detection.data['conf']
            cls = obj.last_detection.data['cls']
            
            # Apply temporal smoothing
            final_cls, vote_score, is_confident = reducer.apply_temporal_smoothing(
                tid, cls, conf, box, obj.age
            )
            
            # Get stability score
            stability_score = 1.0 - reducer.temporal_consistency.get_position_stability(tid)

            x1, y1, x2, y2 = map(int, box)
            x1_c = max(0, min(x1, width - 1))
            y1_c = max(0, min(y1, height - 1))
            x2_c = max(0, min(x2, width - 1))
            y2_c = max(0, min(y2, height - 1))
            
            visible_ratio = ((x2_c - x1_c) * (y2_c - y1_c)) / max(1, (x2 - x1) * (y2 - y1))
            if visible_ratio < 0.2:
                continue

            # Determine visualization
            if final_cls == 1:
                motorcycle_ids.add(tid)
            
            # Log violations - now with confidence check
            if final_cls == 2 and is_confident:
                violation_ids.add(tid)
                
                if tid not in last_capture_time or (processed_frame_count - last_capture_time[tid] >= CAPTURE_INTERVAL):
                    timestamp = datetime.now()
                    filename = f"no_helmet_{tid}_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"
                    crop = frame[y1_c:y2_c, x1_c:x2_c]
                    if crop.size > 0:
                        cv2.imwrite(os.path.join(VIOLATION_DIR, filename), crop)
                        
                        file_exists = os.path.isfile(LOG_FILE)
                        with open(LOG_FILE, 'a', newline='') as f:
                            writer = csv.writer(f)
                            if not file_exists:
                                writer.writerow(HEADER_NAMES)
                            writer.writerow([
                                timestamp.isoformat(), 
                                frame_count, 
                                tid, 
                                f"{conf:.2f}", 
                                x1, y1, x2, y2, 
                                filename, 
                                f"{vote_score:.2f}",
                                f"{stability_score:.2f}"
                            ])
                    last_capture_time[tid] = processed_frame_count
                    frame_stats['violations_logged'] += 1

            # Draw on original frame
            color = COLORS.get(final_cls, (255, 255, 255))
            cv2.rectangle(frame, (x1_c, y1_c), (x2_c, y2_c), color, 3)
            
            # Enhanced label with confidence info
            label = f"ID:{tid} {CLASS_NAMES[final_cls]} ({vote_score:.0%})"
            if final_cls == 2:
                label += f" [Stable: {stability_score:.0%}]"
            
            cv2.putText(frame, label, (x1_c, y1_c - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 🚀 UI Update Interval - ลด update ความถี่
        if processed_frame_count % ui_update_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB")
        
        frame_count += 1
        progress_bar.progress(frame_count / total_frames if total_frames > 0 else 0)

    cap.release()
    
    # Get analysis statistics
    elapsed_time = time.time() - start_time
    actual_fps = processed_frame_count / elapsed_time if elapsed_time > 0 else 0
    
    return {
        'motorcycles': len(motorcycle_ids),
        'violations': len(violation_ids),
        'frame_stats': frame_stats,
        'total_frames': frame_count,
        'processed_frames': processed_frame_count,
        'elapsed_time': elapsed_time,
        'actual_fps': actual_fps
    }

# ---------------------------
# 3. STREAMLIT UI
# ---------------------------
st.set_page_config(page_title="Helmet Analytics", layout="wide")
st.title("🛡️ AI Helmet Violation Monitoring")

@st.cache_resource
def get_model(): 
    return YOLO(MODEL_PATH)

model = get_model()

# Sidebar for pipeline configuration
with st.sidebar:
    st.header("⚙️ Performance & Pipeline Settings")
    
    # Performance Optimization Controls
    st.subheader("🚀 Real-Time Performance")
    
    enable_preprocessing = st.checkbox(
        "Enable Frame Preprocessing",
        value=False,
        help="Denoise + Contrast enhancement (slower ~25ms/frame, better quality)"
    )
    
    frame_skip = st.slider(
        "Frame Skip Interval",
        min_value=1,
        max_value=5,
        value=1,
        help="1=process all frames (30fps), 2=skip half (15fps), 3=skip 2/3 (10fps)"
    )
    
    video_scale = st.slider(
        "Video Scale Factor",
        min_value=0.25,
        max_value=1.0,
        value=1.0,
        step=0.25,
        help="0.25=25% size (fastest), 1.0=full size (best quality)"
    )
    
    ui_update_interval = st.slider(
        "UI Update Interval",
        min_value=1,
        max_value=10,
        value=1,
        help="1=update every frame, 3=update every 3rd frame (faster)"
    )
    
    enable_filtering = st.checkbox(
        "Enable Detection Filtering",
        value=False,
        help="Quality check on detections (slower ~10ms, fewer false positives)"
    )
    
    st.divider()
    
    # Pipeline Configuration
    st.subheader("🎯 Detection Settings")
    
    preset = st.radio(
        "Pipeline Preset",
        options=['strict', 'balanced', 'lenient'],
        help="Strict=fewer false positives, Lenient=more detections"
    )
    
    st.divider()
    st.info("💡 **Recommended for Real-Time:**\n- Frame Skip: 2-3\n- Video Scale: 0.5-0.75\n- UI Update: 2-3\n- Preprocessing: OFF")

# Main content area
tab1, tab2 = st.tabs(["🎬 Video Analysis", "📊 Dashboard"])

with tab1:
    uploaded_file = st.file_uploader("Upload Video", type=['mp4', 'avi', 'mov', 'mkv'])
    
    if uploaded_file:
        # Update pipeline config based on sidebar settings
        current_config = PIPELINE_CONFIG.copy()
        current_config['min_brightness'] = float(min_brightness)
        current_config['max_brightness'] = float(max_brightness)
        current_config['min_blur_threshold'] = float(min_blur)
        current_config['confidence_thresholds'][2] = no_helmet_conf
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            if st.button("▶️ Run Detection", use_container_width=True, key="run_btn"):
                with st.spinner("Processing video..."):
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                    tfile.write(uploaded_file.read())
                    tfile.close()
                    
                    results = process_video_with_pipeline(tfile.name, model, current_config)
                    os.unlink(tfile.name)
                    
                    st.success("✅ Analysis Finished!")
                    
                    # Display metrics
                    m1, m2, m3, m4 = st.columns(4)
                    with m1:
                        st.metric("🏍️ Motorcycles", results['motorcycles'])
                    with m2:
                        st.metric("⚠️ Violations", results['violations'])
                    with m3:
                        rate = (results['violations'] / results['motorcycles'] * 100) if results['motorcycles'] > 0 else 0
                        st.metric("📈 Violation Rate", f"{rate:.1f}%")
                    with m4:
                        st.metric("📹 Frames", results['total_frames'])
                    
                    # Pipeline statistics
                    st.divider()
                    st.subheader("📊 Pipeline Statistics")
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Filtered Detections", results['frame_stats']['filtered_detections'])
                        st.metric("Violations Logged", results['frame_stats']['violations_logged'])
                    
                    with col_b:
                        p_stats = results['pipeline_stats']
                        st.metric("Blur Issues", f"{p_stats['blur_issue_rate']:.1f}%")
                        st.metric("Brightness Issues", f"{p_stats['brightness_issue_rate']:.1f}%")

with tab2:
    st.subheader("📈 Analytics & Trends")
    
    if os.path.exists(LOG_FILE):
        try:
            df = pd.read_csv(LOG_FILE, on_bad_lines='skip')
            if not df.empty:
                # Ensure proper column handling
                if len(df.columns) >= len(HEADER_NAMES):
                    df.columns = HEADER_NAMES[:len(df.columns)]
                
                df = df[df['timestamp'] != 'timestamp']
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # Violation timeline
                st.write("### ⏱️ Violation Timeline")
                timeline_df = df.resample('1min', on='timestamp').size().reset_index(name='Violations')
                st.line_chart(timeline_df.set_index('timestamp'))
                
                st.divider()
                
                # Violation records
                st.subheader("📋 Recent Violation Records")
                df_display = df.sort_values(by='timestamp', ascending=False).head(20)
                st.dataframe(df_display, use_container_width=True)
                
                st.divider()
                
                # Violation gallery
                st.subheader("🖼️ Violation Gallery (Latest 12)")
                latest_violations = df.sort_values(by='timestamp', ascending=False).head(12)
                
                grid_cols = st.columns(4)
                for i, row in latest_violations.reset_index().iterrows():
                    with grid_cols[i % 4]:
                        img_path = os.path.join(VIOLATION_DIR, row['image_filename'])
                        if os.path.exists(img_path):
                            stability = float(row.get('stability_score', 0.5))
                            st.image(
                                img_path,
                                caption=f"ID: {int(row['track_id'])} | Vote: {float(row['vote_score']):.0%} | Stable: {stability:.0%}",
                                use_container_width=True
                            )
            else:
                st.info("No violation records yet. Upload and process a video to see data.")
        except Exception as e:
            st.error(f"Error loading dashboard: {e}")
    else:
        st.info("No violation log found. Process a video first to generate data.")