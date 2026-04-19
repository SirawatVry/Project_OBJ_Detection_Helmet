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
MODEL_PATH = "runs/detect/runs/detect/basemodel.pt_finetune_v27map0.857map950.635B/weights/best.pt"
VIOLATION_DIR = "./violations/no_helmet_captures"
LOG_FILE = "./violations/no_helmet_log.csv"

CONF_THRES = 0.4
IOU_THRES = 0.5
VOTING_WINDOW = 10
VOTING_THRESHOLD = 0.7
CAPTURE_INTERVAL = 200

# กำหนดชื่อคอลัมน์มาตรฐาน
HEADER_NAMES = ['timestamp', 'frame_number', 'track_id', 'confidence', 'x1', 'y1', 'x2', 'y2', 'image_filename', 'vote_score']

CLASS_NAMES = ['helmet', 'motorcycle', 'no_helmet']
COLORS = {
    0: (0, 255, 0),      # Helmet - Green
    1: (255, 0, 0),      # Motorcycle - Blue
    2: (0, 0, 255),      # No helmet - Red
}

if not os.path.exists(VIOLATION_DIR):
    os.makedirs(VIOLATION_DIR)

# ---------------------------
# 2. CORE FUNCTIONS
# ---------------------------

def get_majority_class(track_id, current_cls, current_conf, class_history, confidence_history):
    class_history[track_id].append(current_cls)
    confidence_history[track_id].append(current_conf)
    
    if len(class_history[track_id]) > VOTING_WINDOW:
        class_history[track_id].pop(0)
        confidence_history[track_id].pop(0)
    
    no_helmet_count = sum(1 for cls in class_history[track_id] if cls == 2)
    no_helmet_ratio = no_helmet_count / len(class_history[track_id])
    
    final_cls = 2 if no_helmet_ratio >= VOTING_THRESHOLD else current_cls
    return final_cls, no_helmet_ratio

def process_video(video_path, model):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    tracker = Tracker(
        distance_function="euclidean",
        distance_threshold=250,
        initialization_delay=2,
        hit_counter_max=5,
        past_detections_length=7
    )

    class_history = defaultdict(list)
    confidence_history = defaultdict(list)
    last_capture_time = {}
    active_track_ids = set()
    motorcycle_ids = set()
    violation_ids = set() 

    frame_placeholder = st.empty()
    progress_bar = st.progress(0)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        results = model.predict(frame, conf=CONF_THRES, iou=IOU_THRES, verbose=False, agnostic_nms=True)[0]

        detections = []
        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy().astype(int)
            
            for box, conf, cls in zip(boxes, confs, classes):
                cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                detections.append(Detection(
                    points=np.array([[cx, cy]]),
                    scores=np.array([conf]),
                    data={'box': box, 'conf': conf, 'cls': cls}
                ))

        tracked_objects = tracker.update(detections=detections)
        
        current_ids = {obj.id for obj in tracked_objects}
        for tid in (active_track_ids - current_ids):
            for d in [class_history, confidence_history, last_capture_time]:
                if tid in d: del d[tid]
        active_track_ids = current_ids

        for obj in tracked_objects:
            if obj.last_detection is None or obj.age < 3: continue
            
            tid = obj.id
            box = obj.last_detection.data['box']
            conf = obj.last_detection.data['conf']
            cls = obj.last_detection.data['cls']
            
            final_cls, vote_score = get_majority_class(tid, cls, conf, class_history, confidence_history)

            x1, y1, x2, y2 = map(int, box)
            x1_c, y1_c = max(0, min(x1, width-1)), max(0, min(y1, height-1))
            x2_c, y2_c = max(0, min(x2, width-1)), max(0, min(y2, height-1))
            
            visible_ratio = ((x2_c-x1_c)*(y2_c-y1_c)) / max(1, (x2-x1)*(y2-y1))
            if visible_ratio < 0.2: continue 

            if final_cls == 1: motorcycle_ids.add(tid)
            
            if final_cls == 2 and vote_score >= VOTING_THRESHOLD:
                violation_ids.add(tid)
                
                if tid not in last_capture_time or (frame_count - last_capture_time[tid] >= CAPTURE_INTERVAL):
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
                            writer.writerow([timestamp.isoformat(), frame_count, tid, f"{conf:.2f}", x1, y1, x2, y2, filename, f"{vote_score:.2f}"])
                    last_capture_time[tid] = frame_count

            color = COLORS.get(final_cls, (255, 255, 255))
            cv2.rectangle(frame, (x1_c, y1_c), (x2_c, y2_c), color, 3)
            label = f"ID:{tid} {CLASS_NAMES[final_cls]} ({vote_score:.0%})"
            cv2.putText(frame, label, (x1_c, y1_c-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB")
        frame_count += 1
        progress_bar.progress(frame_count / total_frames if total_frames > 0 else 0)

    cap.release()
    return len(motorcycle_ids), len(violation_ids)

# ---------------------------
# 3. STREAMLIT UI
# ---------------------------
st.set_page_config(page_title="Helmet Analytics", layout="wide")
st.title("🛡️ AI Helmet Violation Monitoring")

@st.cache_resource
def get_model(): return YOLO(MODEL_PATH)
model = get_model()

uploaded_file = st.file_uploader("Upload Video", type=['mp4', 'avi', 'mov'])

if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    if st.button("Run Detection"):
        m_count, v_count = process_video(tfile.name, model)
        st.success("Analysis Finished")
        c1, c2, c3 = st.columns(3)
        c1.metric("Motorcycles", m_count)
        c2.metric("Violations", v_count)
        rate = (v_count / m_count * 100) if m_count > 0 else 0
        c3.metric("Violation Rate", f"{rate:.1f}%")

st.divider()

# --- ส่วน DASHBOARD แก้ไขโครงสร้างเพื่อความปลอดภัย ---
if os.path.exists(LOG_FILE):
    try:
        df = pd.read_csv(LOG_FILE, on_bad_lines='skip')
        if not df.empty:
            if len(df.columns) == len(HEADER_NAMES): df.columns = HEADER_NAMES
            df = df[df['timestamp'] != 'timestamp']
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            st.subheader("📈 Analytics & Trends")
            
            # ✅ กราฟแรก: Violation Timeline แบบเดี่ยวๆ เต็มจอ
            st.write("### ⏱️ Violation Timeline (จำนวนการตรวจพบตามช่วงเวลา)")
            timeline_df = df.resample('1min', on='timestamp').size().reset_index(name='Violations')
            st.line_chart(timeline_df.set_index('timestamp'))

            st.divider()
        
        if not df.empty:
            # ตรวจสอบและบังคับใช้หัวตาราง
            if len(df.columns) == len(HEADER_NAMES):
                df.columns = HEADER_NAMES
            
            # ล้างแถวข้อมูลที่เป็นหัวข้อซ้ำ
            df = df[df['timestamp'] != 'timestamp']

            st.subheader("📋 Violation Records")
            if 'timestamp' in df.columns:
                # เรียงลำดับจากล่าสุด
                df_display = df.sort_values(by='timestamp', ascending=False).head(20)
                st.dataframe(df_display, use_container_width=True)

                st.subheader("🖼️ Violation Gallery (Latest)")
                latest_violations = df.sort_values(by='timestamp', ascending=False).head(12)
                
                # แสดงแกลเลอรีภาพ
                grid_cols = st.columns(4)
                for i, row in latest_violations.reset_index().iterrows():
                    with grid_cols[i % 4]:
                        img_path = os.path.join(VIOLATION_DIR, row['image_filename'])
                        if os.path.exists(img_path):
                            st.image(
                                img_path, 
                                caption=f"ID: {row['track_id']} | Vote: {float(row['vote_score']):.0%}",
                                use_container_width=True
                            )
            else:
                st.error("ไม่พบคอลัมน์ 'timestamp' ในไฟล์ Log กรุณาลบไฟล์เก่าแล้วรันใหม่")
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการโหลด Dashboard: {e}")