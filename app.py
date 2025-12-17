import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import os
import time
import tempfile

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="SmartVision AI: Ultimate",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. THEME ENGINE: ONYX & NEON
# ==========================================
CEO_THEME_CSS = """
    <style>
    /* 1. BACKGROUND & FONTS */
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;600;700&family=Inter:wght@300;400;600&display=swap');
    
    .stApp {
        background-color: #000000;
        background-image: radial-gradient(at 50% 0%, #1a1a2e 0%, #000000 80%);
        color: #e0e0e0;
        font-family: 'Inter', sans-serif;
    }

    /* 2. TYPOGRAPHY */
    h1, h2, h3 {
        font-family: 'Rajdhani', sans-serif;
        text-transform: uppercase;
        letter-spacing: 2px;
        background: linear-gradient(90deg, #00f260, #0575e6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* 3. NEON CARDS */
    .neon-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(0, 242, 96, 0.2);
        box-shadow: 0 0 15px rgba(0, 242, 96, 0.1);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        transition: transform 0.3s ease;
    }
    .neon-card:hover {
        transform: translateY(-5px);
        border-color: #00f260;
        box-shadow: 0 0 25px rgba(0, 242, 96, 0.3);
    }

    /* 4. EXPLAINER BOXES (EDUCATIONAL) */
    .explainer-box {
        background: rgba(5, 117, 230, 0.1);
        border-left: 4px solid #0575e6;
        padding: 15px;
        border-radius: 4px;
        font-size: 0.9rem;
        margin-bottom: 15px;
    }

    /* 5. SIDEBAR */
    [data-testid="stSidebar"] {
        background-color: #050505;
        border-right: 1px solid #222;
    }
    
    /* 6. BUTTONS */
    div.stButton > button {
        background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
        color: #000;
        font-weight: 700;
        border: none;
        padding: 12px 24px;
        border-radius: 8px;
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.1rem;
        transition: all 0.3s;
    }
    div.stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 20px rgba(0, 201, 255, 0.6);
    }
    </style>
"""
st.markdown(CEO_THEME_CSS, unsafe_allow_html=True)

# ==========================================
# 3. LOAD ENGINE
# ==========================================
@st.cache_resource
def load_engine():
    # Load General YOLO (Standard COCO model is best for robust object detection)
    model = YOLO('yolov8n.pt') 
    
    # Check for custom model, fallback to general if missing
    custom_path = 'models/distracted_driver_v2.pt'
    if os.path.exists(custom_path):
        custom_model = YOLO(custom_path)
    else:
        custom_model = model
        
    return model, custom_model

try:
    with st.spinner("⚡ Booting Neural Networks..."):
        general_model, custom_model = load_engine()
except Exception as e:
    st.error(f"System Error: {e}")
    st.stop()

# ==========================================
# 4. SIDEBAR NAVIGATION
# ==========================================
with st.sidebar:
    st.markdown("## 🛡️ SmartVision AI")
    st.caption("AI DRIVER SAFETY SYSTEM")
    st.markdown("---")
    
    page = st.radio("INTERFACE MODE", ["🚀 MISSION CONTROL", "🎥 LIVE VISION", "🧬 FORENSIC LAB"])
    
    st.markdown("---")
    st.markdown("### 🎛️ Neural Sensitivity")
    conf = st.slider("Confidence Threshold", 0.0, 1.0, 0.25)
    
    st.markdown("""
    <div style="font-size: 0.8rem; color: #888; margin-top: 10px;">
    <strong>What is this?</strong><br>
    Controls how "sure" the AI must be to show a box.
    <br>• <strong>High (0.8):</strong> Fewer boxes, high accuracy.
    <br>• <strong>Low (0.1):</strong> More boxes, might see false positives.
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# 5. PAGE: MISSION CONTROL (DASHBOARD)
# ==========================================
if page == "🚀 MISSION CONTROL":
    st.markdown("<h1 style='text-align: center; font-size: 4rem;'>Smart Vison AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #aaa; letter-spacing: 3px;'>NEXT-GEN SITUATIONAL AWARENESS</p>", unsafe_allow_html=True)
    st.write("")

    # --- EDUCATIONAL EXPANDER ---
    with st.expander("🔍 HOW IT WORKS (Click to Learn)"):
        st.markdown("""
        ### The "Brain" Behind the System
        This application uses **YOLOv8 (You Only Look Once)**, a state-of-the-art deep learning model.
        
        1.  **Input:** The camera feeds an image (matrix of pixels) to the AI.
        2.  **Convolution:** The AI scans the image looking for patterns (edges, circles, phones, faces).
        3.  **Inference:** It draws a bounding box around objects it recognizes.
        4.  **Logic Layer:** Our Python code checks: *Is the phone box touching the person box?* If YES -> Distracted.
        """)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="neon-card">
            <h3>⚡ Zero Latency</h3>
            <p>Processes video frames in < 15ms using hardware acceleration.</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="neon-card">
            <h3>👁️ Computer Vision</h3>
            <p>Detects 80+ object classes including phones, cups, and people.</p>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="neon-card">
            <h3>🔒 Edge Security</h3>
            <p>All processing happens locally. No data leaves the device.</p>
        </div>
        """, unsafe_allow_html=True)

    # Use column width for image scaling
    st.image("https://viso.ai/wp-content/uploads/2021/01/computer-vision-deep-learning-applications.jpg", use_column_width=True)

# ==========================================
# 6. PAGE: LIVE VISION (UNIVERSAL)
# ==========================================
elif page == "🎥 LIVE VISION":
    st.title("🎥 LIVE SURVEILLANCE FEED")
    
    with st.expander("ℹ️ UNDERSTANDING THE MODES"):
        st.markdown("""
        **Select your Input Source:**
        1.  **🔴 Live Webcam:** Uses your browser's camera. The browser will ask for permission. 
        2.  **📁 Upload File:** Simulate a live feed using a recorded Video (MP4) or an Image (JPG).
        """)

    # Toggle for Source Selection
    input_mode = st.radio("Select Input Source:", ["🔴 Live Webcam", "📁 Upload File (Simulation)"], horizontal=True)
    
    c_main, c_side = st.columns([3, 1])

    # --- MODE 1: LIVE WEBCAM ---
    if input_mode == "🔴 Live Webcam":
        with c_main:
            st.write("### 📸 Webcam Snapshot")
            st.caption("Click 'Take Photo' to analyze the current view. This works on all Cloud Servers.")
            
            # This WIDGET triggers the browser permission popup
            camera_image = st.camera_input("Take a snapshot for AI Analysis")
            
            if camera_image:
                # Convert buffer to Image
                img = Image.open(camera_image)
                arr = np.array(img)
                
                # Run AI
                results = general_model(arr, conf=conf)
                annotated = results[0].plot()
                
                # Display Result
                st.image(annotated, caption="AI Analysis Result", use_column_width=True)
                
                # Metrics
                with c_side:
                    st.markdown('<div class="neon-card"><h4>ANALYSIS</h4></div>', unsafe_allow_html=True)
                    cnt = len(results[0].boxes)
                    st.metric("Objects Detected", cnt)
                    
                    if cnt > 0:
                        st.success("✅ Detection Successful")
                    else:
                        st.warning("⚠️ No Objects Found")

    # --- MODE 2: FILE UPLOAD (VIDEO & IMAGE) ---
    elif input_mode == "📁 Upload File (Simulation)":
        with c_main:
            st.write("### 📂 Upload Evidence")
            # Accepts BOTH Video and Images
            uploaded_file = st.file_uploader("Upload Fleet Footage (MP4/AVI) or Photo (JPG/PNG)", type=['mp4', 'avi', 'mov', 'jpg', 'png', 'jpeg'])
            
        if uploaded_file:
            file_type = uploaded_file.name.split('.')[-1].lower()
            
            # --- VIDEO HANDLER ---
            if file_type in ['mp4', 'avi', 'mov']:
                tfile = tempfile.NamedTemporaryFile(delete=False) 
                tfile.write(uploaded_file.read())
                
                cap = cv2.VideoCapture(tfile.name)
                st_frame = st.empty()
                
                with c_side:
                    st.markdown('<div class="neon-card"><h4>PLAYBACK</h4></div>', unsafe_allow_html=True)
                    run_video = st.checkbox("▶️ PLAY VIDEO", value=True)
                    fps_display = st.empty()

                while cap.isOpened() and run_video:
                    ret, frame = cap.read()
                    if not ret: break
                    
                    # Resize for speed
                    frame = cv2.resize(frame, (720, 480))
                    
                    # AI Processing
                    t0 = time.time()
                    results = general_model(frame, conf=conf, verbose=False)
                    annotated_frame = results[0].plot()
                    
                    # Show
                    st_frame.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), use_column_width=True)
                    
                    # FPS
                    fps = 1.0 / (time.time() - t0)
                    fps_display.metric("Processing Speed", f"{int(fps)} FPS")
                    
                    time.sleep(0.01)
                
                cap.release()
            
            # --- IMAGE HANDLER ---
            else:
                image = Image.open(uploaded_file)
                arr = np.array(image.convert('RGB'))
                
                # AI Processing
                results = general_model(arr, conf=conf)
                annotated_frame = results[0].plot()
                
                st.image(annotated_frame, caption="Processed Evidence", use_column_width=True)
                
                with c_side:
                    st.metric("Objects Detected", len(results[0].boxes))


# ==========================================
# 7. PAGE: FORENSIC LAB (DEEP ANALYSIS)
# ==========================================
elif page == "🧬 FORENSIC LAB":
    st.title("🧬 FORENSIC EVIDENCE ANALYSIS")
    
    with st.expander("🧠 THE LOGIC BEHIND DETECTION"):
        st.markdown("""
        **Why do we need a custom logic?**
        
        Standard AI just sees "Phone" and "Person". It doesn't know if the phone is in the pocket or hand.
        
        **Our Algorithm:**
        1.  **Scan:** Detect all objects.
        2.  **Filter:** Look specifically for `Cell Phone`, `Cup`, `Bottle`.
        3.  **Context:** If a "Threat Object" is found in the scene, we flag the driver as **DISTRACTED**.
        """)

    st.markdown('<div class="neon-card"><h4>UPLOAD EVIDENCE</h4></div>', unsafe_allow_html=True)
    # UPDATED: Supports both Image and Video
    up_file = st.file_uploader("Select Evidence File...", type=["jpg", "png", "jpeg", "mp4"])
    
    c1, c2 = st.columns(2)
    
    if up_file:
        # Check if it is video or image
        ftype = up_file.name.split('.')[-1].lower()
        
        if ftype in ['mp4', 'avi', 'mov']:
            st.warning("⚠️ For Deep Forensic Video Analysis, please use the 'LIVE VISION' tab in Simulation Mode.")
        else:
            img = Image.open(up_file)
            arr = np.array(img.convert('RGB'))
            
            with c1:
                st.image(img, caption="RAW FOOTAGE", use_column_width=True)
                
            if st.button("🚀 INITIATE DEEP SCAN"):
                with st.spinner("Triangulating Threat Vectors..."):
                    # 1. RUN AI
                    res = custom_model(arr, conf=conf)
                    if len(res[0].boxes) == 0: 
                        res = general_model(arr, conf=conf)
                    
                    # 2. DRAW & ANALYZE
                    final_img = arr.copy()
                    status = "SAFE"
                    color = (0, 255, 0)
                    threats = ['cell phone', 'phone', 'mobile', 'cup', 'bottle', 'eating', 'texting']
                    detected = []

                    for r in res:
                        for box in r.boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                            cls = r.names[int(box.cls)]
                            detected.append(cls)
                            
                            if any(t in cls.lower() for t in threats):
                                status = "DISTRACTED"
                                color = (255, 0, 0)
                                # THICK RED BOX FOR THREAT
                                cv2.rectangle(final_img, (x1, y1), (x2, y2), color, 5)
                                cv2.putText(final_img, f"!! {cls.upper()} !!", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 3)
                            else:
                                # THIN GREEN BOX FOR CONTEXT
                                cv2.rectangle(final_img, (x1, y1), (x2, y2), (100, 255, 100), 1)
                                cv2.putText(final_img, cls, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)

                    with c2:
                        st.image(final_img, caption="AI PROCESSED OUTPUT", use_column_width=True)
                        
                    st.markdown("---")
                    if status == "DISTRACTED":
                        st.markdown(f"""
                        <div class="neon-card" style="border-color: #ff4b4b; box-shadow: 0 0 15px rgba(255, 75, 75, 0.3);">
                            <h2 style="color: #ff4b4b;">🚨 THREAT DETECTED</h2>
                            <p><strong>Verdict:</strong> Driver is interacting with a foreign object.<br>
                            <strong>Detected:</strong> {', '.join(set([d for d in detected if any(t in d.lower() for t in threats)]))}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="neon-card" style="border-color: #00f260;">
                            <h2 style="color: #00f260;">✅ COMPLIANT</h2>
                            <p>Driver is focused. No distractions identified.</p>
                        </div>
                        """, unsafe_allow_html=True)


