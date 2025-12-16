import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import os
import time
import gc

# ==========================================
# 1. VISUAL CONFIGURATION (THEME ENGINE)
# ==========================================
st.set_page_config(
    page_title="PatrolIQ: Enterprise AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- PREMIUM CSS STYLING ---
st.markdown("""
    <style>
    .stApp { background: radial-gradient(circle at 50% 10%, #1a202c 0%, #000000 100%); color: #ffffff; font-family: 'Inter', sans-serif; }
    [data-testid="stSidebar"] { background-color: rgba(10, 10, 10, 0.95); border-right: 1px solid rgba(255, 255, 255, 0.05); }
    h1, h2, h3 { background: linear-gradient(to right, #00f260, #0575e6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .neo-card { background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 16px; padding: 24px; margin-bottom: 20px; }
    div.stButton > button { background: linear-gradient(135deg, #0575e6 0%, #00f260 100%); color: white; border: none; padding: 12px 24px; border-radius: 10px; font-weight: bold; width: 100%; transition: transform 0.2s; }
    div.stButton > button:hover { transform: scale(1.02); }
    div[data-testid="stMetricValue"] { color: #00f260; font-size: 2rem !important; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. MODEL LOADING (CACHED)
# ==========================================
@st.cache_resource
def load_models():
    # Load General YOLO (Backup)
    detector_general = YOLO('yolov8n.pt')
    
    # Load Custom Models
    yolo_path = 'runs/detect/smartvision_yolo/weights/best.pt'
    classifier_path = 'models/mobilenet_v2_smartvision.h5'
    
    detector_custom = YOLO(yolo_path) if os.path.exists(yolo_path) else detector_general
    
    classifier = None
    if os.path.exists(classifier_path):
        from tensorflow.keras.models import load_model
        classifier = load_model(classifier_path)
        
    return detector_general, detector_custom, classifier

try:
    with st.spinner("🚀 Initializing Neural Core..."):
        detector_general, detector_custom, classifier = load_models()
except Exception as e:
    st.error(f"System Error: {e}")
    st.stop()

# ==========================================
# 3. SIDEBAR NAVIGATION
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1169/1169382.png", width=60)
    st.markdown("### PatrolIQ Enterprise")
    st.caption("v3.0 | Stable Build")
    st.markdown("---")
    
    selected_page = st.radio(
        "MODULES",
        ["Dashboard", "Live Surveillance", "Biometric Audit", "Diagnostics"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### ⚙️ Calibration")
    conf_threshold = st.slider("Sensitivity", 0.0, 1.0, 0.35)
    st.info("System: ONLINE")

# ==========================================
# 4. MODULE: DASHBOARD
# ==========================================
if selected_page == "Dashboard":
    st.markdown("<h1 style='text-align: center; font-size: 3.5rem;'>PatrolIQ</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #a0aec0; letter-spacing: 2px;'>ADVANCED SITUATIONAL AWARENESS PLATFORM</p>", unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    with c1: st.markdown("""<div class="neo-card"><h3>👁️ Omni-Watch</h3><p>Real-time YOLOv8 surveillance.</p></div>""", unsafe_allow_html=True)
    with c2: st.markdown("""<div class="neo-card"><h3>🧠 Neuro-Guard</h3><p>Hybrid Driver Behavior Analysis.</p></div>""", unsafe_allow_html=True)
    with c3: st.markdown("""<div class="neo-card"><h3>⚡ Live-Sync</h3><p>Zero-latency hardware acceleration.</p></div>""", unsafe_allow_html=True)
    st.image("https://viso.ai/wp-content/uploads/2021/01/computer-vision-deep-learning-applications.jpg", use_container_width=True)

# ==========================================
# 5. MODULE: LIVE SURVEILLANCE (FIXED!)
# ==========================================
elif selected_page == "Live Surveillance":
    st.title("🎥 Active Surveillance Feed")
    st.markdown("Protocol: **Direct Hardware Access** (Lag-Free)")
    
    col_video, col_stats = st.columns([3, 1])
    
    with col_stats:
        st.markdown("""<div class="neo-card"><h4>📡 Controls</h4></div>""", unsafe_allow_html=True)
        # Use a checkbox as an ON/OFF switch
        run_camera = st.checkbox("🔴 ACTIVATE CAMERA", value=False)
        st.markdown("---")
        kpi1 = st.empty()
        kpi2 = st.empty()
        
    with col_video:
        video_placeholder = st.empty()
        
        if not run_camera:
            video_placeholder.markdown("""
            <div style="background: #111; border-radius: 12px; height: 400px; display: flex; align-items: center; justify-content: center; border: 1px solid #333;">
                <p style="color: #666;">FEED TERMINATED - CLICK ACTIVATE</p>
            </div>
            """, unsafe_allow_html=True)

    # --- CAMERA LOGIC (THE FIX) ---
    if run_camera:
        # 0 is the default webcam. Try 1 if you have an external cam.
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("❌ Error: Could not access webcam. Please check permissions or try a different browser.")
        else:
            stop_button = False
            while run_camera and not stop_button:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to read frame stream.")
                    break
                
                # AI Processing
                results = detector_general(frame, conf=conf_threshold, verbose=False)
                annotated_frame = results[0].plot()
                
                # Convert BGR (OpenCV) to RGB (Streamlit)
                frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                
                # Update UI Elements
                video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                
                # Update Stats
                obj_count = len(results[0].boxes)
                kpi1.metric("Objects", obj_count)
                
                # Check if user unchecked the box to stop immediately
                # (Streamlit re-runs script on interaction, but loop needs manual break)
                # Note: The checkbox state is handled by Streamlit's rerun mechanism mostly,
                # but adding a small sleep helps CPU usage.
                time.sleep(0.01)
            
            cap.release()

# ==========================================
# 6. MODULE: BIOMETRIC AUDIT (DRIVER SAFETY)
# ==========================================
elif selected_page == "Biometric Audit":
    st.title("🚗 Driver Safety Audit")
    
    col_input, col_result = st.columns(2)
    
    with col_input:
        st.markdown("""<div class="neo-card"><h4>📂 Evidence Ingestion</h4></div>""", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Source", use_container_width=True)
            img_array = np.array(image.convert('RGB'))
            
            st.write("")
            if st.button("🚀 EXECUTE ANALYSIS"):
                with st.spinner("Analyzing..."):
                    # Hybrid Logic
                    results = detector_custom(img_array, conf=conf_threshold)
                    used_model = "Custom Core"
                    if len(results[0].boxes) == 0:
                        results = detector_general(img_array, conf=0.15)
                        used_model = "Backup Core"
                    
                    annotated_img = img_array.copy()
                    final_status = "Safe Driving"
                    is_danger = False
                    
                    # Detection Loop
                    detected_classes = [results[0].names[int(b.cls)] for b in results[0].boxes]
                    distractions = ['cell phone', 'cup', 'bottle', 'remote', 'sandwich']
                    has_distraction = any(x in distractions for x in detected_classes)

                    for result in results:
                        for box in result.boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                            cls_name = result.names[int(box.cls)]
                            
                            if cls_name in distractions:
                                label = f"THREAT: {cls_name.upper()}"
                                color = (255, 0, 0)
                                is_danger = True
                                final_status = "Distracted"
                            elif cls_name == 'person' and classifier:
                                if has_distraction:
                                    label = "DISTRACTED (Object Confirmed)"
                                    color = (255, 0, 0)
                                    is_danger = True
                                    final_status = "Distracted"
                                else:
                                    crop = img_array[y1:y2, x1:x2]
                                    if crop.size > 0:
                                        crop_resized = cv2.resize(crop, (224, 224)) / 255.0
                                        crop_input = np.expand_dims(crop_resized, axis=0)
                                        pred = classifier.predict(crop_input)
                                        sub_id = np.argmax(pred)
                                        CLASS_NAMES = ['Distracted', 'Safe Driving', 'Talking', 'Texting']
                                        if sub_id < len(CLASS_NAMES):
                                            sub_label = CLASS_NAMES[sub_id]
                                            if "Safe" not in sub_label:
                                                is_danger = True
                                                final_status = sub_label
                                            label = sub_label
                                            color = (0, 255, 0) if "Safe" in sub_label else (255, 0, 0)
                                        else: label = "Person"; color = (0, 255, 0)
                            else: label = cls_name.title(); color = (0, 255, 0)

                            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 4)
                            cv2.putText(annotated_img, label, (x1, y1-15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    
                    with col_result:
                        st.markdown(f"""<div class="neo-card"><h4>🤖 AI Verdict ({used_model})</h4></div>""", unsafe_allow_html=True)
                        st.image(annotated_img, use_container_width=True)
                        st.markdown("---")
                        if is_danger:
                            st.error(f"🚨 VIOLATION: {final_status}")
                        elif len(results[0].boxes) == 0:
                            st.warning("⚠️ Inconclusive Scan")
                        else:
                            st.success("✅ COMPLIANT: Safe Driving")

# ==========================================
# 7. MODULE: DIAGNOSTICS
# ==========================================
elif selected_page == "Diagnostics":
    st.title("📊 System Diagnostics")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""<div class="neo-card"><h4>📈 Learning Curves</h4></div>""", unsafe_allow_html=True)
        if os.path.exists('runs/detect/smartvision_yolo/results.png'): st.image('runs/detect/smartvision_yolo/results.png', use_container_width=True)
        else: st.info("No Log Data")
    with c2:
        st.markdown("""<div class="neo-card"><h4>🧩 Confusion Matrix</h4></div>""", unsafe_allow_html=True)
        if os.path.exists('runs/detect/smartvision_yolo/confusion_matrix.png'): st.image('runs/detect/smartvision_yolo/confusion_matrix.png', use_container_width=True)
        else: st.info("No Log Data")
