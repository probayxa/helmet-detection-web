"""
Helmet Detection Web Application
Streamlit-based UI for viewing reports and performing predictions
"""
import streamlit as st
import streamlit.components.v1 as components
from ultralytics import YOLO
import cv2
import os
import numpy as np
from pathlib import Path
import tempfile
from PIL import Image
import pandas as pd
import time
import shutil

# ⚙️ Page configuration
st.set_page_config(
    page_title="Helmet Detection System",
    page_icon="🪖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 🎨 Load custom CSS
def load_css(file_name: str = "style.css"):
    css_path = Path(__file__).parent / file_name
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

load_css()

# 🪖 Hero header
st.markdown(
    """
    <div style="
        padding: 1.5rem 1.75rem;
        border-radius: 1.25rem;
        background: linear-gradient(90deg, rgba(37,99,235,0.9), rgba(59,130,246,0.7), rgba(56,189,248,0.7));
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        box-shadow: 0 18px 45px rgba(15,23,42,0.6);
        margin-bottom: 1.8rem;
    ">
        <div>
            <div style="font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.18em; color: rgba(15,23,42,0.9);">
                AI-POWERED SAFETY
            </div>
            <div style="font-size: 2rem; font-weight: 800; letter-spacing: 0.06em; text-transform: uppercase; color: #0b1120; margin-top: 0.25rem;">
                Helmet Detection System
            </div>
            <div style="font-size: 0.95rem; color: #0f172a; margin-top: 0.35rem;">
                Real-time detection and analytics for helmet usage in traffic and industrial environments.
            </div>
        </div>
        <div style="font-size: 2.8rem;">
            🪖
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Configuration
MODEL_PATH = "runs/train/s_huq03l/weights/best.pt"
# Sửa lại đúng "utilities"
REPORTS_DIR = "utilities/analysis_reports"
CLASS_NAMES = ["With Helmet", "Without Helmet"]

# Output directory for saving uploads and results
DATA_DIR = "streamlit_data"
os.makedirs(DATA_DIR, exist_ok=True)


# Load model with caching
@st.cache_resource
def load_model():
    """Load YOLO model once and cache it"""
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found: {MODEL_PATH}")
        return None
    return YOLO(MODEL_PATH)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = load_model()

# Sidebar navigation
st.sidebar.title("Helmet Detection")
page = st.sidebar.radio(
    "Navigation",
    ["Reports", "Predict"],
    index=0
)

# Reports Page
if page == "Reports":
    st.title("Model Analysis Reports")
    st.markdown("View comprehensive analysis reports of the helmet detection model performance and dataset statistics.")
    
    if not os.path.exists(REPORTS_DIR):
        st.error(f"Reports directory not found: {REPORTS_DIR}")
        st.info("Please run the analysis scripts first to generate reports.")
    else:
        # Get all report files
        reports_path = Path(REPORTS_DIR)
        png_files = sorted(reports_path.glob("*.png"))
        csv_files = sorted(reports_path.glob("*.csv"))
        
        if not png_files and not csv_files:
            st.warning("No report files found in the reports directory.")
        else:
            # Display PNG images
            if png_files:
                st.header("Visual Reports")
                
                for png_file in png_files:
                    # Create a nice title from filename
                    title = png_file.stem.replace("_", " ").title()
                    
                    st.subheader(title)
                    
                    # Display image
                    st.image(str(png_file), width='stretch')
                    
                    # Download button
                    with open(png_file, "rb") as f:
                        st.download_button(
                            label=f"Download {png_file.name}",
                            data=f.read(),
                            file_name=png_file.name,
                            mime="image/png",
                            key=f"download_{png_file.name}"
                        )
                    
                    st.divider()
            
            # Display CSV files as tables
            if csv_files:
                st.header("Data Reports")
                
                for csv_file in csv_files:
                    title = csv_file.stem.replace("_", " ").title()
                    st.subheader(title)
                    
                    try:
                        df = pd.read_csv(csv_file)
                        st.dataframe(df, width='stretch')
                        
                        # Download button
                        csv_data = df.to_csv(index=False)
                        st.download_button(
                            label=f"Download {csv_file.name}",
                            data=csv_data,
                            file_name=csv_file.name,
                            mime="text/csv",
                            key=f"download_{csv_file.name}"
                        )
                    except Exception as e:
                        st.error(f"Error reading {csv_file.name}: {str(e)}")
                    
                    st.divider()

# Predict Page
# Predict Page
elif page == "Predict":
    # Tiêu đề nhỏ hơn vì header lớn đã có ở trên
    st.markdown(
        "### 🔍 Prediction & Streaming\nUpload media or use webcam for real-time helmet detection.",
    )

    # Hàng 3 stats card (demo, bạn có thể chỉnh text sau)
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown(
            """
            <div class="stats-card">
                <div class="stats-card-title">Mode</div>
                <div class="stats-card-value">Image • Video • Webcam</div>
                <div class="stats-card-desc">Choose input type using the tabs below.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_b:
        st.markdown(
            """
            <div class="stats-card">
                <div class="stats-card-title">Model</div>
                <div class="stats-card-value">YOLOv11</div>
                <div class="stats-card-desc">Helmet / no-helmet binary classification.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_c:
        st.markdown(
            """
            <div class="stats-card">
                <div class="stats-card-title">Tips</div>
                <div class="stats-card-desc">
                    • Use clear images<br>
                    • Faces & helmets visible<br>
                    • Adjust thresholds in the sidebar.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    
    if st.session_state.model is None:
        st.error("Model could not be loaded. Please check the model path.")
        st.stop()
    
    # Sidebar settings
    st.sidebar.header("Detection Settings")
    conf_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        step=0.05,
        help="Minimum confidence score for detections"
    )
    
    iou_threshold = st.sidebar.slider(
        "IoU Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Intersection over Union threshold for NMS"
    )
    
    show_conf = st.sidebar.checkbox("Show Confidence Scores", value=True)
    
    # Tabs for different input methods
    tab1, tab2, tab3 = st.tabs(["Upload Image", "Upload Video", "Webcam Stream"])
    
    # Tab 1: Image Upload
    # Tab 1: Image Upload
    with tab1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)

        st.header("Image Prediction")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=["jpg", "jpeg", "png", "bmp"],
            help="Upload an image containing people with/without helmets."
        )

        # 👉 PHẦN CODE XỬ LÝ ẢNH CỦA BẠN GIỮ NGUYÊN Ở DƯỚI ĐÂY
        # (predict, hiển thị ảnh, bảng, v.v.)

        st.markdown("</div>", unsafe_allow_html=True)
        
        if uploaded_file is not None:
            # Display original image
            image = Image.open(uploaded_file)
            st.subheader("Original Image")
            st.image(image, width='stretch')
            
            # Predict button
            if st.button("Detect Helmets", type="primary"):
                with st.spinner("Processing image..."):
                    # Automatically save uploaded image
                    upload_filename = f"upload_{int(time.time())}_{uploaded_file.name}"
                    upload_path = os.path.join(DATA_DIR, upload_filename)
                    with open(upload_path, "wb") as f:
                        f.write(uploaded_file.getvalue())
                    st.info(f"Uploaded image saved to: {upload_path}")
                    
                    # Convert PIL to numpy array
                    img_array = np.array(image)
                    
                    # Run prediction
                    results = st.session_state.model.predict(
                        source=img_array,
                        conf=conf_threshold,
                        iou=iou_threshold,
                        verbose=False
                    )
                    
                    # Get annotated image
                    annotated_img = results[0].plot()
                    
                    # Save result image
                    result_filename = f"result_{int(time.time())}_{uploaded_file.name}"
                    result_path = os.path.join(DATA_DIR, result_filename)
                    cv2.imwrite(result_path, cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))
                    st.success(f"Result saved to: {result_path}")
                    
                    # Display results
                    st.subheader("Detection Results")
                    st.image(annotated_img, width='stretch')
                    
                    # Display statistics
                    if len(results[0].boxes) > 0:
                        boxes = results[0].boxes
                        num_detections = len(boxes)
                        
                        st.success(f"Found {num_detections} detection(s)")
                        
                        # Create summary table
                        detection_data = []
                        for i, box in enumerate(boxes):
                            cls = int(box.cls[0])
                            conf = float(box.conf[0])
                            detection_data.append({
                                "Detection": i + 1,
                                "Class": CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else f"Class {cls}",
                                "Confidence": f"{conf:.2%}"
                            })
                        
                        df = pd.DataFrame(detection_data)
                        st.dataframe(df, width='stretch', hide_index=True)
                    else:
                        st.info("No helmets detected in this image.")
    
        # Tab 2: Video Upload
    with tab2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)

        st.header("Video Prediction")
        uploaded_video = st.file_uploader(
            "Choose a video file",
            type=["mp4", "avi", "mov", "mkv"],
            help="Upload a video to detect helmets"
        )
        
        if uploaded_video is not None:
            # Read video data once
            video_data = uploaded_video.read()
            
            # Automatically save uploaded video
            upload_filename = f"upload_{int(time.time())}_{uploaded_video.name}"
            upload_path = os.path.join(DATA_DIR, upload_filename)
            with open(upload_path, "wb") as f:
                f.write(video_data)
            st.info(f"Uploaded video saved to: {upload_path}")
            
            # Save uploaded video to temporary file
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(video_data)
            tfile.close()
            
            st.subheader("Original Video")
            st.video(tfile.name)
            
            # Process video button
            if st.button("Process Video", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                progress_info = st.empty()
                
                try:
                    # Open input video
                    cap = cv2.VideoCapture(tfile.name)
                    if not cap.isOpened():
                        st.error("Cannot open video file")
                    else:
                        # Get video properties
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        
                        status_text.text(
                            f"Video info: {width}x{height}, {fps:.2f} FPS, {total_frames} frames"
                        )
                        
                        # Create output video path
                        output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                        output_path.close()
                        
                        # VideoWriter setup
                        fourcc = cv2.VideoWriter_fourcc(*"avc1")
                        out = cv2.VideoWriter(output_path.name, fourcc, int(fps), (width, height))
                        
                        if not out.isOpened():
                            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                            out = cv2.VideoWriter(output_path.name, fourcc, int(fps), (width, height))
                            if not out.isOpened():
                                st.error("Failed to initialize video writer. Please check codec support.")
                                raise IOError("Cannot initialize VideoWriter")
                        
                        # Process video frame-by-frame
                        frame_count = 0
                        start_time = time.time()
                        
                        while True:
                            ret, frame = cap.read()
                            if not ret:
                                break
                            
                            results = st.session_state.model.predict(
                                source=frame,
                                conf=conf_threshold,
                                iou=iou_threshold,
                                verbose=False
                            )
                            
                            annotated_frame = results[0].plot()
                            out.write(annotated_frame)
                            
                            frame_count += 1
                            progress = frame_count / total_frames
                            progress_bar.progress(progress)
                            
                            # elapsed / ETA
                            elapsed_time = time.time() - start_time
                            if frame_count > 0:
                                current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                                remaining_frames = total_frames - frame_count
                                eta_seconds = remaining_frames / current_fps if current_fps > 0 else 0
                                
                                elapsed_str = f"{int(elapsed_time // 60):02d}:{int(elapsed_time % 60):02d}"
                                eta_str = (
                                    f"{int(eta_seconds // 60):02d}:{int(eta_seconds % 60):02d}"
                                    if eta_seconds < 3600 else ">1h"
                                )
                                
                                bar_length = 30
                                filled_length = int(bar_length * progress)
                                bar = "█" * filled_length + "░" * (bar_length - filled_length)
                                
                                progress_percent = int(progress * 100)
                                progress_string = (
                                    f"Processing video: {progress_percent:3d}%|{bar}| "
                                    f"{frame_count}/{total_frames} "
                                    f"[{elapsed_str}<{eta_str}, {current_fps:.2f}frame/s]"
                                )
                                
                                progress_info.text(progress_string)
                        
                        cap.release()
                        out.release()
                        time.sleep(0.5)
                        
                        if not os.path.exists(output_path.name) or os.path.getsize(output_path.name) == 0:
                            st.error("Output video file is empty or was not created properly.")
                            raise IOError("Video file creation failed")
                        
                        total_time = time.time() - start_time
                        final_fps = frame_count / total_time if total_time > 0 else 0
                        total_time_str = f"{int(total_time // 60):02d}:{int(total_time % 60):02d}"
                        
                        bar = "█" * 30
                        final_progress_string = (
                            f"Processing video: 100%|{bar}| "
                            f"{frame_count}/{frame_count} "
                            f"[{total_time_str}<00:00, {final_fps:.2f}frame/s]"
                        )
                        progress_info.text(final_progress_string)
                        
                        time.sleep(1)
                        progress_bar.empty()
                        status_text.empty()
                        progress_info.empty()
                        
                        result_filename = f"result_{int(time.time())}_{uploaded_video.name}"
                        result_path = os.path.join(DATA_DIR, result_filename)
                        
                        shutil.copy2(output_path.name, result_path)
                        st.success(f"Processed video saved to: {result_path}")
                        
                        st.subheader("Processed Video")
                        st.video(result_path)
                        
                        with open(result_path, "rb") as f:
                            st.download_button(
                                label="Download Processed Video",
                                data=f.read(),
                                file_name="helmet_detection_output.mp4",
                                mime="video/mp4"
                            )
                        
                        st.success(
                            f"Video processing completed! Processed {frame_count} frames in "
                            f"{total_time_str} ({final_fps:.2f} FPS)."
                        )
                        
                except Exception as e:
                    st.error(f"Error processing video: {str(e)}")
                    st.exception(e)
                    progress_bar.empty()
                    status_text.empty()
                    if 'progress_info' in locals():
                        progress_info.empty()
            
            # Cleanup temp file
            if os.path.exists(tfile.name):
                os.unlink(tfile.name)

        st.markdown("</div>", unsafe_allow_html=True)

    # Tab 3: Webcam Stream
    with tab3:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)

        st.header("Webcam Stream")
        st.markdown("Use your webcam for real-time helmet detection.")
        
        col1, col2 = st.columns(2)
        run_stream = col1.button("Start Stream", type="primary")
        stop_stream = col2.button("Stop Stream")
        
        if run_stream:
            st.session_state.streaming = True
        if stop_stream:
            st.session_state.streaming = False
        
        frame_placeholder = st.empty()
        stats_placeholder = st.empty()
        
        if st.session_state.get("streaming", False):
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Cannot open webcam. Please check connection.")
                st.session_state.streaming = False
            else:
                st.success("Streaming started! Press 'Stop Stream' to end.")
                frame_placeholder = st.empty()
                stats_placeholder = st.empty()
                
                while st.session_state.streaming:
                    ret, frame = cap.read()
                    if not ret:
                        st.warning("Failed to read frame from webcam.")
                        break
                    
                    frame = cv2.flip(frame, 1)
                    
                    results = st.session_state.model.predict(
                        source=frame,
                        conf=conf_threshold,
                        iou=iou_threshold,
                        verbose=False
                    )
                    
                    annotated = results[0].plot()
                    frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    
                    frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
                    
                    if len(results[0].boxes) > 0:
                        boxes = results[0].boxes
                        num_detections = len(boxes)
                        data = []
                        for i, box in enumerate(boxes):
                            cls = int(box.cls[0])
                            conf = float(box.conf[0])
                            data.append({
                                "Detection": i + 1,
                                "Class": CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else f"Class {cls}",
                                "Confidence": f"{conf:.2%}"
                            })
                        df = pd.DataFrame(data)
                        with stats_placeholder.container():
                            st.success(f"Found {num_detections} detection(s)")
                            st.dataframe(df, use_container_width=True, hide_index=True)
                    else:
                        with stats_placeholder.container():
                            st.info("No helmets detected.")
                    
                    time.sleep(0.07)
                    
                    if stop_stream:
                        st.session_state.streaming = False
                
                cap.release()
                frame_placeholder.empty()
                stats_placeholder.empty()
                st.info("Streaming stopped.")
        else:
            st.info("Click **Start Stream** to begin webcam detection.")

        st.markdown("</div>", unsafe_allow_html=True)


# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**Helmet Detection System**")
st.sidebar.markdown("YOLOv11 Model")

