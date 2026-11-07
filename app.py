"""Start with: python -m streamlit run app.py"""
import hashlib
import importlib.util
from pathlib import Path

from PIL import UnidentifiedImageError
import streamlit as st

from src.inference import detect_image

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title='RoadVision AI', page_icon='🚗', layout='wide')
st.title('RoadVision AI')
st.write('Upload a road image to detect vehicles with your trained model.')

checkpoints = []
if (ROOT / 'models/best.pt').is_file():
    checkpoints.append(ROOT / 'models/best.pt')
checkpoints += sorted((ROOT / 'reports').glob('*/weights/best.pt'),
                      key=lambda p: p.stat().st_mtime, reverse=True)
with st.sidebar:
    st.header('Detection settings')
    weights = st.selectbox('Trained model', checkpoints,
                          format_func=lambda p: str(p.relative_to(ROOT))) if checkpoints else None
    confidence = st.slider('Minimum confidence', 0.05, 0.95, 0.25, 0.05)
    device_label = st.selectbox('Processor', ['Automatic', 'GPU 0', 'CPU'])
    device = {'Automatic': '', 'GPU 0': '0', 'CPU': 'cpu'}[device_label]
    st.caption('Lower confidence shows more predictions. CPU inference can be slower.')

ready = True
if weights is None:
    st.info('Train the model in the notebook, then run its Streamlit export cell. Your trained model will appear here after refreshing.')
    ready = False
if not (ROOT / 'third_party/yolov9/detect_dual.py').is_file():
    st.info('Run the notebook’s YOLOv9 setup cells on this machine before detecting vehicles.')
    ready = False
if importlib.util.find_spec('torch') is None:
    st.info('Run the app in your PyTorch environment on Vast.ai.')
    ready = False

upload = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png', 'webp'])
if upload is not None:
    if upload.size > 20 * 1024 * 1024:
        st.error('Choose an image smaller than 20 MB.')
        st.stop()
    data = upload.getvalue()
    signature = (hashlib.sha256(data).hexdigest(), str(weights),
                 weights.stat().st_mtime_ns if weights else None, confidence, device)
    if st.session_state.get('signature') != signature:
        st.session_state.pop('prediction', None)
        st.session_state['signature'] = signature
    left, right = st.columns(2)
    with left:
        st.subheader('Original')
        try:
            st.image(data)
        except Exception:
            st.error('This file could not be displayed as an image.')
            st.stop()
    if st.button('Detect vehicles', type='primary', disabled=not ready):
        try:
            with st.spinner('Detecting vehicles…'):
                st.session_state['prediction'] = detect_image(
                    data, weights, ROOT / 'third_party/yolov9', confidence, device)
        except (RuntimeError, OSError, ValueError, UnidentifiedImageError) as exc:
            st.error('Detection could not finish. Check your model and runtime setup.')
            with st.expander('Error details'):
                st.code(str(exc))
    if 'prediction' in st.session_state:
        annotated, labels = st.session_state['prediction']
        with right:
            st.subheader('Detections')
            st.image(annotated)
            count = sum(bool(line.strip()) for line in labels.splitlines())
            st.metric('Objects detected', count)
            if not count:
                st.info('No objects found at this confidence. Try another image or lower the threshold.')
            st.download_button('Download annotated image', annotated,
                               'roadvision_prediction.jpg', 'image/jpeg')
            st.download_button('Download detections', labels,
                               'roadvision_detections.txt', 'text/plain')
            st.caption('Detection rows: class ID, normalized center x/y, width/height, confidence.')
else:
    st.session_state.pop('prediction', None)
    st.session_state.pop('signature', None)
    st.caption('Supports JPG, PNG, and WebP. Results include bounding boxes and confidence scores.')
