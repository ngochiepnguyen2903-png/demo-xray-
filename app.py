import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

st.set_page_config(page_title="X-Ray Analyzer Demo", page_icon="🩻")
st.title("🩻 Demo Quét Ảnh X-Quang ngực")

@st.cache_resource
def get_model():
       return load_model('file5-densenet121') 

model = get_model()

def analyze_xray(image):
    img = image.convert('RGB')
    img = img.resize((224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_array)
    confidence = float(predictions[0][0])
    
    if confidence > 0.5:
        return "Cảnh báo: Có dấu hiệu bất thường", confidence
    else:
        return "Bình thường", (1 - confidence)

uploaded_file = st.file_uploader("Chọn ảnh X-quang", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Ảnh chụp đã tải lên", use_container_width=True)
    
    if st.button("Phân tích ảnh", type="primary"):
        with st.spinner("AI đang quét..."):
            result_text, conf_score = analyze_xray(image)
        st.subheader(f"Kết quả: {result_text} (Độ tự tin: {conf_score:.2%})")