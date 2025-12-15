import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model, Model
import matplotlib.pyplot as plt
import cv2
import gdown
import os

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Image Classifier Deployment",
    layout="centered"
)

st.title("📸 Skin Cancer Classifier")
st.write("Upload gambar untuk diklasifikasi")

# --- FUNGSI UTILITY --- 

@st.cache_resource
def load_trained_model():
    model_path = 'model_resnet_final.h5'
    
    # Cek apakah file sudah ada, jika tidak, download
    if not os.path.exists(model_path):
        # GANTI ID DI BAWAH DENGAN ID FILE ANDA
        file_id = '1LP3FL15VQ0rkxevLOHA2KJ8CL-0L-0nw'
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, model_path, quiet=False)
    
    model = load_model(model_path)
    return model

# Panggil fungsi
model = load_trained_model()

def process_image(image):
    image = ImageOps.fit(image, (64, 64), Image.Resampling.LANCZOS)
    img_array = np.array(image)
    
    if len(img_array.shape) == 2:
        img_array = np.stack((img_array,)*3, axis=-1)
    
    img_array = np.expand_dims(img_array, axis=0)
    
    img_array = preprocess_input(img_array)
    
    return img_array

def get_gradcam(model, img_array, last_conv_layer_name):
    # Membuat model gradien
    grad_model = Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)[0]
    conv_outputs = conv_outputs[0]
    
    # Global Average Pooling pada gradien
    weights = tf.reduce_mean(grads, axis=(0, 1))
    
    # Kalkulasi CAM
    cam = np.zeros(conv_outputs.shape[:2], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * conv_outputs[:, :, i]

    cam = np.maximum(cam, 0)
    if np.max(cam) != 0:
        cam = cam / np.max(cam)
    
    return cam

# --- LOAD MODEL ---
try:
    model = load_trained_model()
    st.success("Model berhasil dimuat!")
except Exception as e:
    st.error(f"Gagal memuat model. Pastikan file .h5 ada. Error: {e}")
    st.stop()

# --- MAIN INTERFACE ---
uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Tampilkan gambar asli
    image = Image.open(uploaded_file)
    st.image(image, caption='Gambar yang diupload', use_container_width=True)
    
    # Tombol Prediksi
    if st.button('Prediksi'):
        with st.spinner('Sedang memproses...'):
            # Preprocessing
            processed_img = process_image(image)
            
            # Prediksi
            prediction = model.predict(processed_img)
            score = prediction[0][0] # Output sigmoid (0 s.d 1)
            
            # Logic Klasifikasi (Sesuaikan Label Kelas Anda di sini)
            # Asumsi: < 0.5 adalah Kelas 0, > 0.5 adalah Kelas 1
            if score > 0.5:
                label = "Kelas 1 (Positive)" 
                confidence = score
            else:
                label = "Kelas 0 (Negative)"
                confidence = 1 - score
            
            st.write("---")
            st.subheader(f"Hasil Prediksi: **{label}**")
            st.write(f"Confidence: **{confidence*100:.2f}%**")
            
            # --- VISUALISASI GRAD-CAM ---
            st.write("---")
            st.write("### Analisis Grad-CAM")
            st.write("Area merah menunjukkan bagian gambar yang paling berpengaruh terhadap keputusan model.")
            
            try:
                # Nama layer terakhir ResNet50 biasanya 'conv5_block3_out'
                # Jika error, cek model.summary() untuk nama layer yang tepat
                heatmap = get_gradcam(model, processed_img, 'conv5_block3_out')
                
                # Resize heatmap agar sesuai ukuran gambar asli untuk display
                heatmap = cv2.resize(heatmap, (image.size[0], image.size[1]))
                heatmap = np.uint8(255 * heatmap)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                
                # Konversi gambar asli ke array agar bisa digabung
                original_img_cv = np.array(image)
                if len(original_img_cv.shape) == 2: # handle grayscale
                     original_img_cv = cv2.cvtColor(original_img_cv, cv2.COLOR_GRAY2RGB)
                
                # Superimpose
                superimposed_img = cv2.addWeighted(original_img_cv, 0.6, heatmap, 0.4, 0)
                
                st.image(superimposed_img, caption='Grad-CAM Overlay', clamp=True, use_container_width=True)
            
            except Exception as e:
                st.warning(f"Tidak dapat membuat Grad-CAM. Pastikan nama layer benar. Error: {e}")