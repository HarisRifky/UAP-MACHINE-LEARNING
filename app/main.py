import streamlit as st
import pandas as pd
import time
from PIL import Image
from utils import load_model_file, preprocess_image, predict_image, CLASS_NAMES

# 1. Konfigurasi Halaman 
st.set_page_config(
    page_title="Mushroom Classifier AI",
    page_icon="🍄",
    layout="wide"
)


# 2. CSS 
st.markdown("""
    <style>
    /* Mengatur warna background utama menjadi gelap */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    
    /* Mengatur sidebar agar warnanya sedikit berbeda */
    [data-testid="stSidebar"] {
        background-color: #262730;
    }
    
    /* Tombol Custom */
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        border-radius: 8px;
        height: 3em;
        font-weight: bold;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #FF2B2B;
        box-shadow: 0px 4px 15px rgba(255, 75, 75, 0.4);
    }
    
    /* Card Hasil Prediksi */
    .metric-card {
        background: linear-gradient(135deg, #1E1E1E 0%, #2D2D2D 100%);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #3D3D3D;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        text-align: center;
        margin-bottom: 20px;
    }
    
    /* Teks judul dalam card */
    .metric-card h2 {
        color: #FFFFFF;
        margin: 0;
        font-size: 1.8rem;
    }
    
    /* Teks deskripsi kecil */
    .metric-card p {
        color: #A0A0A0;
        margin: 5px 0;
    }
    
    /* Angka persentase */
    .metric-card h1 {
        color: #00CC96; /* Warna hijau neon untuk confidence */
        font-size: 3.5rem;
        margin: 10px 0;
        text-shadow: 0 0 10px rgba(0, 204, 150, 0.3);
    }
    </style>
    """, unsafe_allow_html=True)

# 3. Sidebar: Navigasi & Pemilihan Model
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2917/2917995.png", width=100)
    st.title("Panel Kontrol")
    
    st.markdown("---")
    
    # Dropdown Pemilihan Model 
    st.subheader("Pilih Model AI:")
    model_option = st.selectbox(
        "Silakan pilih model untuk prediksi:",
        ("Model Base CNN", "Model MobileNetV2", "Model ResNet50V2")
    )
    
    # Mapping nama pilihan ke nama file
    model_mapping = {
        "Model Base CNN": "model_base_cnn",
        "Model MobileNetV2": "model_mobilenetv2",  
        "Model ResNet50V2": "model_resnet50v2"       
    }
    
    st.info(f"Model Aktif: **{model_option}**")
    st.markdown("---")
    st.write("Dibuat untuk **UAP Machine Learning**")

# 4. Konten Utama
st.title("🍄 Klasifikasi Jenis Jamur")
st.markdown("Sistem cerdas untuk mendeteksi jenis jamur dari gambar menggunakan **Deep Learning**.")

# Layout Kolom 
col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("1. Upload Gambar")
    uploaded_file = st.file_uploader("Pilih gambar jamur (JPG/PNG)...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar yang diupload", use_column_width=True)
        predict_btn = st.button("🔍 Identifikasi Jamur")
    else:
        st.info("Silakan upload gambar untuk memulai.")

with col2:
    st.subheader("2. Hasil Analisis")
    
    if uploaded_file is not None and 'predict_btn' in locals() and predict_btn:
        
        # Loading Animation
        with st.spinner(f'Sedang memproses dengan {model_option}...'):
            # Load Model
            selected_model_name = model_mapping[model_option]
            model = load_model_file(selected_model_name)
            
            if model is None:
                st.error(f"Gagal memuat model: {selected_model_name}. Pastikan file .h5 ada di folder 'models/'.")
            else:
                # Preprocessing & Prediksi
                processed_img = preprocess_image(image)
                result = predict_image(model, processed_img)
                
                # Simulasi waktu tunggu 
                time.sleep(1)

                # Tampilkan Hasil Utama 
                st.markdown(f"""
                <div class="metric-card">
                    <h2 style="margin:0; color:#333;">{result['class']}</h2>
                    <p style="color:gray;">Prediksi Utama</p>
                    <h1 style="color:#FF4B4B; font-size: 3rem;">{result['confidence']*100:.1f}%</h1>
                    <p style="color:gray;">Tingkat Keyakinan</p>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("### 📊 Statistik Probabilitas")
                
                # Membuat DataFrame untuk Bar Chart
                probs_df = pd.DataFrame({
                    'Jenis Jamur': CLASS_NAMES,
                    'Probabilitas': result['all_probabilities']
                })
                
                # Mengurutkan dan mengambil Top 5
                probs_df = probs_df.sort_values(by='Probabilitas', ascending=False).head(5)
                
                # Menampilkan Bar Chart Interaktif
                st.bar_chart(
                    probs_df.set_index('Jenis Jamur'),
                    color="#FF4B4B"
                )
                
                # Analisis Sederhana
                st.success(f"Sistem memprediksi gambar ini sebagai **{result['class']}** dengan akurasi **{result['confidence']:.2%}**.")

    elif uploaded_file is None:
        st.write("👈 Menunggu input gambar dari pengguna...")

# Footer
st.markdown("---")
st.caption("© 2025 Laboratorium Informatika UMM | Modul Pembelajaran Mesin")