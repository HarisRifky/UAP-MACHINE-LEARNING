# UAP-MACHINE-LEARNING

# 🍄 Mushroom Classification System - UAP Machine Learning

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0%2B-red)
![PDM](https://img.shields.io/badge/PDM-Package_Manager-purple)

## 📝 Deskripsi Proyek
Proyek ini dikembangkan sebagai tugas **Ujian Akhir Praktikum (UAP) Pembelajaran Mesin**.Tujuan utama dari proyek ini adalah membangun sistem klasifikasi citra otomatis yang mampu mengenali berbagai jenis jamur (*mushrooms*) untuk membantu identifikasi spesies secara cepat dan akurat.

Sistem ini mengimplementasikan pendekatan **Deep Learning** dengan membandingkan tiga arsitektur model berbeda:
1. **Neural Network Base (CNN Custom)** - Dibangun dari awal (*scratch*).
2. **MobileNetV2** - Transfer Learning (Pretrained).
3. **ResNet50V2** - Transfer Learning (Pretrained).

Hasil model terbaik diintegrasikan ke dalam antarmuka website sederhana menggunakan **Streamlit** agar dapat dijalankan secara lokal.

---

## 📂 Dataset & Preprocessing

### Sumber Data
Dataset yang digunakan mencakup **17 Kelas Jamur** dengan format anotasi COCO JSON.
* **Sumber Dataset:** [Roboflow / Kaggle](https://www.kaggle.com/datasets/mathieuduverne/mushroom-classification)
* **Jumlah Data:** [8857 images]
* **Pembagian Data:** Train, Validation, dan Test.

### Tahapan Preprocessing 
Karena struktur dataset menggunakan file JSON (`_annotations.coco.json`) dan bukan folder terpisah per kelas, dilakukan teknik preprocessing khusus:
1.  **JSON Parsing:** Mengubah anotasi JSON menjadi DataFrame (Pandas) untuk memetakan *filename* ke *label*.
2.  **Image Resizing:** Mengubah ukuran citra menjadi **224x224 piksel** sesuai input standar model MobileNet/ResNet.
3.  **Data Augmentation:** Diterapkan pada data *training* untuk mencegah overfitting, meliputi: *Rotation, Width/Height Shift, Shear, Zoom,* dan *Horizontal Flip*.
4.  **Normalization:** Rescaling nilai piksel menjadi rentang 0-1 (1./255).

---

## 🧠 Model yang Digunakan

Sesuai ketentuan UAP, proyek ini menerapkan tiga skenario model:

### 1. Model Base (CNN Custom) 
Model Convolutional Neural Network (CNN) yang dibangun dari awal (*from scratch*) tanpa bobot pretrained.
* **Arsitektur:** Terdiri dari 3 blok Konvolusi (Conv2D + MaxPooling), diikuti oleh Flatten, Dense Layer (512 neuron), Dropout (0.5), dan Output Layer (Softmax).
* **Tujuan:** Sebagai *baseline* untuk melihat kemampuan model mengenali fitur dasar tanpa bantuan pengetahuan sebelumnya.

### 2. MobileNetV2 (Transfer Learning) 
* **Deskripsi:** Model pretrained ringan yang dirancang untuk perangkat mobile.
* **Konfigurasi:** Layer dasar (*base*) dibekukan (*freeze*), ditambahkan *GlobalAveragePooling* dan *Dense Layer* baru untuk klasifikasi 17 kelas jamur.

### 3. ResNet50V2 (Transfer Learning) 
* **Deskripsi:** Model pretrained yang lebih dalam menggunakan *residual connections* untuk menangani *vanishing gradient*.
* **Konfigurasi:** Sama seperti MobileNet, menggunakan teknik *feature extraction* dengan membekukan layer *base* ImageNet.

---

## 📊 Hasil Evaluasi & Analisis Perbandingan

Berikut adalah perbandingan performa ketiga model setelah dilakukan pelatihan selama **15 Epoch**.

### Tabel Perbandingan Performa

| Nama Model | Akurasi (Test) | Loss (Test) | Hasil Analisis |
| :--- | :---: | :---: | :--- |
| **Base CNN** | **51%** | 1.1035 | Model ini memiliki performa paling rendah dibanding model pretrained. Cenderung mengalami *overfitting* karena keterbatasan data latih dan arsitektur yang sederhana. |
| **MobileNetV2** | **71 %** | 0.1139 | Memberikan keseimbangan terbaik antara akurasi dan kecepatan. Training berjalan cepat dan model cukup stabil dalam memprediksi kelas validasi. |
| **ResNet50V2** | **79%** | 0.1205 | Menghasilkan akurasi tertinggi/sebanding dengan MobileNet. Namun, waktu komputasi lebih lama dan ukuran file model lebih besar. |

*(Catatan: Nilai di atas berdasarkan hasil Classification Report pada data Test)* 

### Grafik Evaluasi

## TRAINING MODEL 1: BASE CNN
<img width="981" height="374" alt="cnn1" src="https://github.com/user-attachments/assets/e34c3635-8d2b-445e-8fea-686606d049cd" />
<img width="943" height="855" alt="cnn2" src="https://github.com/user-attachments/assets/578f32c6-6cbd-42ee-89f4-cc3221f41ec8" />


## TRAINING MODEL 2: MOBILENETV2
<img width="990" height="374" alt="mobilenet1" src="https://github.com/user-attachments/assets/34da0845-6dfe-4b8d-9e1f-aad0d47cc86a" />
<img width="943" height="855" alt="mobilenet2" src="https://github.com/user-attachments/assets/8ea26fe7-a1bc-426f-91a4-2d76c54376ee" />

## TRAINING MODEL 3: RESNET50V2
<img width="999" height="374" alt="resnet1" src="https://github.com/user-attachments/assets/5170a15d-f2f7-413a-9552-db215a10030e" />
<img width="943" height="855" alt="resnet2" src="https://github.com/user-attachments/assets/23502e60-7960-4f13-9227-48b28a4d50ea" />


---

## 🚀 Panduan Menjalankan Sistem (Lokal)

Sistem ini dibangun menggunakan Python dan PDM sebagai package manager. Ikuti langkah berikut untuk menjalankannya di komputer Anda.

### 1. Instalasi Dependensi (Menggunakan PDM)
Pastikan Anda sudah menginstall PDM. Jika belum: pip install pdm.

### 2. Struktur Folder
Pastikan struktur folder Anda seperti ini agar aplikasi berjalan lancar:

uap-machine-learning/
├── data/               # Folder Dataset (Wajib ada)
├── models/             # File model (.h5) disimpan di sini
├── app/
│   ├── main.py         # File utama Streamlit
│   └── utils.py        # Logika backend
├── notebooks/          # File training .ipynb
├── pyproject.toml
└── README.md


### 3. Menjalankan Website
Jalankan perintah berikut di terminal:
pdm run streamlit run app/main.py

Website akan otomatis terbuka di browser pada alamat http://localhost:8501.

👤 Identitas Pengembang
Nama: *Haris Rifky Juliantoro*

NIM: *202210370311421*

Kelas: *Machine learning A*

Laboratorium Informatika UMM
