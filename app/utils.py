import tensorflow as tf
import numpy as np
from PIL import Image

# ==========================================
# KONFIGURASI KELAS 
# ==========================================
CLASS_NAMES = [
    'Auricularia auricula', 'Boletus', 'Cantharellus cibarius', 'Clitocybe maxima', 'Collybia albuminosa', 
    'Coprinus comatus', 'Cordyceps militaris', 'Dictyophora indusiate', 'Flammulina velutiper', 'Hericium erinaceus', 
    'Hypsizygus marmoreus', 'Lentinus edodes', 'Morchella esculenta', 'Pleurotus citrinopileatus', 'Pleurotus cystidiosus', 
    'Pleurotus eryngii', 'Pleurotus ostreatus'
]

def load_model_file(model_name):
    """
    Memuat model .h5 berdasarkan nama yang dipilih user.
    Menggunakan cache resource agar tidak berat saat reload.
    """
    model_path = f"models/{model_name}.h5"
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        return None

def preprocess_image(image):
    """
    Mengubah gambar upload menjadi format yang bisa dibaca model.
    Target: (224, 224), Normalisasi 1./255
    """
    image = image.resize((224, 224))
    image_array = np.array(image)
    
    if image_array.shape[-1] == 4:
        image_array = image_array[..., :3]
        
    # Normalisasi 
    image_array = image_array / 255.0
    
    # Tambah dimensi batch
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def predict_image(model, processed_image):
    """
    Melakukan prediksi dan mengembalikan probabilitas tertinggi.
    """
    predictions = model.predict(processed_image)
    
    # Ambil index 
    predicted_index = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    
    predicted_class = CLASS_NAMES[predicted_index]
    
    # Mengembalikan dictionary 
    result = {
        'class': predicted_class,
        'confidence': confidence,
        'all_probabilities': predictions[0]
    }
    return result