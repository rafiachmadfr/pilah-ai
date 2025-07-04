# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
import os

# Inisialisasi aplikasi Flask
app = Flask(__name__)
# Mengaktifkan CORS untuk mengizinkan permintaan dari frontend (penting untuk deployment)
CORS(app)

# Path ke model .h5 Anda.
# Pastikan path ini benar relatif terhadap lokasi file app.py atau gunakan path absolut.
# Dalam struktur folder yang kita sepakati, model berada di 'nama-proyek-klasifikasi-sampah/python-model/saved_models/'
# Jadi, dari 'backend/app.py', kita perlu naik dua level dan masuk ke 'python-model/saved_models/'
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'python-model', 'models', 'klasifikasi_sampah_final_v1.h5')

# Muat model TensorFlow/Keras
# Model akan dimuat sekali saat aplikasi dimulai untuk efisiensi
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    # Cetak ringkasan model untuk memastikan model berhasil dimuat
    model.summary()
    print(f"Model berhasil dimuat dari: {MODEL_PATH}")
except Exception as e:
    print(f"Gagal memuat model: {e}")
    model = None # Set model ke None jika gagal dimuat

# Definisikan kelas-kelas output model Anda
# Sesuaikan dengan urutan output dari model Anda (misalnya, 0 untuk organik, 1 untuk anorganik)
CLASS_NAMES = ['Organik', 'Anorganik'] # Sesuaikan ini dengan kelas model Anda

@app.route('/')
def home():
    """Rute dasar untuk menguji apakah server berjalan."""
    return "API Klasifikasi Sampah Berjalan!"

@app.route('/predict', methods=['POST'])

def predict():
    """
    Rute untuk menerima gambar, melakukan klasifikasi, dan mengembalikan hasilnya.
    Gambar diharapkan dalam format base64 di body request JSON.
    """
    if model is None:
        return jsonify({'error': 'Model belum dimuat. Periksa log server.'}), 500

    if not request.json or 'image' not in request.json:
        return jsonify({'error': 'Permintaan tidak valid. Harap sertakan data gambar base64.'}), 400

    try:
        # Ambil data gambar base64 dari request JSON
        image_data = request.json['image']
        # Hapus prefix "data:image/jpeg;base64," jika ada
        if "base64," in image_data:
            image_data = image_data.split("base64,")[1]

        # Dekode gambar base64 menjadi byte
        image_bytes = base64.b64decode(image_data)
        # Buka gambar menggunakan Pillow
        img = Image.open(io.BytesIO(image_bytes))

        # Pra-pemrosesan gambar
        # Pastikan ukuran dan format input sesuai dengan model Anda
        img = img.resize((224, 224)) # Ubah ukuran gambar ke 224x224
        img_array = np.array(img)   # Konversi ke NumPy array

        # Normalisasi gambar jika model Anda mengharapkannya (misalnya, nilai piksel 0-1)
        # Jika model dilatih dengan gambar dinormalisasi ke [0, 1], gunakan:
        img_array = img_array / 255.0

        # Jika model Anda mengharapkan 3 channel (RGB) dan gambar mungkin grayscale, pastikan:
        if img_array.shape[-1] == 4: # Jika ada alpha channel (RGBA), konversi ke RGB
            img_array = img_array[:, :, :3]
        elif img_array.shape[-1] == 1: # Jika grayscale, konversi ke RGB (duplikasi channel)
            img_array = np.stack([img_array[:,:,0]]*3, axis=-1)
        elif img_array.shape[-1] != 3: # Jika bukan 3 channel, ini mungkin masalah
            return jsonify({'error': 'Format gambar tidak didukung atau channel tidak sesuai.'}), 400

        # Tambahkan dimensi batch (model mengharapkan input dalam bentuk batch)
        img_array = np.expand_dims(img_array, axis=0) # Bentuk menjadi (1, 224, 224, 3)

        # Lakukan prediksi
        predictions = model.predict(img_array)
        # Ambil indeks kelas dengan probabilitas tertinggi
        predicted_class_index = np.argmax(predictions[0])
        # Ambil nama kelas
        predicted_class_name = CLASS_NAMES[predicted_class_index]
        # Ambil probabilitas (opsional)
        confidence = float(np.max(predictions[0]))

        # Kembalikan hasil prediksi
        return jsonify({
            'prediction': predicted_class_name,
            'confidence': confidence
        })

    except Exception as e:
        print(f"Kesalahan saat memproses gambar atau prediksi: {e}")
        return jsonify({'error': f'Terjadi kesalahan saat memproses gambar: {str(e)}'}), 500

# Jalankan aplikasi Flask
# Host 0.0.0.0 agar dapat diakses dari luar (penting untuk deployment)
# Debug=True hanya untuk pengembangan, set ke False untuk produksi
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
