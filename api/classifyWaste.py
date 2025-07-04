    # api/classifyWaste.py
    # Ini adalah file fungsi Vercel Anda
    # Nama file ini (classifyWaste.py) akan menjadi path fungsi: /api/classifyWaste

    import json
    import base64
    import os
    import numpy as np
    from PIL import Image
    import io
    import tensorflow as tf
    from http.server import BaseHTTPRequestHandler # Digunakan untuk CORS OPTIONS request

    # Path ke model .h5 Anda.
    # Karena model sudah disalin ke folder 'api/', pathnya relatif mudah.
    MODEL_PATH = os.path.join(os.path.dirname(__file__), 'my_waste_classifier_model.h5')

    # Global variable untuk menyimpan model yang sudah dimuat
    # Ini akan mencegah model dimuat ulang di setiap invocation (jika container tetap aktif)
    model = None

    def load_model():
        global model
        if model is None:
            try:
                model = tf.keras.models.load_model(MODEL_PATH)
                print(f"Model berhasil dimuat dari: {MODEL_PATH}")
            except Exception as e:
                print(f"Gagal memuat model dari {MODEL_PATH}: {e}")
                model = None
        return model

    # Panggil saat fungsi pertama kali di-inisialisasi (cold start)
    load_model()

    # Definisikan kelas-kelas output model Anda
    CLASS_NAMES = ['Organik', 'Anorganik']

    def handler(request):
        """
        Handler untuk Vercel Function.
        Menerima objek permintaan (request) dan mengembalikan respons HTTP.
        """
        # Tangani CORS Preflight request (OPTIONS method)
        if request.method == 'OPTIONS':
            return BaseHTTPRequestHandler.send_response(
                None, 204, # No Content
                {
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': 'POST, OPTIONS',
                    'Access-Control-Allow-Headers': 'Content-Type',
                    'Access-Control-Max-Age': '3600'
                }
            )

        if model is None:
            return {
                "statusCode": 500,
                "headers": {
                    "Content-Type": "application/json",
                    "Access-Control-Allow-Origin": "*" # CORS for actual request
                },
                "body": json.dumps({'error': 'Model belum dimuat. Periksa log fungsi.'})
            }

        if request.method != 'POST':
            return {
                "statusCode": 405,
                "headers": {
                    "Content-Type": "application/json",
                    "Access-Control-Allow-Origin": "*"
                },
                "body": json.dumps({'error': 'Metode tidak diizinkan. Hanya POST yang didukung.'})
            }

        try:
            # Ambil data JSON dari body request
            body = json.loads(request.body)
            image_data = body.get('image')

            if not image_data:
                return {
                    "statusCode": 400,
                    "headers": {
                        "Content-Type": "application/json",
                        "Access-Control-Allow-Origin": "*"
                    },
                    "body": json.dumps({'error': 'Permintaan tidak valid. Harap sertakan data gambar base64.'})
                }

            # Hapus prefix "data:image/jpeg;base64," jika ada
            if "base64," in image_data:
                image_data = image_data.split("base64,")[1]

            # Dekode gambar base64 menjadi byte
            image_bytes = base64.b64decode(image_data)
            # Buka gambar menggunakan Pillow
            img = Image.open(io.BytesIO(image_bytes))

            # Pra-pemrosesan gambar (HARUS SESUAI DENGAN SAAT PELATIHAN MODEL)
            img = img.resize((224, 224))
            img_array = np.array(img)
            img_array = img_array / 255.0

            if img_array.shape[-1] == 4:
                img_array = img_array[:, :, :3]
            elif img_array.shape[-1] == 1:
                img_array = np.stack([img_array[:,:,0]]*3, axis=-1)
            elif img_array.shape[-1] != 3:
                return {
                    "statusCode": 400,
                    "headers": {
                        "Content-Type": "application/json",
                        "Access-Control-Allow-Origin": "*"
                    },
                    "body": json.dumps({'error': 'Format gambar tidak didukung atau channel tidak sesuai.'})
                }

            # Tambahkan dimensi batch
            img_array = np.expand_dims(img_array, axis=0)

            # Lakukan prediksi
            predictions = model.predict(img_array)
            predicted_class_index = np.argmax(predictions[0])
            predicted_class_name = CLASS_NAMES[predicted_class_index]
            confidence = float(np.max(predictions[0]))

            return {
                "statusCode": 200,
                "headers": {
                    "Content-Type": "application/json",
                    "Access-Control-Allow-Origin": "*" # Penting untuk CORS
                },
                "body": json.dumps({
                    'prediction': predicted_class_name,
                    'confidence': confidence
                })
            }

        except Exception as e:
            print(f"Kesalahan saat memproses gambar atau prediksi: {e}")
            return {
                "statusCode": 500,
                "headers": {
                    "Content-Type": "application/json",
                    "Access-Control-Allow-Origin": "*"
                },
                "body": json.dumps({'error': f'Terjadi kesalahan saat memproses gambar: {str(e)}'})
            }
    