// main.js

// Dapatkan elemen-elemen DOM
const cameraFeed = document.getElementById('cameraFeed');
const captureCanvas = document.getElementById('captureCanvas');
const captureButton = document.getElementById('captureButton');
const retakeButton = document.getElementById('retakeButton');
const predictionText = document.getElementById('predictionText');
const confidenceText = document.getElementById('confidenceText');
const loadingIndicator = document.getElementById('loadingIndicator');
const noCameraMessage = document.getElementById('noCameraMessage');
const messageBox = document.getElementById('messageBox');
const messageText = document.getElementById('messageText');
const messageBoxClose = document.getElementById('messageBoxClose');

let stream; // Variabel untuk menyimpan stream kamera
let model; // Variabel untuk menyimpan model TensorFlow.js

// Path ke model TensorFlow.js Anda setelah dikonversi
// Setelah Anda mengkonversi model .h5 Anda, letakkan folder hasilnya di 'frontend/models/my_waste_classifier_tfjs/'
// URL ini akan menjadi relatif terhadap file index.html

const MODEL_URL = './models/tensorflow-js/model.json'; // PASTIKAN PATH INI BENAR!

// Definisikan kelas-kelas output model Anda (HARUS SESUAI DENGAN MODEL ANDA)
const CLASS_NAMES = ['Organik', 'Anorganik']; // Sesuaikan ini dengan kelas model Anda

// Fungsi untuk menampilkan pesan di modal kustom
function showMessage(message) {
    messageText.textContent = message;
    messageBox.classList.remove('hidden');
}

// Fungsi untuk menyembunyikan pesan modal
messageBoxClose.addEventListener('click', () => {
    messageBox.classList.add('hidden');
});

// Fungsi untuk memulai stream kamera
async function startCamera() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
        cameraFeed.srcObject = stream;
        cameraFeed.classList.remove('hidden');
        captureCanvas.classList.add('hidden');
        noCameraMessage.classList.add('hidden');
        captureButton.classList.remove('hidden');
        retakeButton.classList.add('hidden');
        predictionText.textContent = 'Menunggu gambar...';
        confidenceText.textContent = 'Kepercayaan: -';
    } catch (err) {
        console.error("Error mengakses kamera: ", err);
        cameraFeed.classList.add('hidden');
        noCameraMessage.classList.remove('hidden');
        captureButton.classList.add('hidden');
        retakeButton.classList.add('hidden');
        showMessage('Gagal mengakses kamera. Pastikan Anda memberikan izin akses dan tidak ada aplikasi lain yang menggunakan kamera.');
    }
}

// Fungsi untuk memuat model TensorFlow.js
async function loadTFJSModel() {
    predictionText.textContent = 'Memuat model...';
    loadingIndicator.classList.remove('hidden');
    try {
        // tf.loadGraphModel digunakan untuk SavedModel atau Keras Model yang dikonversi
        model = await tf.loadLayersModel(MODEL_URL);
        console.log('Model TensorFlow.js berhasil dimuat!');
        predictionText.textContent = 'Model siap. Ambil gambar!';
    } catch (error) {
        console.error('Gagal memuat model TensorFlow.js:', error);
        showMessage(`Gagal memuat model: ${error.message}. Pastikan model sudah dikonversi dan pathnya benar.`);
        predictionText.textContent = 'Gagal memuat model.';
    } finally {
        loadingIndicator.classList.add('hidden');
    }
}

// Event listener untuk tombol "Ambil Gambar & Klasifikasi"
captureButton.addEventListener('click', async () => {
    if (!model) {
        showMessage('Model belum dimuat. Harap tunggu atau refresh halaman.');
        return;
    }

    // Sembunyikan video feed dan tampilkan canvas
    cameraFeed.classList.add('hidden');
    captureCanvas.classList.remove('hidden');
    captureButton.classList.add('hidden');
    retakeButton.classList.remove('hidden');

    // Hentikan stream kamera untuk menghemat daya
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
    }

    if (cameraFeed.readyState === cameraFeed.HAVE_ENOUGH_DATA) {
        captureCanvas.width = cameraFeed.videoWidth;
        captureCanvas.height = cameraFeed.videoHeight;
        const context = captureCanvas.getContext('2d');

        // Gambar frame video saat ini ke canvas (dengan flip horizontal)
        context.translate(captureCanvas.width, 0);
        context.scale(-1, 1);
        context.drawImage(cameraFeed, 0, 0, captureCanvas.width, captureCanvas.height);
        context.setTransform(1, 0, 0, 1, 0, 0); // Reset transform

        // Tampilkan loading indicator
        predictionText.textContent = 'Mengklasifikasi...';
        confidenceText.textContent = '';
        loadingIndicator.classList.remove('hidden');

        try {
            // Pra-pemrosesan gambar untuk model TF.js
            const imgTensor = tf.browser.fromPixels(captureCanvas)
                .resizeNearestNeighbor([224, 224]) // Ubah ukuran ke 224x224
                .toFloat()
                .div(tf.scalar(255.0)) // Normalisasi ke 0-1
                .expandDims(); // Tambahkan dimensi batch (bentuk menjadi [1, 224, 224, 3])

            // Lakukan prediksi
            const predictions = model.predict(imgTensor);
            const scores = await predictions.data(); // Dapatkan nilai probabilitas
            const predictedClassIndex = scores.indexOf(Math.max(...scores)); // Ambil indeks probabilitas tertinggi
            const predictedClassName = CLASS_NAMES[predictedClassIndex];
            const confidence = scores[predictedClassIndex];

            // Tampilkan hasil prediksi
            predictionText.textContent = `Klasifikasi: ${predictedClassName}`;
            confidenceText.textContent = `Kepercayaan: ${(confidence * 100).toFixed(2)}%`;

            // Bersihkan tensor dari memori GPU
            imgTensor.dispose();
            predictions.dispose();

        } catch (error) {
            console.error('Error saat klasifikasi:', error);
            predictionText.textContent = 'Gagal mengklasifikasi.';
            confidenceText.textContent = 'Terjadi kesalahan.';
            showMessage(`Gagal mengklasifikasi gambar: ${error.message}`);
        } finally {
            loadingIndicator.classList.add('hidden');
        }
    } else {
        showMessage('Video feed belum siap. Coba lagi.');
        startCamera();
    }
});

// Event listener untuk tombol "Ambil Ulang"
retakeButton.addEventListener('click', () => {
    startCamera(); // Mulai ulang kamera
});

// Mulai kamera dan muat model saat halaman dimuat
window.onload = async () => {
    await startCamera();
    await loadTFJSModel(); // Muat model setelah kamera siap
};
