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

// URL endpoint API Flask Anda
// Saat development, ini mungkin 'http://localhost:5000/predict'
// Saat deployment, ini harus URL publik dari backend Anda (misalnya, 'https://your-backend-app.herokuapp.com/predict')
const API_URL = 'https://vercel.com/rafi-achmadfrs-projects/pilah-ai/26Le2G6vYrokEq1SwCGzQM22UUWC'; // Ganti ini saat deployment!

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
        // Meminta akses kamera pengguna
        stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } }); // 'environment' untuk kamera belakang di perangkat seluler
        cameraFeed.srcObject = stream;
        cameraFeed.classList.remove('hidden'); // Tampilkan video feed
        captureCanvas.classList.add('hidden'); // Sembunyikan canvas
        noCameraMessage.classList.add('hidden'); // Sembunyikan pesan "kamera tidak tersedia"
        captureButton.classList.remove('hidden'); // Tampilkan tombol ambil gambar
        retakeButton.classList.add('hidden'); // Sembunyikan tombol ambil ulang
        predictionText.textContent = 'Menunggu gambar...';
        confidenceText.textContent = 'Kepercayaan: -';
    } catch (err) {
        console.error("Error mengakses kamera: ", err);
        cameraFeed.classList.add('hidden');
        noCameraMessage.classList.remove('hidden'); // Tampilkan pesan "kamera tidak tersedia"
        captureButton.classList.add('hidden'); // Sembunyikan tombol ambil gambar
        retakeButton.classList.add('hidden'); // Sembunyikan tombol ambil ulang
        showMessage('Gagal mengakses kamera. Pastikan Anda memberikan izin akses dan tidak ada aplikasi lain yang menggunakan kamera.');
    }
}

// Event listener untuk tombol "Ambil Gambar & Klasifikasi"
captureButton.addEventListener('click', async () => {
    // Sembunyikan video feed dan tampilkan canvas
    cameraFeed.classList.add('hidden');
    captureCanvas.classList.remove('hidden');
    captureButton.classList.add('hidden'); // Sembunyikan tombol ambil gambar
    retakeButton.classList.remove('hidden'); // Tampilkan tombol ambil ulang

    // Hentikan stream kamera untuk menghemat daya
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
    }

    // Pastikan video sudah dimuat dan memiliki dimensi
    if (cameraFeed.readyState === cameraFeed.HAVE_ENOUGH_DATA) {
        // Set ukuran canvas sesuai dengan ukuran video
        captureCanvas.width = cameraFeed.videoWidth;
        captureCanvas.height = cameraFeed.videoHeight;

        const context = captureCanvas.getContext('2d');
        // Gambar frame video saat ini ke canvas
        // Menggunakan transform untuk membalik gambar jika video juga dibalik
        context.translate(captureCanvas.width, 0);
        context.scale(-1, 1);
        context.drawImage(cameraFeed, 0, 0, captureCanvas.width, captureCanvas.height);
        context.setTransform(1, 0, 0, 1, 0, 0); // Reset transform

        // Konversi gambar di canvas ke base64 JPEG
        const imageData = captureCanvas.toDataURL('image/jpeg', 0.9); // 0.9 adalah kualitas JPEG

        // Tampilkan loading indicator
        loadingIndicator.classList.remove('hidden');
        predictionText.textContent = 'Mengklasifikasi...';
        confidenceText.textContent = '';

        try {
            // Kirim gambar base64 ke API Flask
            const response = await fetch(API_URL, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ image: imageData }),
            });

            if (!response.ok) {
                // Tangani error HTTP (misalnya, 404, 500)
                const errorData = await response.json();
                throw new Error(`HTTP error! Status: ${response.status}, Message: ${errorData.error || response.statusText}`);
            }

            const result = await response.json();

            // Tampilkan hasil prediksi
            predictionText.textContent = `Klasifikasi: ${result.prediction}`;
            confidenceText.textContent = `Kepercayaan: ${(result.confidence * 100).toFixed(2)}%`;

        } catch (error) {
            console.error('Error saat mengirim gambar atau menerima prediksi:', error);
            predictionText.textContent = 'Gagal mengklasifikasi.';
            confidenceText.textContent = 'Terjadi kesalahan.';
            showMessage(`Gagal mengklasifikasi gambar: ${error.message}. Pastikan backend berjalan.`);
        } finally {
            // Sembunyikan loading indicator
            loadingIndicator.classList.add('hidden');
        }
    } else {
        showMessage('Video feed belum siap. Coba lagi.');
        startCamera(); // Coba mulai kamera lagi jika belum siap
    }
});

// Event listener untuk tombol "Ambil Ulang"
retakeButton.addEventListener('click', () => {
    startCamera(); // Mulai ulang kamera
});

// Mulai kamera saat halaman dimuat
window.onload = startCamera;
