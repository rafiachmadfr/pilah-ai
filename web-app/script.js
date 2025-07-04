const webcamElement = document.getElementById('webcam');
const captureButton = document.getElementById('captureButton');
const predictionElement = document.getElementById('prediction');
const canvasElement = document.createElement('canvas'); // Buat canvas secara dinamis atau pastikan ada di HTML tapi display:none
const context = canvasElement.getContext('2d');

// --- PENTING: Sesuaikan URL API ini dengan alamat server Flask Anda ---
// Jika Anda menjalankannya secara lokal, ini sudah benar.
// Jika di-deploy ke server lain, ganti 'localhost' dengan IP atau domain server Anda.
const API_URL = 'http://127.0.0.1:5000/predict'; 

// Fungsi untuk menyiapkan akses kamera webcam
async function setupWebcam() {
    try {
        // Meminta akses video dari perangkat pengguna
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        webcamElement.srcObject = stream; // Menghubungkan stream ke elemen video HTML

        // Menunggu sampai data video dimuat untuk mendapatkan dimensi asli video
        webcamElement.addEventListener('loadeddata', () => {
            // Menyesuaikan ukuran canvas dengan dimensi video
            canvasElement.width = webcamElement.videoWidth;
            canvasElement.height = webcamElement.videoHeight;
            console.log("Webcam loaded. Video dimensions:", webcamElement.videoWidth, "x", webcamElement.videoHeight);
        });
    } catch (error) {
        console.error("Error accessing webcam:", error);
        predictionElement.innerText = "Gagal mengakses kamera. Pastikan Anda memberikan izin.";
        alert("Gagal mengakses kamera. Pastikan Anda memberikan izin dan menggunakan HTTPS/localhost.");
    }
}

// Fungsi untuk mengambil gambar dari webcam dan mengirimkannya ke backend
async function sendImageToBackend() {
    predictionElement.innerText = "Mengambil gambar dan memproses...";
    
    // Menggambar frame video saat ini ke elemen canvas
    context.drawImage(webcamElement, 0, 0, canvasElement.width, canvasElement.height);

    // Mengambil gambar dari canvas sebagai Blob (Binary Large Object)
    // Blob lebih efisien untuk transfer gambar daripada Base64
    canvasElement.toBlob(async (blob) => {
        if (!blob) {
            predictionElement.innerText = "Gagal mengambil gambar dari kamera.";
            console.error("Canvas to Blob failed.");
            return;
        }

        // Membuat FormData untuk mengirim file melalui POST request
        const formData = new FormData();
        // 'file' adalah nama field yang diharapkan oleh backend Flask Anda (request.files['file'])
        formData.append('file', blob, 'image.jpg'); 

        try {
            // Mengirim gambar ke API backend menggunakan fetch API
            const response = await fetch(API_URL, {
                method: 'POST',
                body: formData,
            });

            // Memeriksa apakah respons dari server OK (status 200-299)
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(`HTTP error! status: ${response.status}, message: ${errorData.error || 'Unknown error'}`);
            }

            // Memparsing respons JSON dari server
            const result = await response.json();
            console.log("Prediction result:", result);

            // Menampilkan hasil prediksi di halaman web
            predictionElement.innerText = `Hasil Klasifikasi: ${result.predicted_class} (${result.confidence}%)`;

        } catch (error) {
            console.error("Error sending image to backend or receiving response:", error);
            predictionElement.innerText = `Terjadi kesalahan saat klasifikasi: ${error.message}`;
        }
    }, 'image/jpeg', 0.9); // Format gambar (JPEG) dan kualitas (0.9)
}

// Event Listener untuk tombol "Ambil Gambar"
captureButton.addEventListener('click', sendImageToBackend);

// Inisialisasi: Panggil fungsi setupWebcam saat halaman web selesai dimuat
window.onload = setupWebcam;