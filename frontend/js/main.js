// frontend/js/main.js

// Dapatkan elemen-elemen DOM
const cameraFeed = document.getElementById('cameraFeed');
const captureCanvas = document.getElementById('captureCanvas'); // Untuk jepretan kamera
const uploadedPreviewCanvas = document.getElementById('uploadedPreviewCanvas'); // Untuk pratinjau upload
const contextCapture = captureCanvas.getContext('2d');
// const contextUpload = uploadedPreviewCanvas.getContext('2d'); // Tidak lagi diperlukan konteks terpisah

const captureButton = document.getElementById('captureButton');
const retakeButton = document.getElementById('retakeButton');
const cameraToggleButton = document.getElementById('cameraToggleButton'); // Tombol On/Off Kamera
const imageUpload = document.getElementById('imageUpload'); // Input file
const uploadButton = document.getElementById('uploadButton'); // Tombol upload

const toggleCameraModeButton = document.getElementById('toggleCameraModeButton');
const toggleUploadModeButton = document.getElementById('toggleUploadModeButton');
const cameraModeSection = document.getElementById('cameraModeSection');
const uploadModeSection = document.getElementById('uploadModeSection');

const predictionText = document.getElementById('predictionText');
const loadingIndicator = document.getElementById('loadingIndicator');
const noCameraMessage = document.getElementById('noCameraMessage');
const messageBox = document.getElementById('messageBox');
const messageText = document.getElementById('messageText');
const messageBoxClose = document.getElementById('messageBoxClose');

let stream = null; // Variabel untuk menyimpan stream kamera
let currentMode = 'camera'; // Mode aktif saat ini: 'camera' atau 'upload'
let isCameraOn = false; // Status kamera
let lastCapturedImageBase64 = null; // Untuk menyimpan gambar terakhir yang diambil/diunggah

// URL endpoint API Flask Anda (untuk lokal)
const API_URL = 'http://127.0.0.1:5000/predict'; 

// Fungsi untuk menampilkan pesan di modal kustom
function showMessage(message) {
    messageText.textContent = message;
    messageBox.classList.remove('hidden');
}

// Fungsi untuk menyembunyikan pesan modal
if (messageBoxClose) {
    messageBoxClose.addEventListener('click', () => {
        messageBox.classList.add('hidden');
    });
}

// Fungsi untuk menampilkan image (baik dari webcam atau upload) di canvas tertentu
function displayImageOnCanvas(base64Data, canvasElement) {
    const img = new Image();
    img.onload = () => {
        console.log("DEBUG: Gambar dimuat untuk pratinjau. Dimensi asli:", img.naturalWidth, "x", img.naturalHeight);

        // Atur dimensi buffer gambar internal canvas ke dimensi asli gambar
        canvasElement.width = img.naturalWidth;
        canvasElement.height = img.naturalHeight;

        const ctx = canvasElement.getContext('2d');
        ctx.clearRect(0, 0, canvasElement.width, canvasElement.height); // Bersihkan canvas
        
        // Gambar gambar pada dimensi aslinya ke canvas
        ctx.drawImage(img, 0, 0, canvasElement.width, canvasElement.height);
        console.log("DEBUG: Gambar digambar ke canvas.");
        
        canvasElement.classList.remove('hidden');
        // Pastikan elemen canvas itu sendiri ditampilkan sebagai blok dan di tengah oleh CSS
        canvasElement.style.display = 'block'; 
        canvasElement.style.margin = '0 auto'; 

        // Sembunyikan feed kamera jika gambar ditampilkan (khusus mode kamera)
        if (currentMode === 'camera' && cameraFeed && !cameraFeed.classList.contains('hidden')) {
            cameraFeed.classList.add('hidden');
        }
    };
    img.onerror = (error) => {
        console.error("ERROR: Gagal memuat gambar untuk pratinjau:", error);
        showMessage("Gagal menampilkan pratinjau gambar.");
        canvasElement.classList.add('hidden'); // Sembunyikan canvas jika gagal
    };
    img.src = base64Data;
}


// Fungsi untuk memulai stream kamera
async function startCamera() {
    if (isCameraOn) {
        console.log("DEBUG: Kamera sudah menyala.");
        return;
    }

    // Reset UI dan sembunyikan semua pesan
    predictionText.textContent = 'Menunggu gambar...';
    loadingIndicator.classList.add('hidden');
    messageBox.classList.add('hidden'); // Sembunyikan message box jika terbuka
    
    // Sembunyikan canvas yang mungkin menampilkan gambar sebelumnya
    captureCanvas.classList.add('hidden');
    uploadedPreviewCanvas.classList.add('hidden');
    lastCapturedImageBase64 = null; // Hapus gambar yang disimpan

    console.log("DEBUG: Memulai kamera...");
    try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true }); 
        cameraFeed.srcObject = stream;
        cameraFeed.classList.remove('hidden'); // Tampilkan video feed
        noCameraMessage.classList.add('hidden'); // Sembunyikan pesan "kamera tidak tersedia"
        
        isCameraOn = true;
        cameraToggleButton.textContent = 'Matikan Kamera'; // Ubah teks tombol
        captureButton.classList.remove('hidden'); // Tampilkan tombol ambil gambar
        retakeButton.classList.add('hidden'); // Sembunyikan tombol ambil ulang (karena sudah live feed)

        cameraFeed.addEventListener('loadeddata', () => {
            // DEBUG: Log dimensi video setelah dimuat
            console.log("DEBUG: Video loadeddata event fired. Dimensions:", cameraFeed.videoWidth, "x", cameraFeed.videoHeight);
            captureCanvas.width = cameraFeed.videoWidth;
            captureCanvas.height = cameraFeed.videoHeight;
        }, { once: true });

    } catch (err) {
        console.error("ERROR: Gagal mengakses kamera:", err);
        cameraFeed.classList.add('hidden');
        noCameraMessage.classList.remove('hidden'); // Tampilkan pesan "kamera tidak tersedia"
        cameraToggleButton.textContent = 'Nyalakan Kamera'; // Kembalikan teks tombol
        captureButton.classList.add('hidden'); // Sembunyikan tombol ambil gambar
        retakeButton.classList.add('hidden'); // Sembunyikan tombol ambil ulang
        isCameraOn = false;
        showMessage('Gagal mengakses kamera. Pastikan Anda memberikan izin akses dan tidak ada aplikasi lain yang menggunakan kamera. Akses kamera hanya berfungsi di localhost atau HTTPS.');
    }
}

// Fungsi untuk menghentikan stream kamera
function stopCamera() {
    if (stream) {
        console.log("DEBUG: Menghentikan stream kamera...");
        stream.getTracks().forEach(track => track.stop());
        cameraFeed.srcObject = null;
        isCameraOn = false;
        cameraToggleButton.textContent = 'Nyalakan Kamera'; // Ubah teks tombol
        cameraFeed.classList.add('hidden'); // Sembunyikan video feed
    }
}

// Fungsi untuk mengelola tombol On/Off Kamera
if (cameraToggleButton) {
    cameraToggleButton.addEventListener('click', () => {
        if (isCameraOn) {
            stopCamera();
            captureButton.classList.add('hidden'); // Sembunyikan tombol ambil
            retakeButton.classList.add('hidden'); // Sembunyikan tombol ambil ulang
            captureCanvas.classList.add('hidden'); // Sembunyikan canvas juga jika kamera dimatikan
        } else {
            startCamera();
        }
    });
}

// Fungsi terpadu untuk mengirim gambar ke backend (baik dari kamera atau upload)
async function sendImageForPrediction(base64Image) {
    predictionText.textContent = 'Mengklasifikasi...';
    loadingIndicator.classList.remove('hidden');
    
    // DEBUG: Log panjang base64 yang akan dikirim
    console.log("DEBUG: Panjang base64 image yang dikirim:", base64Image ? base64Image.length : "null/undefined");

    try {
        const response = await fetch(API_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ image: base64Image }),
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(`HTTP error! Status: ${response.status}, Message: ${errorData.error || response.statusText}`);
        }

        const result = await response.json();
        predictionText.textContent = `${(result.confidence * 100).toFixed(2)}% ${result.prediction}`;

    } catch (error) {
        console.error('ERROR: Error saat mengirim gambar atau menerima prediksi:', error);
        predictionText.textContent = 'Gagal mengklasifikasi.';
        showMessage(`Gagal mengklasifikasi gambar: ${error.message}. Pastikan backend berjalan dan URL API benar.`);
    } finally {
        loadingIndicator.classList.add('hidden');
    }
}

// Event listener untuk tombol "Ambil Gambar & Klasifikasi" (Mode Kamera)
if (captureButton) {
    captureButton.addEventListener('click', async () => {
        if (!isCameraOn) {
            showMessage("Kamera belum dinyalakan. Silakan nyalakan kamera terlebih dahulu.");
            return;
        }

        console.log("DEBUG: Tombol 'Ambil Gambar' diklik.");
        
        // Pastikan video sudah dimuat dan memiliki dimensi SEBELUM menghentikan stream
        if (cameraFeed.readyState === cameraFeed.HAVE_ENOUGH_DATA) {
            console.log("DEBUG: Camera feed is ready for capture.");

            // Gambar frame video saat ini ke canvas
            contextCapture.save();
            contextCapture.translate(captureCanvas.width, 0);
            contextCapture.scale(-1, 1);
            contextCapture.drawImage(cameraFeed, 0, 0, captureCanvas.width, captureCanvas.height);
            contextCapture.restore();
            console.log("DEBUG: Gambar telah digambar ke captureCanvas.");

            stopCamera(); // Sekarang aman untuk menghentikan stream

            const imageData = captureCanvas.toDataURL('image/jpeg', 0.9);
            console.log("DEBUG: Panjang imageData (Base64) dari canvas:", imageData.length);
            lastCapturedImageBase64 = imageData; // Simpan gambar yang diambil
            displayImageOnCanvas(lastCapturedImageBase64, captureCanvas); // Tampilkan di canvas capture
            
            captureButton.classList.add('hidden'); // Sembunyikan tombol ambil gambar
            retakeButton.classList.remove('hidden'); // Tampilkan tombol ambil ulang

            const base64ImageWithoutPrefix = imageData.split("base64,")[1];
            console.log("DEBUG: Panjang base64 string tanpa prefix:", base64ImageWithoutPrefix ? base64ImageWithoutPrefix.length : "null/undefined (Kosong)");

            if (!base64ImageWithoutPrefix || base64ImageWithoutPrefix.length < 100) { // Cek apakah base64 terlalu pendek/kosong
                console.error("ERROR: Gambar yang diambil sepertinya kosong atau tidak valid.");
                showMessage("Gagal mengambil gambar. Pastikan kamera menampilkan feed yang benar.");
                loadingIndicator.classList.add('hidden');
                predictionText.textContent = 'Gagal.';
                return;
            }

            sendImageForPrediction(base64ImageWithoutPrefix);

        } else {
            console.warn("WARN: Camera feed not ready (readyState:", cameraFeed.readyState, "). Cannot capture.");
            showMessage('Video feed belum siap. Coba lagi.');
            startCamera();
        }
    });
}

// Event listener untuk tombol "Ambil Ulang" (Mode Kamera)
if (retakeButton) {
    retakeButton.addEventListener('click', () => {
        console.log("DEBUG: Tombol 'Ambil Ulang' diklik.");
        startCamera(); // Mulai ulang kamera
        predictionText.textContent = 'Menunggu gambar...'; // Reset teks prediksi
    });
}

// Event listener untuk input file (Mode Unggah)
if (imageUpload) {
    imageUpload.addEventListener('change', (event) => {
        const file = event.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (e) => {
                lastCapturedImageBase64 = e.target.result; // Simpan gambar yang diunggah
                displayImageOnCanvas(lastCapturedImageBase64, uploadedPreviewCanvas); // Tampilkan di canvas pratinjau upload
                predictionText.textContent = 'Gambar siap diklasifikasi.';
            };
            reader.readAsDataURL(file); // Baca file sebagai Base64
        } else {
            uploadedPreviewCanvas.classList.add('hidden');
            lastCapturedImageBase64 = null;
            predictionText.textContent = 'Menunggu gambar...';
        }
    });
}

// Event listener untuk tombol "Klasifikasikan Gambar" (Mode Unggah)
if (uploadButton) {
    uploadButton.addEventListener('click', async () => {
        console.log("DEBUG: Tombol 'Klasifikasikan Gambar' diklik (Upload Mode).");
        if (!lastCapturedImageBase64) {
            showMessage("Harap unggah gambar terlebih dahulu.");
            return;
        }
        const base64ImageWithoutPrefix = lastCapturedImageBase64.split("base64,")[1];
        sendImageForPrediction(base64ImageWithoutPrefix);
    });
}

// Fungsi untuk beralih mode
function switchMode(mode) {
    console.log("DEBUG: Beralih ke mode:", mode);
    // Sembunyikan semua section terlebih dahulu
    cameraModeSection.classList.add('hidden');
    uploadModeSection.classList.add('hidden');
    predictionText.textContent = 'Menunggu input...'; // Reset teks prediksi
    loadingIndicator.classList.add('hidden');
    lastCapturedImageBase64 = null; // Reset gambar yang disimpan
    captureCanvas.classList.add('hidden'); // Sembunyikan canvas
    uploadedPreviewCanvas.classList.add('hidden'); // Sembunyikan canvas upload
    
    // Pastikan kamera mati saat beralih mode atau saat mode upload dipilih
    stopCamera(); 

    // Reset input file agar pengguna bisa memilih file baru setiap kali
    if (imageUpload) imageUpload.value = '';

    // Atur tombol mode
    toggleCameraModeButton.classList.remove('bg-blue-600', 'bg-gray-400');
    toggleUploadModeButton.classList.remove('bg-blue-600', 'bg-gray-400');

    if (mode === 'camera') {
        cameraModeSection.classList.remove('hidden');
        toggleCameraModeButton.classList.add('bg-blue-600');
        toggleUploadModeButton.classList.add('bg-gray-400');
        predictionText.textContent = 'Silakan nyalakan kamera.';
        // Kamera tidak otomatis on, pengguna harus klik tombol
    } else if (mode === 'upload') {
        uploadModeSection.classList.remove('hidden');
        toggleUploadModeButton.classList.add('bg-blue-600');
        toggleCameraModeButton.classList.add('bg-gray-400');
        predictionText.textContent = 'Unggah gambar dari perangkat Anda.';
    }
    currentMode = mode;
}

// Event listener untuk tombol beralih mode
if (toggleCameraModeButton) {
    toggleCameraModeButton.addEventListener('click', () => switchMode('camera'));
}
if (toggleUploadModeButton) {
    toggleUploadModeButton.addEventListener('click', () => switchMode('upload'));
}


// Inisialisasi: Panggil fungsi ini saat halaman dimuat
if (window.location.pathname.includes('prediction.html')) {
    window.onload = () => {
        // Mulai dengan mode kamera secara default saat halaman dimuat
        switchMode('camera'); 
    };
}