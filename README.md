# ✨ Pilah AI: Revolusi Pemilahan Sampah di Ujung Jari Anda ✨

!
*(Ganti dengan tangkapan layar (screenshot) nyata dari halaman utama aplikasi Anda)*

## Daftar Isi

1.  [Pendahuluan](#pendahuluan)
2.  [Latar Belakang Masalah](#latar-belakang-masalah)
3.  [Fitur Utama](#fitur-utama)
4.  [Teknologi yang Digunakan](#teknologi-yang-digunakan)
5.  [Arsitektur Proyek](#arsitektur-proyek)
6.  [Hasil dan Evaluasi Model ML](#hasil-dan-evaluasi-model-ml)
7.  [Panduan Instalasi dan Penggunaan (Lokal)](#panduan-instalasi-dan-penggunaan-lokal)
    * [Prasyarat](#prasyarat)
    * [Struktur Folder](#struktur-folder)
    * [Langkah-langkah Instalasi](#langkah-langkah-instalasi)
    * [Menjalankan Aplikasi](#menjalankan-aplikasi)
8.  [Kontribusi](#kontribusi)
9.  [Lisensi](#lisensi)

## 1. Pendahuluan

Pilah AI adalah aplikasi web inovatif yang memanfaatkan kecerdasan buatan (AI) *computer vision* untuk memudahkan klasifikasi sampah menjadi organik atau anorganik. Proyek ini dibangun sebagai solusi *end-to-end* yang kuat, mulai dari pra-pemrosesan data gambar, pelatihan model *deep learning* yang canggih, hingga antarmuka web yang responsif dan mudah digunakan.

## 2. Latar Belakang Masalah

Tantangan pengelolaan sampah yang kompleks seringkali menghambat upaya keberlanjutan dan daur ulang efektif di berbagai sektor. Aplikasi web "Pilah AI" hadir untuk mengatasi masalah ini dengan memanfaatkan kecerdasan buatan (AI) *computer vision*. Cukup dengan kamera perangkat atau unggahan gambar, sistem ini secara cerdas mengklasifikasikan sampah menjadi organik atau anorganik, memberikan kemudahan bagi setiap individu untuk memilah sampah dengan tepat dan berkontribusi langsung pada upaya lingkungan yang lebih besar.

## 3. Fitur Utama

* **Klasifikasi Sampah Cerdas:** Mampu mengidentifikasi jenis sampah (organik/anorganik) secara otomatis menggunakan AI.
* **Input Gambar Fleksibel:** Pengguna dapat mengklasifikasikan sampah melalui:
    * Pengambilan gambar langsung dari webcam perangkat.
    * Pengunggahan file gambar dari penyimpanan lokal.
* **Antarmuka Pengguna Intuitif:** Desain web yang bersih dan responsif, mudah digunakan di desktop maupun perangkat seluler.
* **Visualisasi Hasil:** Menampilkan hasil prediksi klasifikasi beserta tingkat kepercayaannya.

## 4. Teknologi yang Digunakan

Proyek ini dibangun menggunakan kombinasi teknologi *machine learning* dan pengembangan web modern:

**Backend & Machine Learning:**

* **Python:** Bahasa "ajaib" yang menggerakkan seluruh sistem AI kami.
* **TensorFlow / Keras:** "Otak" utama untuk membangun dan melatih model AI kami dalam mengenali gambar.
    * **MobileNetV2 (Transfer Learning):** Kami tidak melatih AI dari nol! Kami menggunakan "pengetahuan" yang sudah ada dari model MobileNetV2 (yang sudah sangat pintar mengenali banyak gambar) dan mengajarinya tentang topik baru. Ini seperti mengajari seorang jenius tentang topik baru!
* **Keras Tuner (Bayesian Optimization):** Ini adalah "pelatih super" yang membantu kami menemukan setelan terbaik untuk model AI kami, memastikan kinerjanya optimal.
* **ImageDataGenerator:** Kami membuat AI kami lebih pintar dengan menunjukkan variasi gambar yang sama (misalnya, gambar sampah yang sedikit diputar, diperbesar, atau digeser). Ini disebut augmentasi data.
* **Flask:** Sebuah "pelayan" web Python yang memungkinkan aplikasi web Anda "berbicara" dengan model AI kami.
* **Flask-CORS:** Memastikan pelayan web kami bisa "berbicara" dengan aplikasi web Anda tanpa hambatan keamanan.
* **NumPy, Pandas, Pillow, Scikit-learn:** Alat-alat penting untuk membersihkan, mengatur, dan menyiapkan data gambar agar siap dipelajari AI.

**Wajah Aplikasi (Frontend Web):**

* **HTML5:** Kerangka dasar dari halaman web yang Anda lihat.
* **CSS3 & Tailwind CSS:** "Pakaian" dan "gaya rambut" aplikasi kami, membuatnya terlihat modern dan responsif di berbagai perangkat.
* **JavaScript:** "Jantung" aplikasi web, yang mengelola interaksi Anda, mengaktifkan kamera, mengirim gambar ke AI, dan menampilkan hasilnya.

## 5. Arsitektur Proyek: Bagaimana Semuanya Terhubung

Bayangkan proyek ini seperti kota dengan beberapa distrik yang saling terhubung:
```
Pilah-AI-Project/
├── data/                  # Gudang data (gambar mentah & yang sudah diproses)
│   ├── raw/               # Sampah mentah (gambar asli & daftar anotasi)
│   └── processed/         # Sampah siap olah (gambar yang sudah diatur untuk AI)
├── python-model/          # Laboratorium AI kami
│   ├── ml-model-training.ipynb # Buku resep untuk melatih otak AI
│   └── pre-processing.ipynb    # Buku resep untuk menyiapkan bahan (data)
├── models/                # Ruang Penyimpanan Otak AI
│   ├── klasifikasi_sampah_final_v1.h5  # Otak AI kami yang sudah terlatih
│   └── keras_tuner/                    # Catatan eksperimen pelatih AI
├── backend/               # Dapur API (tempat AI melayani permintaan)
│   └── app.py                  # Resep untuk pelayan API kami
├── frontend/              # Wajah Aplikasi (tempat Anda berinteraksi)
│   ├── index.html              # Halaman sambutan
│   ├── prediction.html         # Halaman utama untuk memilah
│   ├── css/
│   │   └── style.css           # Gaya & warna aplikasi
│   ├── js/
│   │   └── main.js             # Logika interaksi & komunikasi
│   └── img/                    # Galeri contoh sampah
└── README.md                   # Peta proyek ini
```

## 6. Hasil & Evaluasi Model AI: Seberapa Pintar AI Kami?

Kami telah menguji "otak" AI kami dengan cermat, dan hasilnya sangat menjanjikan!

* **Arsitektur Otak AI:** Kami menggunakan arsitektur Jaringan Saraf Tiruan (CNN) yang sangat canggih, yang telah disempurnakan oleh "pelatih super" kami (Keras Tuner) dan dibangun di atas fondasi MobileNetV2.
* **Kinerja Uji:** Pada data yang belum pernah dilihat AI sebelumnya (Test Set), model kami menunjukkan kinerja luar biasa:
    * **Akurasi: 96.34%** (Artinya, 96.34% prediksinya benar!)
    * **Loss: 0.1706** (Angka ini menunjukkan seberapa "salah" prediksi kami, semakin kecil semakin baik!)
* **Laporan Detail (Precision, Recall, F1-score):**

    ```
    Kelas     Precision  Recall  F1-score  Support
    Anorganik   0.96     0.96      0.96      270
    Organik     0.96     0.96      0.96      267
    accuracy                         0.96      537
    macro avg   0.96     0.96      0.96      537
    weighted avg 0.96     0.96      0.96      537
    ```
    Angka-angka ini (0.96 untuk Precision, Recall, dan F1-score) menunjukkan bahwa model kami tidak hanya sering benar, tetapi juga sangat baik dalam menemukan semua sampah yang relevan dan tidak salah mengklasifikasikan sampah.

* **Peta Kebingungan (Confusion Matrix):**
    !
    *(Ganti dengan tangkapan layar (screenshot) nyata dari confusion matrix Anda)*
    Peta ini menunjukkan seberapa sering AI kami bingung antara Organik dan Anorganik. Semakin gelap kotak diagonal, semakin baik!

Singkatnya, AI kami sangat andal dalam memilah sampah Anda!

## 🚀 Panduan Instalasi & Penggunaan (Lokal): Ayo Coba Sendiri!

Ingin menjalankan Pilah AI di komputer Anda? Ikuti langkah-langkah mudah ini:

### Prasyarat: Apa yang Anda Butuhkan

* **Python 3.8+:** Pastikan Anda punya Python (versi 3.8 atau lebih baru).
* **pip:** Alat untuk menginstal pustaka Python (biasanya sudah ada bersama Python).
* **Git:** (Opsional) Untuk mengunduh kode proyek dengan mudah.
* **VS Code:** (Sangat Disarankan) Editor kode yang bagus dengan ekstensi "Live Server" untuk melihat aplikasi web Anda secara langsung.

### Struktur Folder: Pastikan Semuanya di Tempatnya

Pastikan folder proyek Anda terlihat seperti yang dijelaskan di bagian [Arsitektur Proyek](#arsitektur-proyek). Terutama:
* File `_annotations.csv` awal dan gambar mentah Anda di `data/raw/`.
* Folder `data/processed/` dan `models/` akan dibuat otomatis.

### Langkah-langkah Instalasi: Membangun Fondasi

1.  **Dapatkan Kodenya:**
    * **Jika pakai Git:**
        ```bash
        git clone <URL_REPO_ANDA>
        cd Pilah-AI-Project
        ```
    * **Jika manual:** Unduh semua file dan atur dalam struktur folder yang benar.

2.  **Siapkan Lingkungan Python (Sangat Disarankan!):**
    * Buka terminal/Command Prompt di folder utama proyek Anda (`Pilah-AI-Project/`).
    * Buat lingkungan virtual:
        ```bash
        python -m venv venv
        ```
    * Aktifkan:
        * **Windows:**
            ```bash
            .\venv\Scripts\activate
            ```
        * **macOS/Linux:**
            ```bash
            source venv/bin/activate
            ```

3.  **Instal "Bahan-bahan" Python:**
    * Buat file bernama `requirements.txt` di folder utama proyek Anda. Isi dengan daftar ini:

    * Instal semua bahan:
        ```bash
        pip install -r requirements.txt
        ```

4.  **Siapkan Data (Pra-pemrosesan):**
    * Pastikan file `_annotations.csv` dan gambar mentah Anda ada di `data/raw/`.
    * Buka `python-model/pre-processing.ipynb` di Jupyter Notebook/Lab (jalankan `jupyter notebook` di terminal).
    * Jalankan **semua sel** di notebook ini. Ini akan membersihkan, mengatur, dan membagi data Anda, lalu menyimpannya di `data/processed/`.

5.  **Latih Otak AI (Model ML):**
    * Buka `python-model/ml-model-training.ipynb` di Jupyter Notebook/Lab.
    * Jalankan **semua sel** di notebook ini. Proses ini akan melatih model AI Anda, mencari setelan terbaik, dan menyimpannya di `models/klasifikasi_sampah_final_v1.h5`. Ini mungkin butuh waktu lama, tergantung spesifikasi komputer Anda!

### Menjalankan Aplikasi: Saatnya Beraksi!

1.  **Nyalakan "Pelayan" API (Backend):**
    * Buka terminal **baru** (pastikan lingkungan virtual Anda masih aktif).
    * Masuk ke folder `backend`:
        ```bash
        cd backend
        ```
    * Jalankan pelayan Flask:
        ```bash
        python app.py
        ```
    * Anda akan melihat pesan bahwa server berjalan di `http://127.0.0.1:5000`. Biarkan terminal ini tetap berjalan.

2.  **Buka Aplikasi Web (Frontend):**
    * Buka proyek Anda di VS Code.
    * Instal ekstensi **Live Server** jika belum (cari "Live Server" di Extensions marketplace).
    * Navigasi ke file `frontend/index.html`.
    * Klik kanan pada `index.html` dan pilih **"Open with Live Server"**.
    * Aplikasi web Anda akan terbuka di browser (biasanya `http://127.0.0.1:5500`).

3.  **Mulai Memilah!**
    * Dari halaman utama, klik tombol "Mulai Pilah Sekarang!".
    * Anda akan diarahkan ke halaman prediksi.
    * Pilih "Mode Kamera" atau "Mode Unggah".
    * **Mode Kamera:** Klik "Nyalakan Kamera", berikan izin browser. Arahkan kamera ke sampah dan klik "Ambil Gambar & Klasifikasi".
    * **Mode Unggah:** Pilih gambar sampah dari komputer Anda, lalu klik "Klasifikasikan Gambar".
    * Lihat hasilnya tampil di layar!

## 8. Kontribusi

Jika Anda tertarik untuk berkontribusi untuk membuat Pilah AI lebih baik, silakan ikuti langkah-langkah berikut:

1.  *Fork* repositori ini.
2.  Buat cabang baru (`git checkout -b feature/nama-fitur-baru`).
3.  Lakukan perubahan Anda.
4.  *Commit* perubahan Anda (`git commit -m 'Tambahkan fitur baru'`).
5.  *Push* ke cabang Anda (`git push origin feature/nama-fitur-baru`).
6.  Buka *Pull Request* baru.

## 9. Lisensi

Proyek ini dilisensikan di bawah Lisensi MIT. Lihat file `LICENSE` untuk detail lebih lanjut.
