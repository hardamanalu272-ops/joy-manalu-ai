# Joy Manalu AI Studio 🎯
Sistem Pengenalan & Deteksi Multi-Wajah Real-Time Berbasis AI Deep Learning (Facenet-512 & YuNet SOTA).

---

## 🌟 Fitur Utama
- **Multi-Face Detector**: Mendeteksi 6+ wajah sekaligus dalam satu frame menggunakan OpenCV YuNet SOTA.
- **DeepFace Facenet-512**: Ekstraksi biometrik wajah 512-dimensi akurasi tinggi (~97% calibrated rank-1).
- **Cross-Platform**: Berjalan mulus di Browser Laptop, PC, dan Smartphone (Android & iOS).
- **Flip Camera**: Penggantian kamera depan (selfie) dan kamera belakang di smartphone.
- **Real-Time Telemetry**: Menampilkan FPS, waktu inferensi AI, dan status koneksi live.
- **Privasi Aman**: Foto dataset tidak pernah diunggah ke internet/repository publik.

---

## 🚀 Cara Deploy Gratis 24 Jam ke Render.com

Aplikasi ini membutuhkan backend Python (tidak cukup web statis biasa). Render.com menyediakan server gratis yang bisa menjalankan Python, OpenCV, dan DeepFace.

### Langkah 1: Push Kode ke GitHub
1. Buat repository baru di [GitHub](https://github.com/new).
   *(Bisa pilih **Private** agar lebih aman, atau **Public** karena data wajah sudah otomatis terlindungi oleh `.gitignore`).*
2. Jalankan perintah git push:
   ```bash
   git add .
   git commit -m "Deploy Joy Manalu AI Studio"
   git push -u origin main
   ```

### Langkah 2: Hubungkan ke Render.com
1. Buka [Render.com](https://render.com) dan daftar/login dengan akun GitHub.
2. Klik tombol **New +** di dashboard → pilih **Web Service**.
3. Pilih repository GitHub Anda (`joy-manalu-ai`).
4. Konfigurasi otomatis terisi dari `render.yaml` atau isi manual:
   - **Name**: `joy-manalu-ai`
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120`
   - **Instance Type**: `Free`
5. Klik **Create Web Service**. Tunggu 2-3 menit hingga status menjadi **Live**!

Aplikasi Anda kini memiliki link publik HTTPS gratis (contoh: `https://joy-manalu-ai.onrender.com`) yang bisa dibuka oleh siapa saja di seluruh dunia.

---

## ⏰ Trik Agar Server Aktif 24 Jam Non-Stop (Tanpa Tidur)

Paket gratis Render secara otomatis akan tidur (*spin down*) jika tidak ada pengunjung selama 15 menit. Agar aplikasi tetap menyala 24 jam tanpa jeda:

1. Buka layanan gratis [cron-job.org](https://cron-job.org) atau [UptimeRobot](https://uptimerobot.com).
2. Buat monitor/cron baru:
   - **URL**: `https://<nama-app-anda>.onrender.com/api/ping`
   - **Interval**: Setiap 10 menit.
3. Selesai! Layanan ini akan otomatis "mencolek" server Anda setiap 10 menit, sehingga server Anda **selalu terjaga 24 jam penuh tanpa biaya tambahan**.

---

## 🔒 Privasi Data Wajah
- Foto mentah di folder `dataset/` dan database biometrik `embeddings_cache.pkl` sengaja dikecualikan di `.gitignore` demi keamanan dan privasi.
- Di server cloud, Anda dapat mendaftarkan wajah baru secara langsung melalui antarmuka web (klik tombol **Daftarkan Wajah**).
