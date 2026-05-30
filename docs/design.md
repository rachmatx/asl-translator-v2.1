# ASL Detector - Design System & Theme Specifications

Dokumen ini memuat spesifikasi desain dari antarmuka web Detektor Bahasa Isyarat ASL. Sistem antarmuka ini dirancang menggunakan arsitektur CSS berlapis (*layered*) yang mengizinkan perpindahan tema secara mulus (*seamless theme switching*) hanya dengan mengganti kelas (`class`) pada tag `<body>`.

Saat ini, sistem memuat tiga (3) tema utama: **Modern Glassy (Default)**, **Clean Minimalist**, dan **Neobrutalism**.

---

## 1. Tema Modern Glassy (Default)
Tema ini mengusung estetika *cyberpunk* yang elegan dengan latar belakang super gelap, efek *glassmorphism* (kaca tembus pandang), dan pendaran cahaya neon. Sangat cocok untuk nuansa aplikasi AI berteknologi tinggi.

**Tipografi:**
- Font Utama (Body): `Inter`
- Font Judul (Display): `Space Grotesk`
- Font Kode (Mono): `monospace`
- Ukuran Dasar: `14px`

**Warna Latar & Kaca:**
- Latar Belakang Terdalam (`--bg-deep`): `#050810` (Biru sangat gelap)
- Latar Kartu Utama (`--bg-card`): `#0d1117`
- Efek Kaca (`--bg-glass`): `rgba(255, 255, 255, 0.04)` dengan `backdrop-filter: blur(20px)`
- Garis Tepi (`--border`): `rgba(255, 255, 255, 0.08)`

**Aksen Neon:**
- Aksen 1 (Ungu/Biru): `#6c63ff`
- Aksen 2 (Cyan): `#00d4ff`
- Aksen 3 (Pink): `#ff6b9d`
- Efek Pendaran (*Glow*): `0 0 40px rgba(108, 99, 255, 0.25)`

**Warna Teks & Status:**
- Teks Utama (`--text-1`): `#f0f4ff`
- Teks Redup (`--text-2`): `#8892a4`
- Teks Latar (`--text-3`): `#4a5568`
- Sukses (Hijau): `#00e676`
- Peringatan (Oranye): `#ffab40`

---

## 2. Tema Clean Minimalist
Tema ini difokuskan pada keterbacaan tingkat tinggi, antarmuka yang terang (*light mode*), dan desain yang mirip dengan *dashboard* analitik modern (seperti Vercel atau Stripe). Sangat nyaman untuk penggunaan jangka panjang di siang hari.

**Tipografi:**
- Font Utama & Judul: `Inter`
- Font Kode: `JetBrains Mono`
- Ukuran Dasar: `16px` (Lebih besar untuk aksesibilitas)

**Warna Latar & Kartu (Light Mode):**
- Latar Belakang Terdalam (`--bg-deep`): `#f8fafc` (Slate muda)
- Latar Kartu & Kaca (`--bg-card`, `--bg-glass`): `#ffffff` (Putih solid)
- Garis Tepi (`--border`): `#e2e8f0`
- Bayangan Halus (*Drop Shadow*): `rgba(0, 0, 0, 0.08) 0px 4px 16px 0px`

**Aksen Biru Profesional:**
- Aksen 1 (Biru Utama): `#3b82f6`
- Aksen 2 (Biru Terang): `#0ea5e9`
- Aksen 3 (Nila): `#6366f1`

**Warna Teks & Status:**
- Teks Utama (`--text-1`): `#0f172a` (Hampir hitam)
- Teks Redup (`--text-2`): `#475569`
- Teks Latar (`--text-3`): `#94a3b8`
- Sukses (Hijau): `#16a34a`
- Peringatan (Oranye): `#d97706`

---

## 3. Tema Neobrutalism
Tema ini mengusung tren *Neobrutalism* yang mencolok: batas luar (*border*) hitam tebal, bayangan padat (*solid drop-shadows*), ketidaksimetrisan, dan palet warna permen yang sangat kontras. Desain ini bertujuan untuk terlihat sangat "kasar" namun kekinian dan ceria.

**Tipografi:**
- Font Utama & Judul: `DM Sans`
- Ukuran Dasar: `14px`
- Sudut Melengkung (`--radius-lg, md, sm`): `5px` (Hampir kotak sepenuhnya)

**Struktur Brutalis:**
- Latar Belakang (`--bg-deep`, `--bg-card`): `#ffffff` (Putih solid)
- Garis Tepi (`--border`): `#000000` (Hitam pekat, biasanya ditebalkan jadi `2px` atau `4px`)
- Bayangan Padat (*Shadow Glow*): `4px 4px 0px 0px #000000`

**Warna Kustom Setiap Komponen (Override):**
Tema ini memiliki keunikan di mana setiap panel (*card*) memiliki warna latar (*background*) yang berbeda-beda untuk menciptakan kontras tinggi yang menyenangkan:
- Panel Kamera: `#FFDE59` (Vibrant Yellow)
- Panel Pengaturan/Prediksi Utama: `#FF66C4` (Hot Pink)
- Panel Kontrol/Penyusun Kata: `#00E5FF` (Bright Cyan)
- Panel Statistik/Matrix: `#00FF66` (Neon Green)
- Panel Log Sistem: `#CB6CE6` (Vivid Purple)

**Warna Teks & Status:**
- Teks (Semua Tingkatan): `#000000` (Hitam solid)
- Sukses: `#22c55e`
- Peringatan: `#f97316`

---

## 4. Cara Penggunaan Tema

Penggantian tema dilakukan secara *real-time* lewat JavaScript dengan menimpa (override) *class* pada elemen `<body>`. 
File utama `theme.css` secara cerdas mendeteksi *class* ini dan akan secara berjenjang (*cascade*) mengganti nilai variabel `--bg-`, `--text-`, dll., tanpa perlu mengubah struktur HTML.

```javascript
// Contoh Logika Penggantian Tema (theme.js)
document.body.classList.remove('theme-clean', 'theme-neobrutalism');
document.body.classList.add('theme-neobrutalism'); // Mengaktifkan Neobrutalism
```
