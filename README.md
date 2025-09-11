# FlavoRise: Flavors That Grow With You 🍲✨
Deployed Application 

[https://flavourise-192807546815.asia-southeast1.run.app/](https://flavourise-192807546815.asia-southeast1.run.app/)
## Quick Installation

### Using Docker
```python
docker compose up
```

## 📌 Proposal Category
**Productivity Boost**

## 🚀 Overview
**FlavoRise** adalah aplikasi pintar berbasis AI yang membantu pengguna menentukan resep masakan dari bahan seadanya di rumah.  
Tidak seperti aplikasi resep konvensional, FlavoRise menghadirkan **AI Recipe Remix** yang memungkinkan resep berkembang secara dinamis sesuai **rating & feedback pengguna**.

---

## 🎯 Target Audience
- Anak muda dan keluarga yang sering memasak dengan bahan seadanya
- Food enthusiast yang suka bereksperimen dengan resep baru
- Komunitas kuliner sosial media yang senang berbagi ide

---

## 🛠️ Problem Statement
Banyak orang kesulitan menentukan menu masakan hanya dengan bahan yang tersedia di rumah.  
Aplikasi resep konvensional biasanya hanya memberi saran statis, **tanpa memperhatikan preferensi rasa, kesehatan, maupun kreativitas**.  

👉 **Belum ada mekanisme agar resep bisa berevolusi sesuai feedback pengguna.**

---

## 💡 Solution: AI Recipe Remix
1. **Input Bahan** → pengguna memasukkan bahan atau mengunggah foto
2. **AI Multimodal (LLaMA)** → mengenali bahan dari gambar
3. **RAG (Retrieval-Augmented Generation)** → mencari resep relevan dari database
4. **User Rating** → pengguna memberikan feedback/rating
5. **Enhancement Chain** → sistem meningkatkan rekomendasi resep menggunakan **MMR (Maximal Marginal Relevance)** berdasarkan rating sebelumnya

---

## 🌟 Key Features
- 📷 **Image to Ingredient**: LLaMA mengenali bahan dari gambar
- 🔍 **AI Recipe Search**: rekomendasi resep dari bahan sederhana
- ⭐ **Rating Enhancement**: resep berkembang sesuai feedback
- 🧠 **Dynamic Learning**: preferensi user diperkuat di setiap interaksi
- 👥 **Komunitas Kolaboratif**: berbagi ide & hasil eksperimen resep

---

## 🔮 Expected Impact & Future Plan
- ✅ Membantu pengguna memasak dengan efisien & kreatif  
- ✅ Mengurangi food waste  
- ✅ Menawarkan opsi sehat & sesuai preferensi diet  
- 📈 Roadmap:
  - Integrasi ke **smart fridge**  
  - **Marketplace kreator resep**  
  - AI yang mendukung preferensi user secara massal  
  - Konten **short video interaktif**  

---

## 🧑‍💻 Technical Integration
**User Input**: daftar bahan / gambar bahan / preferensi gaya masak  

**System Prompt (LLaMA):**
Apa saja yang ada di gambar itu? sebutkan saja bahan makanan nya, dipisah oleh koma. Jangan tambahkan penjelasan apapun selain daftar bahan makanannya.


**Pipeline:**
1. User upload gambar / input bahan manual
2. LLaMA → generate caption berupa daftar bahan
3. User dapat mengoreksi hasil caption
4. Caption bahan → vektor search ke database resep (BM25 / sparse search)
5. User memberikan rating
6. Algoritma MMR → meningkatkan rekomendasi resep selanjutnya

---

## 🏗️ Tech Stack
- **Frontend**: [Gradio](https://gradio.app/)
- **Backend**: Python
- **AI Model**: `meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8`
- **Retrieval**: RAG (BM25 / sparse search)
- **Database**: Pinecone (vector storage)
- **Deployment**: Google Cloud Platform (GCP)

---

## 📦 Current Development
- ✅ Prototipe awal dengan input bahan + multimodal AI
- ✅ Sistem rating & feedback pengguna
- ✅ Konsep rating enhancement chain sudah dirancang

---

## 🤝 Contributing
Kontribusi selalu terbuka!  
Jika ingin membantu, silakan fork repo ini, buat branch baru, lalu kirimkan pull request 🚀

---

## 📄 License
MIT License © 2025 FlavoRise Team
