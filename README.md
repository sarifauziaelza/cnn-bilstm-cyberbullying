# Klasifikasi Cyberbullying pada Media Sosial dengan Model Deep Learning Hybrid CNN–BiLSTM

Proyek ini merupakan bagian dari penelitian tesis yang bertujuan untuk membangun sistem klasifikasi cyberbullying pada media sosial menggunakan pendekatan deep learning. Model utama yang digunakan adalah hybrid CNN–BiLSTM, dengan perbandingan performa pada berbagai jenis word embedding seperti FastText, GloVe, dan Word2Vec.

## 📁 Struktur Folder

- `Dataset/` : Berisi dataset komentar dari media sosial (dalam format CSV).
- `Notebooks/` : Notebook eksplorasi, preprocessing, training, dan evaluasi model.
- `Flask/` : File model `.h5` hasil pelatihan.
- `Embedding/` : pretrained word embedding yang digunaakan.
- `README.md` : Deskripsi proyek ini.

## 🧠 Model yang Digunakan

- CNN
- LSTM
- BiLSTM
- CNN–LSTM
- CNN–BiLSTM ✅ *(Model terbaik)*
  
Masing-masing model dilatih dan diuji menggunakan tiga embedding:
- FastText 300D
- GloVe 300D
- Word2Vec 300D

## 🔍 Hasil Evaluasi Terbaik

| Model              | Embedding | Akurasi | AUC    | Precision | Recall | F1-score |
|-------------------|-----------|---------|--------|-----------|--------|----------|
| CNN–BiLSTM        | FastText  | 91.30%  | 0.9410 | 0.9238    | 0.9238 | 0.9238   |

## ⚙️ Tools & Teknologi

- Python 3.x
- TensorFlow / Keras
- Scikit-learn
- Matplotlib & Seaborn
- Flask (untuk web deployment)
- MySQL (untuk penyimpanan hasil prediksi)

"# cnn-bilstm-cyberbullying" 
