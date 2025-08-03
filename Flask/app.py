import os
import pickle
import re
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, send_file
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report, confusion_matrix, roc_curve
from io import BytesIO

# === KONFIGURASI MODEL & TOKENIZER ===
MODEL_PATH = "CNNBiLSTM_fasttext_300_FAIR.h5"
TOKENIZER_PATH = "tokenizer_fasttext.pkl"
MAX_LEN = 55
DEFAULT_THRESHOLD = 0.47

# === FUNGSI BERSIHAN TEKS ===
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)           # hilangkan tanda baca
    text = re.sub(r"\s+", " ", text).strip()       # hilangkan spasi berlebih
    text = re.sub(r"[^\x00-\x7F]+", "", text)      # hilangkan karakter non-ASCII (emoji, dsb)
    return text

# === LOAD MODEL & TOKENIZER ===
model = load_model(MODEL_PATH)
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

# === SETUP FLASK ===
app = Flask(__name__)
uploaded_df = None  # Global untuk simpan hasil prediksi sementara

@app.route("/", methods=["GET", "POST"])
def index():
    global uploaded_df
    predictions = None
    total_data = None
    jumlah_salah = None
    akurasi_prediksi = None
    cyber_count = None
    noncyber_count = None
    error = None
    threshold_used = DEFAULT_THRESHOLD

    if request.method == "POST":
        file = request.files["file"]
        if file:
            try:
                df = pd.read_excel(file, engine="openpyxl")
            except:
                try:
                    df = pd.read_excel(file)
                except Exception as e:
                    return render_template("index.html", error=f"Gagal membaca file: {e}")

            if "text" not in df.columns:
                return render_template("index.html", error="Kolom 'text' tidak ditemukan.")

            texts = df["text"].astype(str).apply(clean_text).tolist()
            sequences = tokenizer.texts_to_sequences(texts)
            padded = pad_sequences(sequences, maxlen=MAX_LEN)
            probs = model.predict(padded)

            # ROC-based threshold
            if "label" in df.columns:
                true_labels = df["label"].astype(int).values
                fpr, tpr, thresholds = roc_curve(true_labels, probs)
                optimal_idx = (tpr - fpr).argmax()
                threshold_used = thresholds[optimal_idx]
                print(f"📌 ROC Threshold used: {round(threshold_used, 4)}")

            labels = (probs >= threshold_used).astype(int).flatten()

            df["Predicted_Label"] = labels
            df["Probability"] = probs.flatten()
            df["Class"] = df["Predicted_Label"].map({1: "Non-Cyberbullying", 0: "Cyberbullying"})

            uploaded_df = df.copy()

            predictions = df[["text", "Class", "Probability"]].values.tolist()
            total_data = len(df)
            cyber_count = int((df["Predicted_Label"] == 0).sum())
            noncyber_count = int((df["Predicted_Label"] == 1).sum())

            if "label" in df.columns:
                jumlah_salah = int((labels != true_labels).sum())
                akurasi_prediksi = round((1 - jumlah_salah / total_data) * 100, 2)

                print("\n=== Classification Report ===")
                print(classification_report(true_labels, labels))
                print("\n=== Confusion Matrix ===")
                print(confusion_matrix(true_labels, labels))
            else:
                error = "Kolom 'label' tidak ditemukan, akurasi tidak dihitung."

    return render_template("index.html",
                           predictions=predictions,
                           total_data=total_data,
                           jumlah_salah=jumlah_salah,
                           akurasi_prediksi=akurasi_prediksi,
                           cyber_count=cyber_count,
                           noncyber_count=noncyber_count,
                           threshold_used=round(threshold_used, 4),
                           error=error)

@app.route("/download")
def download():
    global uploaded_df
    if uploaded_df is None:
        return "Belum ada hasil untuk diunduh."
    output = BytesIO()
    uploaded_df.to_excel(output, index=False)
    output.seek(0)
    return send_file(output, as_attachment=True, download_name="hasil_prediksi.xlsx", mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# === RUN APP ===
if __name__ == "__main__":
    app.run(debug=True)
