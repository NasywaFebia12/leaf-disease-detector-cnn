
import streamlit as st
import numpy as np
import os
import gdown
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Download model dari Google Drive jika belum ada
file_id = "1D3-h5TqUzQIck58fY_6dzMWTeK9EkeX-"
output = "leaf_model_transfer_cnn.h5"
if not os.path.exists(output):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output, quiet=False)

# Load model
model = load_model(output)

# Label kelas model
class_labels = ['Alstonia Scholaris diseased (P2a)', 'Alstonia Scholaris healthy (P2b)', 'Arjun diseased (P1a)', 'Arjun healthy (P1b)', 
                'Bael diseased (P4b)', 'Basil healthy (P8)', 'Chinar diseased (P11b)', 'Chinar healthy (P11a)', 
                'Gauva diseased (P3b)', 'Gauva healthy (P3a)', 'Jamun diseased (P5b)', 'Jamun healthy (P5a)', 
                'Jatropha diseased (P6b)', 'Jatropha healthy (P6a)', 'Lemon diseased (P10b)', 'Lemon healthy (P10a)', 
                'Mango diseased (P0b)', 'Mango healthy (P0a)', 'Pomegranate diseased (P9b)', 'Pomegranate healthy (P9a)', 
                'Pongamia Pinnata diseased (P7b)', 'Pongamia Pinnata healthy (P7a)']

# Dictionary rekomendasi penanganan
disease_treatment = {
    'Alstonia Scholaris diseased (P2a)': 'Buang daun yang terinfeksi dan semprotkan fungisida sistemik.',
    'Alstonia Scholaris healthy (P2b)': 'Tidak perlu penanganan. Lanjutkan penyiraman dan pemupukan rutin.',
    'Arjun diseased (P1a)': 'Gunakan fungisida berbahan aktif tembaga dan kurangi kelembapan sekitar tanaman.',
    'Arjun healthy (P1b)': 'Daun sehat, cukup pantau rutin dan jaga kelembapan tanah.',
    'Bael diseased (P4b)': 'Semprot dengan larutan sulfur untuk mengatasi jamur. Hindari genangan air.',
    'Basil healthy (P8)': 'Daun sehat. Pastikan mendapat cahaya matahari cukup dan drainase baik.',
    'Chinar diseased (P11b)': 'Pangkas bagian yang terkena dan semprotkan fungisida kontak setiap 5 hari.',
    'Chinar healthy (P11a)': 'Tanaman dalam kondisi baik. Lanjutkan pemantauan berkala.',
    'Gauva diseased (P3b)': 'Semprotkan neem oil atau fungisida organik, dan hindari kelembaban berlebih.',
    'Gauva healthy (P3a)': 'Daun sehat. Lakukan pemupukan berkala dan pantau serangga.',
    'Jamun diseased (P5b)': 'Bersihkan area sekitar tanaman dan gunakan fungisida sistemik 2 minggu sekali.',
    'Jamun healthy (P5a)': 'Tidak ada gejala. Cukup lakukan pemangkasan ringan bila perlu.',
    'Jatropha diseased (P6b)': 'Semprotkan campuran insektisida dan fungisida organik pada pagi hari.',
    'Jatropha healthy (P6a)': 'Tanaman sehat. Cukup siram dan beri sinar matahari cukup.',
    'Lemon diseased (P10b)': 'Gunakan fungisida berbahan aktif klorotalonil dan buang daun busuk.',
    'Lemon healthy (P10a)': 'Daun normal. Pertahankan kelembapan tanah dan kebersihan daun.',
    'Mango diseased (P0b)': 'Lakukan pemangkasan dan semprot fungisida tembaga atau sulfur.',
    'Mango healthy (P0a)': 'Daun sehat. Pastikan sirkulasi udara bagus di sekitar tanaman.',
    'Pomegranate diseased (P9b)': 'Gunakan fungisida alami dan hindari penyiraman berlebih.',
    'Pomegranate healthy (P9a)': 'Tanaman normal. Pertahankan nutrisi dan drainase tanah yang baik.',
    'Pongamia Pinnata diseased (P7b)': 'Gunakan larutan sabun insektisida atau neem oil, semprot pagi hari.',
    'Pongamia Pinnata healthy (P7a)': 'Tidak ditemukan masalah. Lanjutkan perawatan rutin.',
}

# Antarmuka Streamlit
st.title("Klasifikasi Penyakit Daun Tanaman")
st.write("Upload gambar daun untuk deteksi kondisi dan dapatkan rekomendasi penanganan.")

uploaded_file = st.file_uploader("Pilih gambar daun...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).resize((128, 128))
    st.image(image, caption="Gambar yang diupload", use_column_width=True)

    img_array = img_to_array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    pred_class = class_labels[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    saran = disease_treatment.get(pred_class, "Rekomendasi belum tersedia.")

    st.markdown("### Prediksi: {} ({:.2f}%)".format(pred_class, confidence))
    st.markdown("**Rekomendasi Penanganan:** {}".format(saran))
