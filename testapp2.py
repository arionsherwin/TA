import pandas as pd
from io import BytesIO
import io
import base64
import tensorflow as tf
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas
from tensorflow.keras.utils import img_to_array
import random
import firebase_admin
from firebase_admin import credentials, storage
import google.auth
import google.auth.transport.requests
import requests
import json
from datetime import datetime

cred = credentials.Certificate("drawingapp3-firebase-adminsdk-v53rh-226ee77a90.json")
firebase_app = None
if not firebase_admin._apps:
    firebase_app = firebase_admin.initialize_app(cred, {'storageBucket': 'drawingapp3.appspot.com'})
else:
    firebase_app = firebase_admin.get_app()
bucket = storage.bucket()



def load_model():
    model = tf.keras.models.load_model('model4.h5')
    return model
model = load_model()

def preprocess(image):
    image = image.resize((350, 350))
    image_array = np.array(image)
    image_batch = np.expand_dims(image_array, axis=0)
    image_normalized = image_batch / 255.0
    return image_normalized

def prediksi(image_data, model):
    size = (350,350)
    img = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    imgarray = img_to_array(img)
    img = imgarray[np.newaxis,...]
    nama_class = ['delapan',
                    'dua',
                    'empat',
                    'enam',
                    'lima',
                    'nol',
                    'satu',
                    'sembilan',
                    'tiga',
                    'tujuh']
    x = imgarray
    x = np.expand_dims(x, axis=0)
    imgs = np.vstack([x])
    predik = model.predict(imgs, batch_size=100)
    for j in range(10):
        if predik[0][j]==1. :
            result = str(nama_class[j])
            return result
            break

def get_data():
    data = db.child("data").get()
    return data
        
def save_data(data, img):
    db.child("data").push(data)
    img_name = data["name"] + ".jpg"
    storage.child(img_name).put(img)

menu = ["Drawing", "Tentang Aplikasi", "Tentang Data"]
choice = st.sidebar.selectbox("Pilih Halaman", menu)

if choice == "Drawing":
    st.title("Drawing Apps For Kids")
    st.subheader('_Belajar menulis tidak harus menggunakan buku._')

    daft_angka = ['Angka 0', 'Angka 1', 'Angka 2', 'Angka 3', 'Angka 4', 'Angka 5', 'Angka 6', 'Angka 7', 'Angka 8', 'Angka 9',]
    pilihan_soal = st.selectbox("Pilih soal:", daft_angka)


    stroke_color = st.sidebar.color_picker("Stroke color hex: ")
    bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")

    realtime_update = st.sidebar.checkbox("Update in realtime", True)
    
    if "canvas_result" not in st.session_state:
        st.session_state.canvas_result = None

    canvas_result = st.session_state.canvas_result

    canvas_result = st_canvas(
        fill_color="rgba(350, 350, 0, 0.1)",
        stroke_color=stroke_color,
        stroke_width=0.5,
        background_color=bg_color,
        update_streamlit=realtime_update,
        width=350,
        height=350,
        drawing_mode="freedraw",
        key="canvas",
    )
    if st.button("Simpan Gambar"):
        image = Image.fromarray(canvas_result.image_data.astype(np.uint8), 'RGB')
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        check = str(pilihan_soal)
        if check == "Angka 0":
            filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_angka0.png"
            path = f"0/{filename}"
            i = 1
            while bucket.blob(path).exists():
                filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_angka0_{i}.png"
                path = f"0/{filename}"
                i += 1
            blob = bucket.blob(path)
            blob.upload_from_string(buffer.getvalue(), content_type='image/png')
            st.success('Gambar berhasil disimpan di database! Terima Kasih.')
      
    if st.button("Cek Jawaban"):
        if canvas_result.image_data is not None:
            image = Image.fromarray(canvas_result.image_data.astype(np.uint8)).convert("RGB")
            processed_image = preprocess(image)
            predictions = prediksi(image, model)
            valid = str(pilihan_soal)
            if valid == "Angka 0":
                if predictions == "nol":
                    st.write("Gambar diprediksi sebagai ",predictions, ", dan gambar merupakan ", valid,". Jawaban Kamu Benar!")
                else:
                    st.write("Gambar diprediksi sebagai ",predictions, ", namun sebenarnya gambar merupakan ", valid)
            elif valid == "Angka 3":
                if predictions == "tiga":
                    st.write("Gambar diprediksi sebagai ",predictions, ", dan gambar merupakan ", valid,". Jawaban Kamu Benar!")
                else:
                    st.write("Gambar diprediksi sebagai ",predictions, ", namun sebenarnya gambar merupakan ", valid)
            elif valid == "Angka 5":
                if predictions == "lima":
                    st.write("Gambar diprediksi sebagai ",predictions, ", dan gambar merupakan ", valid,". Jawaban Kamu Benar!")
                else:
                    st.write("Gambar diprediksi sebagai ",predictions, ", namun sebenarnya gambar merupakan ", valid)
            elif valid == "Angka 6":
                if predictions == "enam":
                    st.write("Gambar diprediksi sebagai ",predictions, ", dan gambar merupakan ", valid,". Jawaban Kamu Benar!")
                else:
                    st.write("Gambar diprediksi sebagai ",predictions, ", namun sebenarnya gambar merupakan ", valid)
            elif valid == "Angka 7":
                if predictions == "tujuh":
                    st.write("Gambar diprediksi sebagai ",predictions, ", dan gambar merupakan ", valid,". Jawaban Kamu Benar!")
                else:
                    st.write("Gambar diprediksi sebagai ",predictions, ", namun sebenarnya gambar merupakan ", valid)
            else:
                st.write("Gambar diprediksi sebagai ",predictions, ", namun sebenarnya gambar merupakan ", valid)
        else:
            st.write("Harap gambar dihasilkan terlebih dahulu menggunakan canvas.")

elif choice == "Tentang Aplikasi":
    st.title("Tentang Aplikasi")
    info_text = """
    ## Drawing Apps for Kids merupakan aplikasi latihan menulis angka 0 - 9 bagi anak pra-sekolah yang dibangun menggunakan metode CNN/Convolutional Neural Network.

    ## Data yang digunakan untuk membangun aplikasi ini merupakan gambar dua dimensi angka 0 – 9 yang dibuat dengan tulisan tangan menggunakan aplikasi paint. 
    
    ## Data yang digunakan berjumlah total 2000 gambar dengan 200 gambar per kelas datanya dan memiliki resolusi yang berbeda pula tiap kelas datanya.

    ## Aplikasi ini dibangun oleh Mikogizka Satria Kartika (1301194086) dan Dr. Putu Harry Gunawan, S.Si., M.Si., M.Sc. sebagai dosen pembimbing.
    """
    st.write(info_text)

elif choice == "Tentang Data":
    st.title("Tentang Data")
    st.write("""
    ## Berikut adalah contoh data yang digunakan dalam proses pembangunan aplikasi ini.
""")
    image1 = Image.open('01.jpg')
    image2 = Image.open('23.jpg')
    image3 = Image.open('45.jpg')
    image4 = Image.open('67.jpg')
    image5 = Image.open('89.jpg')
    
    st.image(image1, caption='Contoh dataset angka 0 dan 1', use_column_width=True)
    st.image(image2, caption='Contoh dataset angka 2 dan 3', use_column_width=True)
    st.image(image3, caption='Contoh dataset angka 4 dan 5', use_column_width=True)
    st.image(image4, caption='Contoh dataset angka 6 dan 7', use_column_width=True)
    st.image(image5, caption='Contoh dataset angka 8 dan 9', use_column_width=True)
    


