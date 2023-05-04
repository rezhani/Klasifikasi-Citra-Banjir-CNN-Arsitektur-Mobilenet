# Program Klasifikasi Citra Banjir pada CCTV Kota Samarinda
Implementasi deep learning dan cnn untuk klasifikasi citra 3 kelas dengan custom dataset arsitektur mobilenet. 

![](https://media.giphy.com/media/8Dmjv4tkbIvAJGclcL/giphy.gif)
# Hal pertama yang diperlukan adalah
Memiliki program IDE seperti jupyter notebook atau spyder atau yang lain tergantung kenyamanan masing2
Python terinstall dengan library Tensorflow, Keras, Numpy, Html2image.
# Kedua
Kumpulkan dataset citra banjir dan tidak banjir. Nah berhubung data yang saya dapatkan berupa video perlu kita ekstrak dahulu frame gambar dari videonya dengan 'ekstrakvidkegambar.ipynb'
Buat folder sesuai dengan nama kelas yang akan digunakan. Saya disini menggunakan 3 kelas yaitu 'Banjir', 'Tidak Banjir', 'Bermasalah'. Maka buat sebanyak 3 folder.
Lakukan Split Dataset dengan 'traintestvalidsplit.py'
# Ketiga
Melatih program dengan 'latih3class(versi pendek).py' dan mendapatkan file model 'modelklasifikasibanjir.h5' yang akan digunakan untuk uji program.
# Keempat
Menguji program dengan 'uji3class(versi pendek).py' menggunakan model sebelumnya. Dan Melihat akurasi setiap kelas label yang diprediksi pada citra scraping CCTV.
# Kelima
Tahap ini melakukan distribusi agar bisa digunakan Enduser. Saya sendiri memilih menyalurkan program ini lewat Telegram Bot yang bisa kalian lihat di [sini](https://t.me/deteksibanjirbot). Sebenernya kepikiran juga sih menggunakan web framework seperti flask ataupun streamlit. Nanti liat deh.
