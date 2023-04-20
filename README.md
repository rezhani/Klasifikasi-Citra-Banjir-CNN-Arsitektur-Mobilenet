# Program Klasifikasi Citra Banjir pada CCTV Kota Samarinda
# Hal pertama yang diperlukan adalah
Nemiliki program IDE seperti jupyter notebook atau spyder atau yang lain tergantung kenyamanan masing2
Python terinstall dengan library Tensorflow, Keras, Numpy, Html2image.
# Kedua
Buat folder 'dataset' pada tempat yang sama dengan menyimpan program,
lalu dalam folder 'dataset' buat lagi folder yakni 'train', 'valid', 'test'.
Dan untuk masing2 folder 'train', 'valid', 'test'. Buat lagi 3 folder didalamnya sesuai dengan nama kelas yang akan kita gunakan.
Saya disini menggunakan 'Banjir', 'Tidak Banjir', 'Bermasalah'.
Contoh:
\n /dataset
  /train
    /Banjir
    /Tidak Banjir
    /Bermasalah
  /valid
    ..
    ..
    ..
  /test
    ..
    ..
    ..
# Ketiga
melatih program dan mendapatkan file model 'modelklasifikasibanjir.h5' yang akan digunakan untuk uji program.
# Keempat
menguji program dengan menggunakan model sebelumnya. Melihat akurasi setiap kelas label yang diprediksi pada citra scraping CCTV.
