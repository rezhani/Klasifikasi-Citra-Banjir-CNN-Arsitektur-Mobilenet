import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input

#load model yang sudah dilatih sebelumnya
model = keras.models.load_model('modelklasifikasibanjir.h5')

#label
labels = ['Banjir', 'Bermasalah', 'Tidak Banjir']

#scraping media cctv dari url
from html2image import Html2Image
hti = Html2Image(output_path='evaluate')
hti.screenshot(url=['https://diskominfo.samarindakota.go.id/api/cctv/simpang-lembuswana',
                    'https://diskominfo.samarindakota.go.id/api/cctv/simpang-lembuswana-analytic',
                    'https://diskominfo.samarindakota.go.id/api/cctv/simpang-3-kebun-agung',
                    'https://diskominfo.samarindakota.go.id/api/cctv/simpang-gn-kapur',
                    'https://diskominfo.samarindakota.go.id/api/cctv/simpang-a-yani-gatsu',
                    'https://diskominfo.samarindakota.go.id/api/cctv/simpang-air-hitam',
                    'https://diskominfo.samarindakota.go.id/api/cctv/simpang-pasundan',
                    'https://diskominfo.samarindakota.go.id/api/cctv/simpang-air-putih',
                    'https://diskominfo.samarindakota.go.id/api/cctv/simpang-antasari-siradj-salman',
                    'https://diskominfo.samarindakota.go.id/api/cctv/simpang-sempaja'], save_as='realtime.jpg')

def preprocess_image(file):
    img_path = 'evaluate/'
    img = image.load_img(img_path + file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array_expanded_dims)

#library image untuk membaca gambar dan display menampilkan gambar
from IPython.display import Image, display

#deklarasi list untuk setiap gambar ke variabel
listrealtime=['evaluate/realtime_0.jpg','evaluate/realtime_1.jpg','evaluate/realtime_2.jpg','evaluate/realtime_3.jpg',
              'evaluate/realtime_4.jpg','evaluate/realtime_5.jpg','evaluate/realtime_6.jpg','evaluate/realtime_7.jpg',
              'evaluate/realtime_8.jpg','evaluate/realtime_9.jpg']

#melakukan iterasi terhadap setiap gambar dalam list realtime
for realtime in listrealtime:
    display(Image(filename=realtime, width=500, height=350))


#%%

#melakukan preposes gambar dan membuat prediksi
preprocessed_image = preprocess_image('realtime_9.jpg')
predictions = model.predict(preprocessed_image)
print(predictions)
#mendapatkan kelas dengan nilai prediksi tertinggi untuk salah satu dari tiga kelas array
result = np.argmax(predictions)
print(result)
#menampilkan kelas prediksi sesuai label
hasil = labels[result]
print('hasil klasifikasi: ', hasil)

#%%

#menyimpan gambar yang baru diklasifikasi

from PIL import Image
image_path = 'evaluate/realtime_8.jpg'
img = Image.open(image_path)
img = img.convert('RGB')
img = img.resize((224, 224)) 
img.save('evaluate/realtime.jpg')

#memindah file yang disimpan ke direktori train

import datetime
import os
date = datetime.datetime.today().strftime ('%Y-%b-%d-%H-%M-%S')
if result==0:
    os.rename(r'evaluate/realtime.jpg',r'dataset/train/Banjir/realtime'+str(date)+'.jpg')
elif result==1:
    os.rename(r'evaluate/realtime.jpg',r'dataset/train/Tidak Banjir/realtime'+str(date)+'.jpg')