import numpy as np
import tensorflow as tf
from PIL import Image
from PIL import ImageFilter
import requests
import matplotlib.pyplot as plt
import os

print(tf.config.list_physical_devices('GPU'))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # для правильной работы tensorflow

img_url = 'https://www.humanesociety.org/sites/default/files/styles/768x326/public/2018/08/kitten-440379.jpg'

img = Image.open(requests.get(img_url, stream=True).raw)

plt.figure()
plt.imshow(img)
plt.axis('off')


#Y = .2126 * R^gamma + .7152 * G^gamma + .0722 * B^gamma
img_gs = img.convert("L") # конвертацию изображения оттенки серого

img_px = img_gs.load()

#Ядро Лапласиана ядро размером 3x3 с коэффициентами (0,-1,0,-1,4,-1,0,-1,0) изменение яркости пикселей
Laplacian = ImageFilter.Kernel((3,3), (0,-1,0,-1,4,-1,0,-1,0), scale=0.1, offset=1)
img_la = img_gs.filter(Laplacian)
plt.figure()
plt.imshow(img_la)
plt.axis('off')

scale = 1.1
input_tensor = tf.convert_to_tensor(img_gs, dtype=tf.float32) # конвертируем наши оттенки серого в тензор
# создаем константный тензор с коэффициентами [[0, -1, 0], [-1, 4, -1], [0, -1, 0]]
laplacian_filter = tf.constant(np.array([[0, -1, 0], [-1, 4*scale, -1], [0, -1, 0]]), dtype=tf.float32)
# выполняем двухмерную свертку
filtered_tensor = tf.nn.conv2d(input_tensor[None, :, :, None], laplacian_filter[:, :, None, None], strides=[1, 1, 1, 1], padding='SAME')
# ограничения значений тензора
filtered_tensor = tf.clip_by_value(filtered_tensor, 0.0, 255.0)
# убирает значения после ,
#filtered_tensor = tf.cast(filtered_tensor, tf.uint8)
#возвращает срез из тензора по заданным осям и преобразует в numpy
output_image = filtered_tensor[0, :, :, 0].numpy()

plt.figure()
plt.imshow(output_image)
plt.axis('off')
plt.show()