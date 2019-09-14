from keras.models import load_model
from matplotlib import pyplot as plt
from time import time
import numpy as np
import os
import cv2

file_path = 'saved/unet.h5'
model = load_model(file_path)

test_images = []
image_src = 'data/test'
for file in os.listdir(image_src):
    file_path = os.path.join(image_src, file)
    test_images.append(file_path)

# 选择测试图片
img_path = test_images[0]

# 绘制原图
plt.subplot(121)
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (224, 224))

# 绘制Mask
plt.imshow(img.astype('uint8'))
plt.subplot(122)
img = np.reshape(img, (1, 224, 224, 3)) / 255.0
start = time()
pred = model.predict(img)
end = time()
print('Time:%.4fs' % (end - start))
pred = np.reshape(pred, (224, 224))
plt.imshow(pred, cmap='gray')