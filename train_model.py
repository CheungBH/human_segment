from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from keras.optimizers import *
import os
from model.model import unet

data_gen_args = dict(featurewise_center=False,
                     featurewise_std_normalization=False,
                     rescale=1 / 255.0,
                     rotation_range=90,
                     width_shift_range=0.2,
                     height_shift_range=0.2,
                     zoom_range=0.2)

# 创建图像和其对应Mask的生成器
train_image_datagen = ImageDataGenerator(**data_gen_args)
train_mask_datagen = ImageDataGenerator(**data_gen_args)

# 设置相同seed，确保image与mask对应一致
seed = 1
batch_size = 32

# specify the params
train_image_generator = train_image_datagen.flow_from_directory(
    'data/train/images',
    class_mode=None,
    batch_size=batch_size,
    target_size=(224, 224),
    color_mode='rgb',
    seed=seed)
train_mask_generator = train_mask_datagen.flow_from_directory(
    'data/train/masks',
    class_mode=None,
    batch_size=batch_size,
    target_size=(224, 224),
    color_mode='grayscale',
    seed=seed)

# 结合image和mask作为训练数据生成器
train_generator = zip(train_image_generator, train_mask_generator)

# specify the params
data_gen_args = dict(featurewise_center=False,
                     featurewise_std_normalization=False,
                     rescale=1 / 255.0)

# 创建图像和其对应Mask的生成器
val_image_datagen = ImageDataGenerator(**data_gen_args)
val_mask_datagen = ImageDataGenerator(**data_gen_args)

# 设置相同seed，确保image与mask对应一致
seed = 1
batch_size = 32

val_image_generator = val_image_datagen.flow_from_directory(
    'data/val/images',
    class_mode=None,
    batch_size=batch_size,
    target_size=(224, 224),
    color_mode='rgb',
    seed=seed)
val_mask_generator = val_mask_datagen.flow_from_directory(
    'data/val/masks',
    class_mode=None,
    batch_size=batch_size,
    target_size=(224, 224),
    color_mode='grayscale',
    seed=seed)

# 结合image和mask作为训练数据生成器
val_generator = zip(val_image_generator, val_mask_generator)





model= unet(input_size=(224,224,3))

from keras.utils import multi_gpu_model
# 以数据并行的方式执行多GPU计算
parallel_model = multi_gpu_model(model, gpus=2)

# 编译并行模型
opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=10e-8, decay=0.001, amsgrad=False)
parallel_model.compile(loss='binary_crossentropy',metrics=['accuracy'],optimizer=opt)

from keras.callbacks import EarlyStopping, ModelCheckpoint

# 配置CheckPoint
check_point = ModelCheckpoint(filepath='./Models/unet.h5', monitor='val_acc',
                             save_weights_only=False,
                             verbose=2,
                             save_best_only=True,
                             period=1)
# 配置Early Stopping
early_stopping = EarlyStopping(monitor='val_acc',
                               patience=10,
                               verbose=0,
                               mode='auto')

# 计算相关参数：steps_per_epoch
steps_train = len(train_image_generator)
steps_val = len(val_image_generator)

# 训练模型
log = parallel_model.fit_generator(train_generator,
                                   epochs=100, verbose=1,
                                   steps_per_epoch=steps_train,
                                   validation_data=val_generator,
                                   validation_steps=steps_val,
                                   callbacks=[early_stopping, check_point])

# 创建模型保存文件夹
folder_path = 'saved'
if not os.path.exists(folder_path):
    os.mkdir(folder_path)

# 保存模型到指定文件夹
file_path = os.path.join(folder_path, 'unet.h5')
model.save(file_path)

plt.plot(log.history['acc'])
plt.plot(log.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='lower right')
plt.tight_layout()

plt.plot(log.history['loss'])
plt.plot(log.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper right')
plt.tight_layout()
