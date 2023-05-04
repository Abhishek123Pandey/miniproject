import tensorflow as tf
import os
from keras import models , layers
import keras.preprocessing.image
import matplotlib.pyplot as plt
IMAGE_SIZE=256
batch_size=32
channel=3
trainsize=0.8
valid_size=0.5
test_size=0.1
epoch=25
dataset=tf.keras.preprocessing.image_dataset_from_directory(
    r"C:\Users\abhis\Downloads\archive (2)\PlantVillage",
      shuffle=True,
      image_size=(IMAGE_SIZE,IMAGE_SIZE),
      batch_size=batch_size
)
class_names=dataset.class_names
def dataset_divider(ds,train_split=0.8,val_split=0.1,test_split=0.1,shuffle=True,shuffle_size=10000):
    ds_size=len(ds)
    if shuffle:
        ds=ds.shuffle(shuffle_size,seed=12)
    train_size=int(train_split*ds_size)
    val_size=int(val_split*ds_size)
    train_ds=ds.take(train_size)
    val_ds=ds.skip(train_size).take(val_size)
    test_ds=ds.skip(train_size).skip(val_size)
    
     
    return train_ds,val_ds,test_ds

train_ds,val_ds,test_ds=dataset_divider(dataset)
train_ds=train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds=val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds=test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
resize_and_rescale=tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.Resizing(IMAGE_SIZE,IMAGE_SIZE),
    tf.keras.layers.experimental.preprocessing.Rescaling(1.0/255)

])
data_augmentation=tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.experimental.preprocessing.Rescaling(0.2)

])
input_shape=(batch_size,IMAGE_SIZE,IMAGE_SIZE,channel)
n_classes=15
model = tf.keras.models.Sequential([   resize_and_rescale,
    data_augmentation,
    tf.keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=input_shape),
    tf.keras.layers.MaxPooling2D((2,2)),
     tf.keras.layers.Conv2D(64,(3,3),activation='relu',input_shape=input_shape),
    tf.keras.layers.MaxPooling2D((2,2)),
     tf.keras.layers.Conv2D(64,(3,3),activation='relu',input_shape=input_shape),
    tf.keras.layers.MaxPooling2D((2,2)),
     tf.keras.layers.Conv2D(64,(3,3),activation='relu',input_shape=input_shape),
    tf.keras.layers.MaxPooling2D((2,2)),
     tf.keras.layers.Conv2D(64,(3,3),activation='relu',input_shape=input_shape),
    tf.keras.layers.MaxPooling2D((2,2)),
     tf.keras.layers.Conv2D(64,(3,3),activation='relu',input_shape=input_shape),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
     tf.keras.layers.Dense(64,activation='relu'),
      tf.keras.layers.Dense(n_classes,activation='softmax')                              
                                
                                    
    
])
model.build(input_shape=input_shape)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])
history=model.fit(
    train_ds,
    epochs=epoch,
    batch_size=batch_size,
    verbose=1,
    validation_data=val_ds)
score=model.evaluate(train_ds)
model_versio=1
model.save(r"C:\Users\abhis\OneDrive\Desktop\miniproject")