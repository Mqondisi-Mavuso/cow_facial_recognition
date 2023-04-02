import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import IPython.display as ipd
%matplotlib inline

from google.colab import drive
drive.mount('/content/drive')

# This is for downloading the YOLOv8 cow face images and labels
# !pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="my_api_here")
project = rf.workspace("face-cow").project("thecowface")
dataset = project.version(1).download("yolov8")

!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="GzhyGHnKKy58hjhTCDdG")
project = rf.workspace("cowseg").project("thecowface-tj1no")
dataset = project.version(1).download("coco-segmentation")

#training the model to segment the cow face using custom dataset
%pip install ultralytics
from ultralytics import YOLO

model = YOLO("/content/drive/MyDrive/Colab_Notebooks/runs/detect/train/weights/best.pt")
# Training the model
#!yolo train model=yolov8n-seg.pt data=coco128.yaml epochs=3 imgsz=640

# displaying the image
def show_img(image_path):
  img = cv2.imread(image_path)
  fig, ax = plt.subplots(figsize=(15,15))
  plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
  ax.grid(False)           # remove the grid lines
  plt.show()

#vliadate the model on images it has not seen

model = YOLO("/content/runs/detect/train2/weights/best.pt")  # load a custom model

#Validate the model
!yolo task=detect mode=val model="/content/runs/detect/train2/weights/best.pt" data=data.yaml

# Face detection and cropping
from os import listdir, path

the_path = "/content/drive/MyDrive/DATA/train/c9"
image_filenames = listdir(the_path)

for filename in image_filenames:
    image_path = path.join(the_path, filename)
    image = cv2.imread(image_path)
    results = model(image)  # predict on an image
    a = results[0].boxes.boxes
    px = pd.DataFrame(a).astype("float")

    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])

        if (x1 != 0) and (x2 != 0):
            # Crop the object from the image
            object_image = image[y1:y2, x1:x2]

            # Save the cropped image as a new file in the same directory as the original image
            new_filename = f"{filename[:-4]}_cropped_{index}.jpg"
            cv2.imwrite(path.join(the_path, new_filename), object_image)
        else:
            continue

# removing the uncropped images
import os

folder_path = "/content/drive/MyDrive/DATA/train/c9"

for file_name in os.listdir(folder_path):
    if file_name.endswith(".jpg") and "cr" not in file_name:
        os.remove(os.path.join(folder_path, file_name))

################################## Cow Face Segmentation Above #########################
################################## Cow Face Identification below #######################

# import the videos to read each frame
cap = cv2.VideoCapture("DATA/test/c22/IMG_3751.MOV")

# total number of frames
num_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#getting the frames from each video
def get_frames(num_frame):
    for frame in range(num_frame//2):
        ret, img = cap.read()
        # this rotates the image before writing to file
        rotated_img = cv2.rotate(img, cv2.ROTATE_180)
        if ret == False:
            break
        else:
            path = r"DATA/test/c22"
            name = f'/b_c{frame + 1}.jpg'
            # write the file
            cv2.imwrite(f'{path}{name}', rotated_img)

# this call will write the images to our training folder
get_frames(num_frame=num_frame)
cap.release()

from keras.preprocessing.image import ImageDataGenerator

image_gen = ImageDataGenerator(rotation_range=30,
                               width_shift_range=0.1,
                               height_shift_range=0.1,
                               rescale=1/255,
                               shear_range=0.2,
                               zoom_range=0.2,
                               horizontal_flip=True,
                               fill_mode="nearest"
                                )

#choosing the folder to connect to
image_gen.flow_from_directory("DATA/train/")

# we'll use this shape for all the images
input_shape = (550, 960, 3)

#preparing the image
the_cow = cv2.imread('DATA/train/c1/a_c102.jpg')
the_cow = cv2.cvtColor(the_cow, cv2.COLOR_BGR2RGB)

resized = cv2.resize(the_cow, (550, 960), interpolation = cv2.INTER_AREA)

from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Conv2D, MaxPooling2D, Dense

# declaring the CNN model
model = Sequential()
# adding our convulutional nueral network layers
model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=input_shape, activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64, kernel_size=(3, 3), input_shape=input_shape, activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64, kernel_size=(3, 3), input_shape=input_shape, activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(128))
model.add(Activation("relu"))

# the dropout layer reduces overfitting by randomly turning off nuerons during training
model.add(Dropout(0.5))

# the output is binary, either cow1 or cow.. hence Dense layer is just one
model.add(Dense(1))
model.add(Activation("sigmoid"))

model.compile(loss="binary_crossentropy",
              optimizer="adam",
              metrics=["accuracy"]
              )

model.summary()

# to see how the data classes will be presented
train_img_gen.class_indices

test_img_gen = image_gen.flow_from_directory("DATA/test/",
                                             target_size=input_shape[:2],
                                             batch_size=batch_size,
                                             class_mode="categorical")

# training the model on our images
results = model.fit(train_img_gen, epochs=5, steps_per_epoch=150,validation_data=test_img_gen, validation_steps=12)

# saving the model
#model.save("Cow_model.h5")
# now we evaluate the model
plt.plot(results.history["accuracy"])