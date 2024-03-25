import numpy as np
import os
import pickle
import cv2
from os import listdir
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization

from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


EPOCHS = 15
INIT_LR = 1e-3
BS = 64
default_image_size = tuple((256,256))
image_size = 0
directory_root = 'PlantVillage'
width=256
height=256
depth=3

NUM_CLASSES = len(os.listdir('PlantVillage'))

def convert_image_to_array(image_dir):
    try:
        image = cv2.imread(image_dir)
        if image is not None :
            image = cv2.resize(image, default_image_size)   
            return img_to_array(image)
        else :
            return np.array([])
    except Exception as e:
        print(f"Error : {e}")
        return None



image_list, label_list = [], []
try:
    print("[INFO] Loading images ...")
    
    # List the subdirectories within the 'PlantVillage' folder
    plant_folders = os.listdir(directory_root)
    
    for plant_folder in plant_folders:
        # Construct the full path to the current plant folder
        plant_folder_path = os.path.join(directory_root, plant_folder)
        
        # Check if it's a directory
        if os.path.isdir(plant_folder_path):
            print(f"[INFO] Processing {plant_folder} ...")
            
            # List the image files in the current plant folder
            plant_disease_image_list = os.listdir(plant_folder_path)
            
            for image_filename in plant_disease_image_list[:200]:
                # Construct the full path to the current image
                image_path = os.path.join(plant_folder_path, image_filename)
                
                # Check if it's a file and has a valid image extension (e.g., .jpg)
                if os.path.isfile(image_path) and image_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    image = cv2.imread(image_path)
                    if image is not None:
                        # Resize the image and append it to the list
                        image = cv2.resize(image, (224, 224))
                        image_list.append(image)
                        label_list.append(plant_folder)
    
    print("[INFO] Image loading completed")

except Exception as e:
    print(f"Error : {e}")

image_size = len(image_list)
print(image_size)

# Now you can convert the image data to a numpy array
np_image_list = np.array(image_list, dtype=np.float16) / 255.0

label_binarizer = LabelBinarizer()
image_labels = label_binarizer.fit_transform(label_list)
pickle.dump(label_binarizer,open('label_transform.pkl', 'wb'))
n_classes = len(label_binarizer.classes_)

print(label_binarizer.classes_)


print("[INFO] Spliting data to train, test")
x_train, x_test, y_train, y_test = train_test_split(np_image_list, image_labels, test_size=0.2, random_state = 42) 


aug = ImageDataGenerator(
    rotation_range=25, width_shift_range=0.1,
    height_shift_range=0.1, shear_range=0.2, 
    zoom_range=0.2,horizontal_flip=True, 
    fill_mode="nearest")

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(NUM_CLASSES, activation='softmax')
])


opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
# distribution
model.compile(loss="binary_crossentropy", optimizer=opt,metrics=["accuracy"])
# train the network
print("[INFO] training network...")


history = model.fit(
    aug.flow(x_train, y_train, batch_size=BS),
    validation_data=(x_test, y_test),
    steps_per_epoch=len(x_train) // BS,
    epochs=EPOCHS,
    verbose=1
)

print("[INFO] Calculating model accuracy")
scores = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {scores[1]*100}")

print("[INFO] Saving model...")
pickle.dump(model,open('cnn_model.pkl', 'wb'))