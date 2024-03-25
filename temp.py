import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model  # Import load_model to load an existing model

# Constants
IMAGE_WIDTH, IMAGE_HEIGHT = 224, 224
BATCH_SIZE = 32
NUM_CLASSES = len(os.listdir('plantVillage'))

# Load the existing trained model
model = load_model('plant_disease_model.h5')  # Load the model from the saved file

# Save the model without retraining
model.save('plant_disease_model.keras')
