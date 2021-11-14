import tensorflow.keras.applications.densenet
from keras.layers.core import activation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras import layers, models
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
gpus = tensorflow.config.experimental.list_physical_devices('GPU')
tensorflow.config.experimental.set_memory_growth(gpus[0], True)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", "-d",
                    default='datasets' ,
                    help="dataset to train the model on")
args = parser.parse_args()



current_full_dir = os.getcwd()
print("Current working directory: " + current_full_dir)
if current_full_dir.split("/")[-1] == "src":
    root = current_full_dir[:-4]
    os.chdir(root)
    print("Changed working directory to: " + root)


NUM_CLASS = 3
class_names = ['with_mask', 'without_mask', 'incorrect_mask']
if args.dataset == "datasets":
    NUM_CLASS = 3
    class_names = ['with_mask', 'without_mask', 'incorrect_mask']

learning_rate = 1e-4
EPOCHS = 50
BATCH_SIZE = 64
IMG_SIZE = 224
dataset_path = "Put your Absolute Path here"

model_save_path = "./ModelResult/data/datasets/model-pro-detect-mask6.h5"
figure_save_path = "./figures/results-graph1.jpg"

print("Num of classes: " + str(NUM_CLASS))
print("Classes: " + str(class_names))
print("Dataset path: " + dataset_path)
print("Figure save path: " + figure_save_path)


data_generator = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest",
    preprocessing_function=preprocess_input,
    validation_split=0.2)


train_generator = data_generator.flow_from_directory(
    dataset_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True,
    subset='training')


validation_generator = data_generator.flow_from_directory(
    dataset_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False,
    subset='validation')


base_model = InceptionV3(weights="imagenet", include_top=False, input_tensor=Input(shape=(IMG_SIZE, IMG_SIZE, 3)))


flatten_layer = layers.Flatten(input_shape=(IMG_SIZE,IMG_SIZE))
dense_layer1 = layers.Dense(256,activation='relu')
dense_layer2 = layers.Dense(128,activation='relu')
dense_layer3 = layers.Dense(64,activation='relu')
dense_layer4 = layers.Dense(32,activation='relu')
dropout = Dropout(0.5)
predictions_layer = layers.Dense(NUM_CLASS, activation='softmax')
model = models.Sequential([
    base_model,
    flatten_layer,
    dense_layer1,
    dense_layer2,
    dense_layer3,
    dense_layer4,
    dropout,
    predictions_layer
])
model.summary()


for layer in base_model.layers:
    layer.trainable = False


print("[INFO] compiling model...")
# model = load_model("./ModelResult/data/datasets/model-pro-detect-mask1.h5")
# print("Load pretrained model..." + str(model))
opt = Adam(learning_rate=learning_rate)

model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])



early_stopping = EarlyStopping(
    monitor='val_accuracy',
    min_delta=0.0001,
    patience=3,
    verbose=1,
    mode='max',
    baseline=None,
    restore_best_weights=True)





print("[INFO] training head...")
# checkpoint_path = './ModelResult/data/datasets/model-pro-detect-mask1.h5'
# checkpoint = ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', verbose=1,
#                              save_best_only=True, save_weights_only=False, mode='auto', save_freq=1)
H = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    callbacks=[early_stopping],
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=EPOCHS)


model.save(model_save_path)


prediction = model.predict_generator(
    generator=validation_generator,
    verbose=1)
print('prediction: '+str(prediction))
print('Done')
y_pred = np.argmax(prediction, axis=1)
print("Classification Report:")
print(classification_report(validation_generator.classes, y_pred, target_names=class_names))


plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, len(H.history["loss"])), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, len(H.history["val_loss"])), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, len(H.history["accuracy"])), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, len(H.history["val_accuracy"])), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(figure_save_path)
