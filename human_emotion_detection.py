# -*- coding: utf-8 -*-
"""Human_Emotion_Detection.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Gu6QP0czDkpjNKPYSpCWYW7qHh4Skrv9
"""

import tensorflow as tf### models
import numpy as np### math computations
import matplotlib.pyplot as plt### plotting bar chart
import sklearn### machine learning library
import cv2## image processing
from sklearn.metrics import confusion_matrix, roc_curve### metrics
import seaborn as sns### visualizations
import datetime
import pathlib
import io
import os
import time
import random
from google.colab import files
from PIL import Image
import albumentations as A
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
import matplotlib.cm as cm
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import (GlobalAveragePooling2D, Activation, MaxPooling2D, Add, Conv2D, MaxPool2D, Dense,
                                     Flatten, InputLayer, BatchNormalization, Input, Embedding, Permute,
                                     Dropout, RandomFlip, RandomRotation, LayerNormalization, MultiHeadAttention,
                                     RandomContrast, Rescaling, Resizing, Reshape)
from tensorflow.keras.losses import BinaryCrossentropy,CategoricalCrossentropy, SparseCategoricalCrossentropy
from tensorflow.keras.metrics import Accuracy,TopKCategoricalAccuracy, CategoricalAccuracy, SparseCategoricalAccuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (Callback, CSVLogger, EarlyStopping, LearningRateScheduler,
                                        ModelCheckpoint, ReduceLROnPlateau)
from tensorflow.keras.regularizers  import L2, L1
from tensorflow.train import BytesList, FloatList, Int64List
from tensorflow.train import Example, Features, Feature
from google.colab import drive

"""# Setup kaggle api and download the dataset"""

!pip install -q kaggle

! mkdir ~/.kaggle
! cp kaggle.json ~/.kaggle/

!chmod 600 /root/.kaggle/kaggle.json

!kaggle datasets download -d muhammadhananasghar/human-emotions-datasethes

"""# Extract the dataset"""

!unzip "/content/human-emotions-datasethes.zip" -d "/content/dataset/"

"""# Setup the path for data"""

train_directory = "/content/dataset/Emotions Dataset/Emotions Dataset/train"
val_directory = "/content/dataset/Emotions Dataset/Emotions Dataset/test"

CLASS_NAMES = ["angry", "happy", "sad"]

!pip install wandb

import wandb
from wandb.integration.keras import WandbCallback
print(wandb.__version__)

!wandb login

wandb.init(project='Human-Emotion-Detection')

"""# Hyperparameter Configuration"""

wandb.config = {
    "BATCH_SIZE": 32,
    "IM_SIZE": 256,
    "LEARNING_RATE": 1e-3,
    "N_EPOCHS": 20,
    "DROPOUT_RATE": 0.0,
    "REGULARIZATION_RATE": 0.0,
    "N_FILTERS": 6,
    "KERNEL_SIZE": 3,
    "N_STRIDES": 1,
    "POOL_SIZE": 2,
    "N_DENSE_1": 1024,
    "N_DENSE_2": 128,
    "NUM_CLASSES": 3,
    "PATCH_SIZE": 16,
    "PROJ_DIM": 768,
    "CLASS_NAMES": ["angry", "happy", "sad"],
}

CONFIGURATION = wandb.config

"""# TensorFlow dataset for train and val"""

train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_directory,
    labels='inferred',
    label_mode='categorical',
    class_names=CONFIGURATION["CLASS_NAMES"],
    color_mode='rgb',
    batch_size=CONFIGURATION["BATCH_SIZE"],
    image_size=(CONFIGURATION["IM_SIZE"], CONFIGURATION["IM_SIZE"]),
    shuffle=True,
    seed=99,
)



val_dataset = tf.keras.utils.image_dataset_from_directory(
    val_directory,
    labels='inferred',
    label_mode='categorical',
    class_names=CONFIGURATION["CLASS_NAMES"],
    color_mode='rgb',
    batch_size=CONFIGURATION["BATCH_SIZE"],
    image_size=(CONFIGURATION["IM_SIZE"], CONFIGURATION["IM_SIZE"]),
    shuffle=True,
    seed=99,
)

for i in val_dataset.take(1):
  print(i)

"""# Visualize the dataset"""

plt.figure(figsize = (12, 12))

for images, labels in train_dataset.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(CONFIGURATION["CLASS_NAMES"][np.argmax(labels[i])])
    plt.axis("off")

"""# Data Augmentaion"""

### tf.keras.layer augment
augment_layers = tf.keras.Sequential([
  RandomRotation(factor = (-0.025, 0.025)),
  RandomFlip(mode='horizontal',),
  RandomContrast(factor=0.1),
])

def augment_layer(image, label):
  return augment_layers(image, training=True), label

"""#### Cut mix augmentaion"""

def box(lamda):

  r_x = tf.cast(tfp.distributions.Uniform(0, CONFIGURATION["IM_SIZE"]).sample(1)[0], dtype = tf.int32)
  r_y = tf.cast(tfp.distributions.Uniform(0, CONFIGURATION["IM_SIZE"]).sample(1)[0], dtype = tf.int32)

  r_w = tf.cast(CONFIGURATION["IM_SIZE"]*tf.math.sqrt(1-lamda), dtype = tf.int32)
  r_h = tf.cast(CONFIGURATION["IM_SIZE"]*tf.math.sqrt(1-lamda), dtype = tf.int32)

  r_x = tf.clip_by_value(r_x - r_w//2, 0, CONFIGURATION["IM_SIZE"])
  r_y = tf.clip_by_value(r_y - r_h//2, 0, CONFIGURATION["IM_SIZE"])

  x_b_r = tf.clip_by_value(r_x + r_w//2, 0, CONFIGURATION["IM_SIZE"])
  y_b_r = tf.clip_by_value(r_y + r_h//2, 0, CONFIGURATION["IM_SIZE"])

  r_w = x_b_r - r_x
  if(r_w == 0):
    r_w  = 1

  r_h = y_b_r - r_y
  if(r_h == 0):
    r_h = 1

  return r_y, r_x, r_h, r_w

def cutmix(train_dataset_1, train_dataset_2):
  (image_1,label_1), (image_2, label_2) = train_dataset_1, train_dataset_2

  lamda = tfp.distributions.Beta(2,2)
  lamda = lamda.sample(1)[0]

  r_y, r_x, r_h, r_w = box(lamda)
  crop_2 = tf.image.crop_to_bounding_box(image_2, r_y, r_x, r_h, r_w)
  pad_2 = tf.image.pad_to_bounding_box(crop_2, r_y, r_x, CONFIGURATION["IM_SIZE"], CONFIGURATION["IM_SIZE"])

  crop_1 = tf.image.crop_to_bounding_box(image_1, r_y, r_x, r_h, r_w)
  pad_1 = tf.image.pad_to_bounding_box(crop_1, r_y, r_x, CONFIGURATION["IM_SIZE"], CONFIGURATION["IM_SIZE"])

  image = image_1 - pad_1 + pad_2

  lamda = tf.cast(1- (r_w*r_h)/(CONFIGURATION["IM_SIZE"]*CONFIGURATION["IM_SIZE"]), dtype = tf.float32)
  label = lamda*tf.cast(label_1, dtype = tf.float32) + (1-lamda)*tf.cast(label_2, dtype = tf.float32)

  return image, label

train_dataset_1 = train_dataset.map(augment_layer, num_parallel_calls = tf.data.AUTOTUNE)
train_dataset_2 = train_dataset.map(augment_layer, num_parallel_calls = tf.data.AUTOTUNE)

mixed_dataset = tf.data.Dataset.zip((train_dataset_1, train_dataset_2))

training_dataset = (
    mixed_dataset
    .map(cutmix, num_parallel_calls = tf.data.AUTOTUNE)
    .prefetch(tf.data.AUTOTUNE)
)

"""# Dataset Preparation"""

# training_dataset = (
#     train_dataset
#     .map(augment_layer, num_parallel_calls=tf.data.AUTOTUNE)
#     .prefetch(tf.data.AUTOTUNE)
# )



validation_dataset = (
    val_dataset.prefetch(tf.data.AUTOTUNE)
)

"""# Data Augmentaion"""

resize_rescale_layers = tf.keras.Sequential([
    Resizing(CONFIGURATION["IM_SIZE"], CONFIGURATION["IM_SIZE"]),
    Rescaling(1./255)
])

"""# Modeling"""

lenet_model = tf.keras.Sequential([

    # InputLayer(input_shape=(CONFIGURATION["IM_SIZE"], CONFIGURATION["IM_SIZE"], 3)),
    InputLayer(input_shape=(None, None, 3)),
    resize_rescale_layers,

    Conv2D(filters=CONFIGURATION["N_FILTERS"], kernel_size=CONFIGURATION["KERNEL_SIZE"], strides=CONFIGURATION["N_STRIDES"], activation="relu", kernel_regularizer=L2(CONFIGURATION["REGULARIZATION_RATE"])),
    BatchNormalization(),
    MaxPool2D(pool_size=CONFIGURATION["POOL_SIZE"], strides=CONFIGURATION["N_STRIDES"]*2),
    Dropout(CONFIGURATION["DROPOUT_RATE"]),

    Conv2D(filters=CONFIGURATION["N_FILTERS"]*2+4, kernel_size=CONFIGURATION["KERNEL_SIZE"], strides=CONFIGURATION["N_STRIDES"], activation="relu", kernel_regularizer=L2(CONFIGURATION["REGULARIZATION_RATE"])),
    BatchNormalization(),
    MaxPool2D(pool_size=CONFIGURATION["POOL_SIZE"], strides=CONFIGURATION["N_STRIDES"]*2),

    Flatten(),

    Dense(CONFIGURATION["N_DENSE_1"], activation="relu", kernel_regularizer=L2(CONFIGURATION["REGULARIZATION_RATE"])),
    BatchNormalization(),
    Dropout(CONFIGURATION["DROPOUT_RATE"]),

    Dense(CONFIGURATION["N_DENSE_2"], activation="relu", kernel_regularizer=L2(CONFIGURATION["REGULARIZATION_RATE"])),
    BatchNormalization(),

    Dense(CONFIGURATION["NUM_CLASSES"], activation="softmax")

])

lenet_model.summary()

"""# Training

#### Setting up the loss funciton and metrics
"""

loss_function = CategoricalCrossentropy()

metrics = [CategoricalAccuracy(name='accuracy'), TopKCategoricalAccuracy(k=3, name='top-3-accuracy')]

"""#### Compile the model"""

lenet_model.compile(
    optimizer = Adam(learning_rate=CONFIGURATION["LEARNING_RATE"]),
    loss = loss_function,
    metrics = metrics
)

"""#### Train the model"""

history = lenet_model.fit(
    training_dataset,
    validation_data = validation_dataset,
    epochs = CONFIGURATION["N_EPOCHS"],
    verbose = 1
)

"""# Plot Loss-Accuracy Curves"""

def plot_training_curves(history, save_path=None, figsize=(14, 5), title_fontsize=16, label_fontsize=12):
    """
    Plots training and validation loss and accuracy side by side.

    Args:
        history (History): Keras history object returned by model.fit().
        save_path (str, optional): If provided, saves the plot to this path.
        figsize (tuple): Size of the plot.
        title_fontsize (int): Font size for titles.
        label_fontsize (int): Font size for labels and legends.
    """
    # Extract metrics
    acc = history.history.get('accuracy')
    val_acc = history.history.get('val_accuracy')
    loss = history.history.get('loss')
    val_loss = history.history.get('val_loss')

    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=figsize)

    # Accuracy subplot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, 'b-', label='Train Accuracy')
    plt.plot(epochs_range, val_acc, 'r--', label='Validation Accuracy')
    plt.title('Model Accuracy', fontsize=title_fontsize)
    plt.xlabel('Epoch', fontsize=label_fontsize)
    plt.ylabel('Accuracy', fontsize=label_fontsize)
    plt.legend(fontsize=label_fontsize)
    plt.grid(True)

    # Loss subplot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, 'b-', label='Train Loss')
    plt.plot(epochs_range, val_loss, 'r--', label='Validation Loss')
    plt.title('Model Loss', fontsize=title_fontsize)
    plt.xlabel('Epoch', fontsize=label_fontsize)
    plt.ylabel('Loss', fontsize=label_fontsize)
    plt.legend(fontsize=label_fontsize)
    plt.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)

    plt.show()

# Save the loss-acc curves
plot_training_curves(history, save_path="Model-1-loss-acc-curves.png")

"""# Evaluate the model"""

lenet_model.evaluate(validation_dataset)

"""# Testing"""

test_image = cv2.imread('/content/dataset/Emotions Dataset/Emotions Dataset/test/sad/113923.jpg_brightness_2.jpg')
test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
plt.imshow(test_image)

im = tf.constant(test_image, dtype=tf.float32)
# im = tf.image.resize(im, (CONFIGURATION["IM_SIZE"], CONFIGURATION["IM_SIZE"]))  # Dont need this because our model have and inbuild resize rescale method
# im = im / 255.0
im = tf.expand_dims(im, axis=0)

CLASS_NAMES[(tf.argmax(lenet_model.predict(im), axis=-1).numpy()[0])]

"""#### Plot the predictions"""

def plot_predictions_grid(model, dataset, class_names, num_images=16, normalize=True, figsize=(12, 12)):
    """
    Plots a grid of images with predicted and true labels.

    Args:
        model (tf.keras.Model): Trained model.
        dataset (tf.data.Dataset): Dataset to visualize (e.g., validation_dataset).
        class_names (list): List of class names.
        num_images (int): Number of images to display (must be square number).
        normalize (bool): Whether to divide pixel values by 255.
        figsize (tuple): Size of the full figure.
    """
    plt.figure(figsize=figsize)

    # Take one batch
    for images, labels in dataset.take(1):
        for i in range(num_images):
            ax = plt.subplot(int(np.sqrt(num_images)), int(np.sqrt(num_images)), i + 1)
            img = images[i].numpy()
            label = tf.argmax(labels[i], axis=-1).numpy()

            # Prediction
            prediction = model(tf.expand_dims(images[i], axis=0), training=False)
            pred_label = tf.argmax(prediction, axis=-1).numpy()[0]

            # Normalize if needed
            img_display = img / 255.0 if normalize else img

            # Show image
            plt.imshow(img_display)
            plt.axis('off')

            # Title formatting
            correct = (label == pred_label)
            color = 'green' if correct else 'red'
            title = f"True: {class_names[label]}\nPred: {class_names[pred_label]}"
            ax.set_title(title, color=color, fontsize=10, pad=4)

        plt.tight_layout()
        plt.show()
        break  # Only one batch

plot_predictions_grid(
    model=lenet_model,
    dataset=validation_dataset,
    class_names=CONFIGURATION["CLASS_NAMES"],
    num_images=16,
    normalize=True
)

"""#### Confusion Matrix"""

def plot_confusion_matrix(model, dataset, class_names, normalize=False, figsize=(10, 8), cmap='Blues'):
    """
    Plots a beautiful confusion matrix using model predictions.

    Args:
        model: Trained Keras model.
        dataset: tf.data.Dataset (validation or test set).
        class_names: List of class names.
        normalize: If True, show percentages instead of raw counts.
        figsize: Size of the confusion matrix plot.
        cmap: Color map used for the heatmap.
    """
    true_labels = []
    pred_labels = []

    for images, labels in dataset:
        preds = model.predict(images, verbose=0)
        pred_indices = np.argmax(preds, axis=1)
        true_indices = np.argmax(labels.numpy(), axis=1)

        pred_labels.extend(pred_indices)
        true_labels.extend(true_indices)

    cm = confusion_matrix(true_labels, pred_labels)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'

    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap, xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix", fontsize=16)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.show()

plot_confusion_matrix(
    model=lenet_model,
    dataset=validation_dataset,
    class_names=CONFIGURATION["CLASS_NAMES"],
    normalize=True
)

from sklearn.metrics import classification_report
import pandas as pd

def classification_dashboard(model, dataset, class_names):
    """
    Prints classification report with precision, recall, F1-score.

    Args:
        model: Trained Keras model.
        dataset: tf.data.Dataset to evaluate.
        class_names: List of class names.
    """
    true_labels = []
    pred_labels = []

    for images, labels in dataset:
        preds = model.predict(images, verbose=0)
        pred_indices = np.argmax(preds, axis=1)
        true_indices = np.argmax(labels.numpy(), axis=1)

        pred_labels.extend(pred_indices)
        true_labels.extend(true_indices)

    # Generate and print classification report
    report = classification_report(true_labels, pred_labels, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    print("\nClassification Report:")
    display(report_df.round(2))

classification_dashboard(
    model=lenet_model,
    dataset=validation_dataset,
    class_names=CONFIGURATION["CLASS_NAMES"]
)

"""# Resnet model"""

class CustomConv2D(Layer):
  def __init__(self, n_filters, kernel_size, n_strides, padding='valid'):
    super(CustomConv2D, self).__init__(name='custom_conv2d')

    self.conv = Conv2D(
        filters=n_filters,
        kernel_size=kernel_size,
        activation='relu',
        strides=n_strides,
        padding=padding
    )
    self.batch_norm = BatchNormalization()

  def call(self, x, training=True):
    x = self.conv(x)
    x = self.batch_norm(x, training=training)
    return x

class ResidualBlock(Layer):
  def __init__(self, n_channels, n_strides=1):
    super(ResidualBlock, self).__init__(name='res_block')

    self.downsample = (n_strides != 1)

    self.custom_conv_1 = CustomConv2D(n_channels, 3, n_strides, padding="same")
    self.custom_conv_2 = CustomConv2D(n_channels, 3, 1, padding="same")

    self.activation = Activation('relu')

    if self.downsample:
      self.custom_conv_3 = CustomConv2D(n_channels, 1, n_strides)

  def call(self, input_tensor, training=True):
    x = self.custom_conv_1(input_tensor, training=training)
    x = self.custom_conv_2(x, training=training)

    if self.downsample:
      shortcut = self.custom_conv_3(input_tensor, training=training)
    else:
      shortcut = input_tensor

    x = Add()([x, shortcut])
    return self.activation(x)

class ResNet34(Model):
  def __init__(self,):
    super(ResNet34, self).__init__(name = 'resnet_34')

    self.conv_1 = CustomConv2D(64, 7, 2, padding = 'same')
    self.max_pool = MaxPooling2D(3,2)

    self.conv_2_1 = ResidualBlock(64)
    self.conv_2_2 = ResidualBlock(64)
    self.conv_2_3 = ResidualBlock(64)

    self.conv_3_1 = ResidualBlock(128, 2)
    self.conv_3_2 = ResidualBlock(128)
    self.conv_3_3 = ResidualBlock(128)
    self.conv_3_4 = ResidualBlock(128)

    self.conv_4_1 = ResidualBlock(256, 2)
    self.conv_4_2 = ResidualBlock(256)
    self.conv_4_3 = ResidualBlock(256)
    self.conv_4_4 = ResidualBlock(256)
    self.conv_4_5 = ResidualBlock(256)
    self.conv_4_6 = ResidualBlock(256)

    self.conv_5_1 = ResidualBlock(512, 2)
    self.conv_5_2 = ResidualBlock(512)
    self.conv_5_3 = ResidualBlock(512)

    self.global_pool = GlobalAveragePooling2D()

    self.fc_3 = Dense(CONFIGURATION["NUM_CLASSES"], activation = 'softmax')

  def call(self, x, training=True):
      x = self.conv_1(x, training=training)
      x = self.max_pool(x)

      x = self.conv_2_1(x, training=training)
      x = self.conv_2_2(x, training=training)
      x = self.conv_2_3(x, training=training)

      x = self.conv_3_1(x, training=training)
      x = self.conv_3_2(x, training=training)
      x = self.conv_3_3(x, training=training)
      x = self.conv_3_4(x, training=training)

      x = self.conv_4_1(x, training=training)
      x = self.conv_4_2(x, training=training)
      x = self.conv_4_3(x, training=training)
      x = self.conv_4_4(x, training=training)
      x = self.conv_4_5(x, training=training)
      x = self.conv_4_6(x, training=training)

      x = self.conv_5_1(x, training=training)
      x = self.conv_5_2(x, training=training)
      x = self.conv_5_3(x, training=training)

      x = self.global_pool(x)
      return self.fc_3(x)

resnet_34 = ResNet34()
resnet_34(tf.zeros([1,256,256,3]), training = False)
resnet_34.summary()

checkpoint_callback = ModelCheckpoint(
    'best_weights.keras',
    monitor='val_accuracy',
    mode = 'max',
    save_best_only=True,
    verbose=1
)

resnet_34.compile(
    optimizer = Adam(learning_rate=CONFIGURATION["LEARNING_RATE"]*10),
    loss = loss_function,
    metrics = metrics
)

history = resnet_34.fit(
    training_dataset,
    validation_data = validation_dataset,
    epochs = CONFIGURATION["N_EPOCHS"],
    verbose = 1,
    callbacks = [checkpoint_callback]
)

plot_training_curves(history)

plot_predictions_grid(
    model=resnet_34,
    dataset=validation_dataset,
    class_names=CONFIGURATION["CLASS_NAMES"],
    num_images=16,
    normalize=True
)

plot_confusion_matrix(
    model=resnet_34,
    dataset=validation_dataset,
    class_names=CONFIGURATION["CLASS_NAMES"],
    normalize=True
)

classification_dashboard(
    model=resnet_34,
    dataset=validation_dataset,
    class_names=CONFIGURATION["CLASS_NAMES"]
)

"""# Transfer Learning"""

backborn = tf.keras.applications.efficientnet.EfficientNetB4(
    include_top=False,
    weights='imagenet',
    input_shape=(CONFIGURATION["IM_SIZE"], CONFIGURATION["IM_SIZE"], 3),
)

backborn.summary()

backborn.trainable = False

pretrained_model = tf.keras.Sequential([
    Input(shape=(CONFIGURATION["IM_SIZE"], CONFIGURATION["IM_SIZE"], 3)),
    backborn,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(CONFIGURATION["N_DENSE_1"], activation='relu'),
    BatchNormalization(),
    tf.keras.layers.Dense(CONFIGURATION["N_DENSE_2"], activation='relu'),
    tf.keras.layers.Dense(CONFIGURATION["NUM_CLASSES"], activation='softmax')
])

pretrained_model.summary()

pretrained_model.compile(
    optimizer = Adam(learning_rate=CONFIGURATION["LEARNING_RATE"]),
    loss = loss_function,
    metrics = metrics
)

history = pretrained_model.fit(
    training_dataset,
    validation_data = validation_dataset,
    epochs = CONFIGURATION["N_EPOCHS"],
    verbose = 1,
    callbacks = [checkpoint_callback]
)

plot_training_curves(history)

plot_predictions_grid(
    model=pretrained_model,
    dataset=validation_dataset,
    class_names=CONFIGURATION["CLASS_NAMES"],
    num_images=16,
    normalize=True
)

plot_confusion_matrix(
    model=pretrained_model,
    dataset=validation_dataset,
    class_names=CONFIGURATION["CLASS_NAMES"],
    normalize=True
)

classification_dashboard(
    model=pretrained_model,
    dataset=validation_dataset,
    class_names=CONFIGURATION["CLASS_NAMES"]
)

"""# Evaluate the model"""

pretrained_model.evaluate(validation_dataset)

test_image = cv2.imread('/content/dataset/Emotions Dataset/Emotions Dataset/test/sad/113923.jpg_brightness_2.jpg')
test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
plt.imshow(test_image)

im = tf.constant(test_image, dtype=tf.float32)
# im = tf.image.resize(im, (CONFIGURATION["IM_SIZE"], CONFIGURATION["IM_SIZE"]))  # Dont need this because our model have and inbuild resize rescale method
# im = im / 255.0
im = tf.expand_dims(im, axis=0)

CLASS_NAMES[(tf.argmax(pretrained_model.predict(im), axis=-1).numpy()[0])]

"""# Fine Tuning"""

backborn.trainable = True

input = Input(shape=(CONFIGURATION["IM_SIZE"], CONFIGURATION["IM_SIZE"], 3))

x = backborn(input, training = False)

x = GlobalAveragePooling2D()(x)
x = Dense(CONFIGURATION["N_DENSE_1"], activation='relu')(x)
x = BatchNormalization()(x)
x = Dense(CONFIGURATION["N_DENSE_2"], activation='relu')(x)

output = Dense(CONFIGURATION["NUM_CLASSES"], activation='softmax')(x)

fine_tuned_model = Model(input, output)

fine_tuned_model.summary()

fine_tuned_model.compile(
    optimizer = Adam(learning_rate=CONFIGURATION["LEARNING_RATE"]),
    loss = loss_function,
    metrics = metrics
)

history = fine_tuned_model.fit(
    training_dataset,
    validation_data = validation_dataset,
    epochs = CONFIGURATION["N_EPOCHS"],
    verbose = 1,
    # callbacks = [checkpoint_callback]
)

plot_training_curves(history)

plot_predictions_grid(
    model=fine_tuned_model,
    dataset=validation_dataset,
    class_names=CONFIGURATION["CLASS_NAMES"],
    num_images=16,
    normalize=True
)

plot_confusion_matrix(
    model=fine_tuned_model,
    dataset=validation_dataset,
    class_names=CONFIGURATION["CLASS_NAMES"],
    normalize=True
)

classification_dashboard(
    model=fine_tuned_model,
    dataset=validation_dataset,
    class_names=CONFIGURATION["CLASS_NAMES"]
)

"""# Feature Map Visualization"""

vgg_backbone = tf.keras.applications.vgg16.VGG16(
    include_top=False,
    weights= None,
    input_shape=(CONFIGURATION["IM_SIZE"], CONFIGURATION["IM_SIZE"], 3),

)

vgg_backbone.summary()

def is_conv(layer_name):
  if 'conv' in layer_name:
    return True
  else:
    return False

feature_maps = [layer.output for layer in vgg_backbone.layers[1:] if is_conv(layer.name)]
feature_map_model = Model(
    inputs = vgg_backbone.input,
    outputs = feature_maps
)

feature_map_model.summary()

test_image = cv2.imread("/content/dataset/Emotions Dataset/Emotions Dataset/test/happy/111073.jpg")
test_image = cv2.resize(test_image, (CONFIGURATION["IM_SIZE"], CONFIGURATION["IM_SIZE"]))

im = tf.constant(test_image, dtype = tf.float32)
im = tf.expand_dims(im, axis = 0)

f_maps = feature_map_model.predict(im)

print(len(f_maps))

for i in range(len(f_maps)):
  print(f_maps[i].shape)

for i in range(len(f_maps)):
  plt.figure(figsize = (256,256))
  f_size = f_maps[i].shape[1]
  n_channels = f_maps[i].shape[3]
  joint_maps = np.ones((f_size, f_size*n_channels ))

  axs = plt.subplot(len(f_maps), 1, i+1)

  for j in range(n_channels):
    joint_maps[:, f_size*j:f_size*(j+1)] = f_maps[i][..., j]

  plt.imshow(joint_maps[:,0:512])
  plt.axis("off")

"""# Ensembling"""

inputs = Input(shape = (CONFIGURATION["IM_SIZE"], CONFIGURATION["IM_SIZE"], 3))

y_1 = resnet_34(inputs)
y_2 = pretrained_model(inputs)

output = 0.5*y_1 + 0.5*y_1

ensemble_model = Model(inputs = inputs, outputs = output)

ensemble_model.compile(
    optimizer = Adam(learning_rate=CONFIGURATION["LEARNING_RATE"]),
    loss = loss_function,
    metrics = metrics
)

ensemble_model.evaluate(validation_dataset)

"""# Imbalance dataset"""

n_sample_0 = 1525 # angry
n_sample_1 = 3019 # happy
n_sample_2 = 2255 # sad

class_weights = {0:6799/n_sample_0, 1: 6799/n_sample_1, 2: 6799/n_sample_2}

backborn_2 = tf.keras.applications.efficientnet.EfficientNetB4(
    include_top=False,
    weights='imagenet',
    input_shape=(CONFIGURATION["IM_SIZE"], CONFIGURATION["IM_SIZE"], 3),
)

backborn_2.trainable = False

pretrained_model_imbalance = tf.keras.Sequential([
    Input(shape=(CONFIGURATION["IM_SIZE"], CONFIGURATION["IM_SIZE"], 3)),
    backborn_2,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(CONFIGURATION["N_DENSE_1"], activation='relu'),
    BatchNormalization(),
    tf.keras.layers.Dense(CONFIGURATION["N_DENSE_2"], activation='relu'),
    tf.keras.layers.Dense(CONFIGURATION["NUM_CLASSES"], activation='softmax')
])

pretrained_model.summary()

pretrained_model_imbalance.compile(
    optimizer = Adam(learning_rate=CONFIGURATION["LEARNING_RATE"]),
    loss = loss_function,
    metrics = metrics
)

history = pretrained_model_imbalance.fit(
    training_dataset,
    validation_data = validation_dataset,
    epochs = CONFIGURATION["N_EPOCHS"]*3,
    verbose = 1,
    class_weight = class_weights,
)

plot_training_curves(history)

plot_predictions_grid(
    model=pretrained_model_imbalance,
    dataset=validation_dataset,
    class_names=CONFIGURATION["CLASS_NAMES"],
    num_images=16,
    normalize=True
)

plot_confusion_matrix(
    model=pretrained_model_imbalance,
    dataset=validation_dataset,
    class_names=CONFIGURATION["CLASS_NAMES"],
    normalize=True
)

classification_dashboard(
    model=pretrained_model_imbalance,
    dataset=validation_dataset,
    class_names=CONFIGURATION["CLASS_NAMES"]
)

"""# Vision Transformers (ViT)"""

test_image = cv2.imread("/content/dataset/Emotions Dataset/Emotions Dataset/train/happy/387249.jpg")
test_image = cv2.resize(test_image, (CONFIGURATION["IM_SIZE"] ,CONFIGURATION["IM_SIZE"]))

patches = tf.image.extract_patches(images=tf.expand_dims(test_image, axis = 0),
                           sizes=[1, CONFIGURATION["PATCH_SIZE"], CONFIGURATION["PATCH_SIZE"], 1],
                           strides=[1, CONFIGURATION["PATCH_SIZE"], CONFIGURATION["PATCH_SIZE"], 1],
                           rates=[1, 1, 1, 1],
                           padding='VALID')

print(patches.shape)
patches = tf.reshape(patches, (patches.shape[0], -1, 768))
print(patches.shape)

plt.figure(figsize = (8,8))

for i in range(patches.shape[1]):

    ax = plt.subplot(16,16, i+1)
    plt.imshow(tf.reshape(patches[0,i,:], (16,16,3)))
    plt.axis("off")

class PatchEncoder(Layer):
  def __init__(self, N_PATCHES, HIDDEN_SIZE):
    super(PatchEncoder, self).__init__(name = 'patch_encoder')

    self.linear_projection = Dense(HIDDEN_SIZE)
    self.positional_embedding = Embedding(N_PATCHES, HIDDEN_SIZE )
    self.N_PATCHES = N_PATCHES

  def call(self, x):
    patches = tf.image.extract_patches(
        images=x,
        sizes=[1, CONFIGURATION["PATCH_SIZE"], CONFIGURATION["PATCH_SIZE"], 1],
        strides=[1, CONFIGURATION["PATCH_SIZE"], CONFIGURATION["PATCH_SIZE"], 1],
        rates=[1, 1, 1, 1],
        padding='VALID')

    patches = tf.reshape(patches, (tf.shape(patches)[0], 256, patches.shape[-1]))

    embedding_input = tf.range(start = 0, limit = self.N_PATCHES, delta = 1 )
    output = self.linear_projection(patches) + self.positional_embedding(embedding_input)

    return output

class TransformerEncoder(Layer):
  def __init__(self, N_HEADS, HIDDEN_SIZE):
    super(TransformerEncoder, self).__init__(name = 'transformer_encoder')

    self.layer_norm_1 = LayerNormalization()
    self.layer_norm_2 = LayerNormalization()

    self.multi_head_att = MultiHeadAttention(N_HEADS, HIDDEN_SIZE )

    self.dense_1 = Dense(HIDDEN_SIZE, activation = tf.nn.gelu)
    self.dense_2 = Dense(HIDDEN_SIZE, activation = tf.nn.gelu)

  def call(self, input):
    x_1 = self.layer_norm_1(input)
    x_1 = self.multi_head_att(x_1, x_1)

    x_1 = Add()([x_1, input])

    x_2 = self.layer_norm_2(x_1)
    x_2 = self.dense_1(x_2)
    output = self.dense_2(x_2)
    output = Add()([output, x_1])

    return output

class ViT(Model):
  def __init__(self, N_HEADS, HIDDEN_SIZE, N_PATCHES, N_LAYERS, N_DENSE_UNITS):
    super(ViT, self).__init__(name = 'vision_transformer')
    self.N_LAYERS = N_LAYERS
    self.patch_encoder = PatchEncoder(N_PATCHES, HIDDEN_SIZE)
    self.trans_encoders = [TransformerEncoder(N_HEADS, HIDDEN_SIZE) for _ in range(N_LAYERS)]
    self.dense_1 = Dense(N_DENSE_UNITS, tf.nn.gelu)
    self.dense_2 = Dense(N_DENSE_UNITS, tf.nn.gelu)
    self.dense_3 = Dense(CONFIGURATION["NUM_CLASSES"], activation = 'softmax')
  def call(self, input, training = True):

    x = self.patch_encoder(input)

    for i in range(self.N_LAYERS):
      x = self.trans_encoders[i](x)
    x = Flatten()(x)
    x = self.dense_1(x)
    x = self.dense_2(x)

    return self.dense_3(x)

vit = ViT(
    N_HEADS = 4, HIDDEN_SIZE = 768, N_PATCHES = 256,
    N_LAYERS = 2, N_DENSE_UNITS = 128)
vit(tf.zeros([2,256,256,3]))

vit.summary()

vit.compile(
    optimizer = Adam(learning_rate=CONFIGURATION["LEARNING_RATE"]),
    loss = loss_function,
    metrics = metrics
)

history = vit.fit(
    training_dataset,
    validation_data = validation_dataset,
    epochs = CONFIGURATION["N_EPOCHS"],
    verbose = 1,
    # class_weight = class_weights,
)

plot_training_curves(history)

plot_predictions_grid(
    model=vit,
    dataset=validation_dataset,
    class_names=CONFIGURATION["CLASS_NAMES"],
    num_images=16,
    normalize=True
)

plot_confusion_matrix(
    model=vit,
    dataset=validation_dataset,
    class_names=CONFIGURATION["CLASS_NAMES"],
    normalize=True
)

classification_dashboard(
    model=vit,
    dataset=validation_dataset,
    class_names=CONFIGURATION["CLASS_NAMES"]
)

"""# Huggingface"""

!pip install transformers

resize_rescale_hf = tf.keras.Sequential([
       Resizing(224, 224),
       Rescaling(1./255),
       Permute((3,1,2))
])

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Resizing, Rescaling, Permute, Lambda
from tensorflow.keras.models import Model
from transformers import TFViTModel

# Preprocessing pipeline
resize_rescale_hf = tf.keras.Sequential([
    Resizing(224, 224),
    Rescaling(1./255),
    Permute((3, 1, 2))  # (H, W, C) -> (C, H, W) for ViT
])

# Load Hugging Face ViT model
base_model = TFViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

# Wrap model call in Lambda to defer execution to runtime
def extract_cls_token(x):
    outputs = base_model(x)
    return outputs.last_hidden_state[:, 0, :]  # CLS token

# Build the model
inputs = Input(shape=(256, 256, 3))
x = resize_rescale_hf(inputs)
x = Lambda(extract_cls_token)(x)
output = Dense(CONFIGURATION["NUM_CLASSES"], activation='softmax')(x)

hf_model = Model(inputs=inputs, outputs=output)

test_image = cv2.imread("/content/dataset/Emotions Dataset/Emotions Dataset/train/happy/387249.jpg")
test_image = cv2.resize(test_image, (CONFIGURATION["IM_SIZE"] ,CONFIGURATION["IM_SIZE"]))

hf_model(tf.expand_dims(test_image, axis = 0))

hf_model.summary()

hf_model.compile(
    optimizer = Adam(learning_rate = 5e-5),
    loss = loss_function,
    metrics = metrics
)

from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

history = hf_model.fit(
    training_dataset,
    validation_data=validation_dataset,
    epochs=CONFIGURATION["N_EPOCHS"],
    verbose=1,
    callbacks=[
        WandbMetricsLogger(),  # Logs metrics to W&B
        WandbModelCheckpoint(filepath='model_checkpoint.keras')  # Optional model saving
    ]
)

# history = hf_model.fit(
#     training_dataset,
#     validation_data = validation_dataset,
#     epochs = CONFIGURATION["N_EPOCHS"],
#     verbose = 1,
#     # class_weight = class_weights,
#     callbacks = [WandbCallback()]
# )

plot_training_curves(history)

plot_predictions_grid(
    model=hf_model,
    dataset=validation_dataset,
    class_names=CONFIGURATION["CLASS_NAMES"],
    num_images=16,
    normalize=True
)

plot_confusion_matrix(
    model=hf_model,
    dataset=validation_dataset,
    class_names=CONFIGURATION["CLASS_NAMES"],
    normalize=True
)

classification_dashboard(
    model=hf_model,
    dataset=validation_dataset,
    class_names=CONFIGURATION["CLASS_NAMES"]
)

