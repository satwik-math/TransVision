# -*- coding: utf-8 -*-

import numpy as np
import tensorflow
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Prepare dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images, test_images = train_images/255.0, test_images/255.0
num_classes = len(np.unique(train_labels))
train_images = train_images.reshape(-1, 28, 28, 1)
test_images = test_images.reshape(-1, 28, 28, 1)
train_labels = to_categorical(train_labels, num_classes)
test_labels = to_categorical(test_labels, num_classes)

# Define vision transformer
def vision_transformer(input_shape = (28, 28, 1)):
    inputs = layers.Input(shape = input_shape)
    token_embedding = layers.Conv2D(filters = 64, kernel_size = 4, strides = (4, 4))(inputs)
    token_embedding = layers.Reshape((-1, 64))(token_embedding)

    position_embedding = layers.Embedding(input_dim = 49, output_dim = 64, input_length = 49)(tensorflow.range(49))
    embedding = token_embedding + position_embedding

    for i in range(4):
        attention_output = layers.MultiHeadAttention(8, 64)(embedding, embedding)
        embedding = layers.Add()([embedding, attention_output])
        embedding = layers.LayerNormalization(epsilon=0.001)(embedding)
        ffn = layers.Dense(256, activation = 'relu')(embedding)
        ffn = layers.Dense(64)(ffn)
        embedding = layers.Add()([ffn, embedding])
        embedding = layers.LayerNormalization(epsilon = 0.001)(embedding)

    embedding = layers.GlobalAveragePooling1D()(embedding)
    output = layers.Dense(num_classes, activation = 'softmax')(embedding)

    return Model(inputs, output)

# Set model
model = vision_transformer()
vision_transformer().summary()
model.compile(optimizer = 'Adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model_dir = 'folder_path'
model_path1 = os.path.join(model_dir, 'training_logs.csv')
model_path2 = os.path.join(model_dir, 'model_weights.h5')
model_path3 = os.path.join(model_dir, 'model_architecture.png')
os.makedirs(model_dir, exist_ok=True)
checkpoint_callback = ModelCheckpoint(model_path2, monitor='val_accuracy', save_best_only=True, save_weights_only=True, verbose=1)
csv_callback = CSVLogger(model_path1)
tensorflow.keras.utils.plot_model(vision_transformer(), to_file = model_path3)
model.fit(train_images, train_labels, epochs = 30, batch_size = 32, validation_data = [test_images, test_labels], callbacks = [checkpoint_callback, csv_callback])
model.save(os.path.join(model_dir, 'model.h5'))

# Check accuracy
loaded_model = tensorflow.keras.models.load_model(os.path.join(model_dir, 'model.h5'))
accuracy = loaded_model.evaluate(test_images, test_labels)
print(accuracy)

# Display first few test images and their predicted labels
predictions = loaded_model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)
plt.figure(figsize=(10, 10))
for i in range(49):
    plt.subplot(7, 7, i + 1)
    plt.imshow(test_images[i].reshape(28, 28), cmap='gray')
    plt.title(f"Pr: {predicted_labels[i]}, Re: {np.argmax(test_labels[i])}")
    plt.axis('off')
plt.show()
