# Vision Transformer for MNIST Classification

This repository contains a TensorFlow implementation of a Vision Transformer (ViT) for MNIST digit classification. The Vision Transformer is a model architecture that has shown promising results in image classification tasks. In this implementation, we train the Vision Transformer on the MNIST dataset, which consists of handwritten digits.

## Dependencies
- Python 3.x
- TensorFlow 2.x
- NumPy
- Matplotlib

## Usage
1. Clone the repository:

    ```
    git clone <https://github.com/satwik-math/TransVision>
    ```

2. Install the dependencies:

    ```
    pip install -r requirements.txt
    ```

3. Run the script:

    ```
    python transvision.py
    ```

## Description
- `transvision.py`: This script contains the implementation of the Vision Transformer model for MNIST digit classification. It loads the MNIST dataset, preprocesses the data, defines the model architecture, compiles the model, trains it on the training data, and evaluates its performance on the test data. Additionally, it saves the model weights, training logs, and model architecture diagram.

## Model Architecture
The Vision Transformer model consists of multiple layers of self-attention mechanisms and feed-forward neural networks. It utilizes token embeddings and position embeddings to capture spatial information in the input images. The model architecture is as follows:

- Input Layer: Accepts input images of size (28, 28, 1).
- Token Embedding Layer: Applies a 2D convolutional layer to generate token embeddings.
- Position Embedding Layer: Generates position embeddings to encode spatial information.
- Transformer Encoder Layers: Consist of self-attention mechanisms and feed-forward neural networks.
- Global Average Pooling Layer: Aggregates features across the spatial dimensions.
- Output Layer: Produces softmax predictions for classifying the input images into 10 classes (digits 0-9).

## Results
After training the Vision Transformer model on the MNIST dataset, we achieved an accuracy of approximately 98% on the test data. The training logs and model weights are saved in the specified directory for further analysis and deployment.

## Acknowledgments
- This implementation is inspired by the original Vision Transformer paper: [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929).
- The MNIST dataset is sourced from the TensorFlow/Keras library.

For any questions or issues, please feel free to contact [Satwik Mukherjee](satwik.applied@gmail.com).
