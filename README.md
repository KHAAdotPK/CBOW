# CBOW Model Implementation in C++

This repository contains an implementation of the **Continuous Bag of Words (CBOW)** model in C++. The CBOW model is part of the Word2Vec architecture and is used to learn word embeddings by predicting the center word in a context of surrounding words. This project is designed for training a simple neural network on text data, generating vector representations for words.

## Features
- **CBOW Model**: Trains word embeddings by using context words to predict a center word.
- **Efficient Training Loop**: Includes forward and backward propagation with custom weight update functions.
- **Custom Structures**: Leverages `Collective` and `Numcy` for efficient reshaping and random shuffling of data.
- **Shuffling Training Data**: Randomly shuffles word pairs in each epoch to prevent bias during training.
- **Adjustable Learning Rate and Epochs**: Easily configurable parameters for learning rate and number of epochs.

## How the CBOW Model Works
1. **Context**: The model takes a set of context words as input.
2. **Prediction**: It tries to predict the center word in the context window.
3. **Training**: The weights between input and hidden layers (W1), and hidden to output layers (W2), are updated using backpropagation.
4. **Loss Calculation**: Uses negative log-likelihood (NLL) for measuring the loss during training.

## Training Loop

The core training loop can be found in the `#define CBOW_TRAINING_LOOP` macro. This loop:
- Shuffles the word pairs before each epoch.
- Implements forward and backward propagation.
- Updates the weights `W1` and `W2` based on gradients.
- Computes and displays the loss for each epoch.

### Training Process
The training loop works in the following steps:
1. **Shuffle Word Pairs**: Before each epoch, the word pairs are shuffled.
2. **Forward Propagation**: Predict the center word using the current weights.
3. **Backward Propagation**: Calculate the gradients and adjust the weights based on the learning rate.
4. **Loss Calculation**: Compute and print the average loss per epoch.

## Usage

1. Clone the repository:
    ```bash
    git clone https://github.com/KHAAdotPK/CBOW.git
    ```
2. Build the training program (ensure you have a C++11/14 compiler):
    ```bash
    cd CBOW
    cd usage
    ./BUILD.cmd    
    ```
3. Run the training program with your dataset:
    ```bash
    cd CBOW
    cd usage
    ./RUN.cmd
    ```

## Parameters
- **Epoch**: Number of times the model will iterate over the entire training data.
- **Learning Rate**: Controls how much the weights are adjusted during training.
- **Verbose**: Enable to print detailed progress during training.

```cpp
// Example usage
#include "../lib/WordEmbedding-Algorithms/Word2Vec/CBOW/header.hh"

int main() {
    // Initialize vocabulary, word pairs, and model weights (W1)
    CORPUS_REF vocab = ...;
    PAIRS pairs = ...;
    Collective<double> W1 = ...;

    // Define training parameters
    size_t epochs = 10;
    bool verbose = true;

    // Start training
    CBOW_TRAINING_LOOP(epochs, pairs, verbose, vocab, W1);

    return 0;
}
```

