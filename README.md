## Continuous Bag of Words (`CBOW`) Implementation in C++
Welcome to the Continuous Bag of Words (`CBOW`) implementation in C++! This project is part of the broader effort to develop and experiment with `ML` models. `Word2Vec` is a popular technique in Natural Language Processing (`NLP`) for learning word embeddings from text data, enabling various downstream tasks such as semantic analysis, text classification, and machine translation.
### Project Overview
#### What is `CBOW`?
`CBOW` is one of the two architectures proposed by **Mikolov et al**. for creating word embeddings, the other being [Skip-Gram](https://github.com/KHAAdotPK/skip-gram.git). In `CBOW`, the model predicts the target word (center word) based on the context words (surrounding words within a fixed window size). This approach captures the semantics of words by analyzing their context in large corpora, resulting in dense vector representations that encode meaningful relationships between words.
## Implementation Details
In this project, I've started implementing the `CBOW` model from scratch in **C++**. The goal is to provide an efficient, flexible, and customizable `CBOW` implementation suitable for various **NLP** tasks. Here's a brief overview of what has been accomplished so far:

### Forward Propagation:
The forward propagation function computes the hidden layer and the predicted probabilities for the context words given a center word. The implementation follows the architecture of a neural network with a single hidden layer.
#### Key Steps:
- Input Layer: Represents the center word as a one-hot encoded vector.
- Hidden Layer: Projects the one-hot encoded vector to a continuous embedding using weights `W1`.
- Output Layer: Predicts the probability distribution over all words in the vocabulary using the softmax function on the output of the hidden layer.
##### Implementation Summary:
- Forward propagation is completed with matrix operations for the hidden and output layers.
- The result is a prediction of the probability distribution over context words for a given center word.

### Backward Propagation:
The backward propagation function computes the gradients of the weights based on the loss between the predicted probabilities and the actual context words (represented as one-hot vectors). These gradients are used to update the weight matrices `W1` and `W2`.
#### Key Steps:
- Error Calculation: The error is calculated as the difference between the predicted probability distribution (`y_pred`) and the one-hot encoded target distribution (`y_true`).
- Gradient Computation:
   * `grad_W1`: The gradient of the hidden layer weights.
   * `grad_W2`: The gradient of the output layer weights.   
- Weight Update: The weights are updated using the gradients and a learning rate.
##### Implementation Summary:
- Backward propagation is now fully implemented and computes the gradients for both `W1` and `W2`.
- The gradients are calculated using the outer product and dot product operations, which follow the neural network's backpropagation algorithm.

### Full Softmax (Negative Sampling Implementation Pending)

This implementation currently uses the full softmax for calculating output probabilities. While full softmax is effective, it can be computationally expensive for large vocabularies. Negative Sampling is a common technique used to optimize training in word embeddings, significantly reducing computational complexity.

**Negative Sampling is currently under development**, and a comprehensive document detailing the approach and design is being created. You can follow the progress and access the document at [https://github.com/KHAAdotPK/CBOW/blob/main/DOCUMENTS/NegativeSampling.md](https://github.com/KHAAdotPK/CBOW/blob/main/DOCUMENTS/NegativeSampling.md). The document is mostly complete and outlines the core ideas and code strategies that will soon be implemented in this CBOW model to enhance training efficiency, especially for large-scale vocabularies.

Future updates will include the ideas from this document, making the CBOW model faster and more scalable.

### Training Loop:
The CBOW training loop defines the main process for training the word embedding model using forward and backward propagation. Each epoch iterates through shuffled word pairs and updates the weights accordingly. Below is an outline of how the training loop works.
#### Training Loop Details
- **Epoch Loop**: The training runs for a specified number of epochs. Each epoch represents one complete pass through the training dataset.
- **Shuffling Word Pairs**: Before each epoch, the word pairs are shuffled to ensure the model doesn't learn in a biased manner.
- **Forward Propagation**: For each word pair, forward propagation computes the prediction probabilities using the current weights.
- **Backward Propagation**: Based on the error (difference between predicted and actual output), the gradients are computed, which are then used to update the weights.
- **Weight Updates**: 
  - `W1` (input-to-hidden weights) and `W2` (hidden-to-output weights) are updated using the learning rate `lr`.
  - `W2` is reshaped and updated to match the dimensional requirements for the subtraction and weight update steps.
- **Loss Calculation**: The training loop calculates the loss using the **Negative Log Likelihood (NLL)** function, where lower values indicate better model performance.
- **Verbose Output**: If verbose mode is enabled, the progress of each epoch and loss values are printed to the console.

### Error Handling:
Robust error handling has been incorporated to catch and report issues such as memory allocation failures and logical errors.
    
### Next Steps
The next steps in this project include:
- Currently, the implementation does not include **Negative Sampling**. This technique is commonly used in word embedding models to improve training efficiency by sampling negative examples during the training process. Implementing Negative Sampling will be essential to enhance the performance of the model.
- Integrating optimization algorithms such as Stochastic Gradient Descent (SGD).
- Expanding the codebase to handle larger datasets efficiently.
- Performing extensive testing and validation to ensure correctness and performance.

### Dependencies
Before building and running the `CBOW` project, you need to clone and set up the following repositories, which contain essential libraries and utilities:
```bash
git clone https://github.com/KHAAdotPK/String.git
git clone https://github.com/KHAAdotPK/parser.git
git clone https://github.com/KHAAdotPK/ala_exception.git
git clone https://github.com/KHAAdotPK/allocator.git
git clone https://github.com/KHAAdotPK/sundry.git
git clone https://github.com/KHAAdotPK/argsv-cpp.git
git clone https://github.com/KHAAdotPK/corpus.git
git clone https://github.com/KHAAdotPK/Numcy.git
git clone https://github.com/KHAAdotPK/csv.git
git clone https://github.com/KHAAdotPK/pairs.git
git clone https://github.com/KHAAdotPK/read_write_weights.git
```
Ensure that these repositories are cloned into `lib` directory of your project directory before building the CBOW. implementation.

Alternatively, in the lib directory of this project, you will find a PULL.cmd file. Executing this file will automatically clone all the above repositories for you:
```bash
cd lib
./PULL.cmd
```

### Usage

1. Clone the repository:
    ```bash
    git clone https://github.com/KHAAdotPK/CBOW.git
    ```
2.  In the lib directory of this project, you will find a PULL.cmd file. Executing this file will automatically clone all the above repositories for you:
    ```bash
    cd CBOW
    cd lib
    ./PULL.cmd
    ```   
3. Build the example training program (ensure you have a C++11/14 compiler):
    ```bash
    cd CBOW
    cd usage
    ./BUILD.cmd    
    ```
4. Run the training program with your dataset:
    ```bash
    cd CBOW
    cd usage
    ./RUN.cmd
    ```

#### Example of skeleton training program.
##### Parameters
- **Epoch**: The number of times the model will iterate over the entire training data. Each epoch involves shuffling the word pairs and updating the weights based on the error between the predicted and actual context words.  
- **Learning Rate**: A scalar value that controls the size of the weight updates during training. A higher learning rate may result in faster convergence but risks overshooting the optimal values, while a lower learning rate leads to slower convergence but more precise updates.
- **Verbose**: A boolean parameter. When set to `true`, detailed progress (such as the current epoch and loss) is printed during training. This is useful for debugging or monitoring training progress.
- **Epoch Loss**: This variable accumulates the total loss over all word pairs in an epoch. In the context of the CBOW model, the loss is typically computed using **negative log-likelihood (NLL)**. Lower values of the loss indicate that the model is improving its predictions. After each epoch, the average loss (`epoch_loss / number_of_word_pairs`) is printed, allowing you to monitor how well the model is learning over time. The lower the loss, the better the model is at predicting the correct context words from the center word.

```cpp
#include "../lib/WordEmbedding-Algorithms/Word2Vec/CBOW/header.hh"

int main() {
    cc_tokenizer::String<char> data = cc_tokenizer::cooked_read<char>(CBOW_DEFAULT_CORPUS_FILE);
    cc_tokenizer::csv_parser<cc_tokenizer::String<char>, char> data_parser(data);
    // Initialize vocabulary
    class Corpus vocab(data_parser);
    // Initialize word pairs    
    PAIRS pairs(vocab);

    // Initialize weights W1, W2
    Collective<double> W1 = Numcy::Random::randn(DIMENSIONS{SKIP_GRAM_EMBEDDNG_VECTOR_SIZE, vocab.numberOfUniqueTokens(), NULL, NULL});
    Collective<double> W2 = Numcy::Random::randn(DIMENSIONS{vocab.numberOfUniqueTokens(), SKIP_GRAM_EMBEDDNG_VECTOR_SIZE, NULL, NULL});

    // Define training parameters    
    size_t default_epochs = 10;
    double default_lr = SKIP_GRAM_DEFAULT_LEARNING_RATE;    
    double epoch_loss = 0.0;

    bool verbose = true;

    // Start training    
    CBOW_TRAINING_LOOP(epoch_loss, default_epochs, default_lr, pairs, double, verbose, vocab, W1, W2);

    return 0;
}
```

### Training Output Summary

The CBOW model was trained on INPUT.txt with the following parameters:

- Learning Rate (lr): 0.0001
- Epochs: 30
- ~Regularization Strength (rs): 0.000001~
- Verbose mode enabled

#### Training Loss Progression

The model's epoch loss decreased consistently over the 30 epochs, demonstrating that the model successfully minimized the objective function over time. Below is a summary of key observations:

1. Initial Epochs: The training started with a loss of 12.7097 in the first epoch.

2. Gradual Convergence: The model exhibited a steady decrease in epoch loss, with minor reductions in each epoch. By the 10th epoch, the loss had reduced to 12.5048.

3. Final Epochs: After 30 epochs, the final epoch loss reached 12.0626, indicating gradual convergence toward a lower loss value.

#### Key Insights:

The consistent reduction in loss across epochs indicates that the model parameters were updated effectively under the specified learning rate. The small learning rate ~and regularization strength~ likely contributed to the model's stable and gradual convergence, helping prevent large oscillations in loss reduction.

```BASH
F:\CBOW\usage>.\cow.exe corpus ./INPUT.txt lr 0.0001 epoch 30 rs 0.000001 verbose
Corpus: ./INPUT.txt
Epoch# 1 of 30 epochs.
epoch_loss = 12.7097
Epoch# 2 of 30 epochs.
epoch_loss = 12.6868
Epoch# 3 of 30 epochs.
epoch_loss = 12.6639
Epoch# 4 of 30 epochs.
epoch_loss = 12.641
Epoch# 5 of 30 epochs.
epoch_loss = 12.6182
Epoch# 6 of 30 epochs.
epoch_loss = 12.5955
Epoch# 7 of 30 epochs.
epoch_loss = 12.5727
Epoch# 8 of 30 epochs.
epoch_loss = 12.5501
Epoch# 9 of 30 epochs.
epoch_loss = 12.5274
Epoch# 10 of 30 epochs.
epoch_loss = 12.5048
Epoch# 11 of 30 epochs.
epoch_loss = 12.4823
Epoch# 12 of 30 epochs.
epoch_loss = 12.4598
Epoch# 13 of 30 epochs.
epoch_loss = 12.4374
Epoch# 14 of 30 epochs.
epoch_loss = 12.415
Epoch# 15 of 30 epochs.
epoch_loss = 12.3926
Epoch# 16 of 30 epochs.
epoch_loss = 12.3703
Epoch# 17 of 30 epochs.
epoch_loss = 12.348
Epoch# 18 of 30 epochs.
epoch_loss = 12.3258
Epoch# 19 of 30 epochs.
epoch_loss = 12.3036
Epoch# 20 of 30 epochs.
epoch_loss = 12.2815
Epoch# 21 of 30 epochs.
epoch_loss = 12.2594
Epoch# 22 of 30 epochs.
epoch_loss = 12.2373
Epoch# 23 of 30 epochs.
epoch_loss = 12.2153
Epoch# 24 of 30 epochs.
epoch_loss = 12.1934
Epoch# 25 of 30 epochs.
epoch_loss = 12.1715
Epoch# 26 of 30 epochs.
epoch_loss = 12.1496
Epoch# 27 of 30 epochs.
epoch_loss = 12.1278
Epoch# 28 of 30 epochs.
epoch_loss = 12.106
Epoch# 29 of 30 epochs.
epoch_loss = 12.0843
Epoch# 30 of 30 epochs.
epoch_loss = 12.0626
Training done!
```

### Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue if you have any suggestions or bug reports.

### License
This project is governed by a license, the details of which can be located in the accompanying file named 'LICENSE.' Please refer to this file for comprehensive information.

### Acknowledgements
[Tomas Mikolov and others](https://arxiv.org/abs/1301.3781)

---
For more detailed information on the implementation, please refer to the source code in this [repository](https://github.com/KHAAdotPK/CBOW/tree/main/lib/WordEmbedding-Algorithms/Word2Vec/CBOW). Additionally, the DOCUMENTS folder contains files which explain the CBOW model in the light of this implementation.