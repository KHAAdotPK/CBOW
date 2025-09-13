/*
    ML/NLP/unsupervised/Word2Vec/Skip-gram/hyper-parameters.hh
    Q@khaa.pk
 */

#ifndef SKIP_GRAM_HYPER_PARAMETERS_HEADER_HH
#define SKIP_GRAM_HYPER_PARAMETERS_HEADER_HH

/*
   Regularization: Neural networks are notorious for their overfitting issues and they tend to memorize the data without learning the underlying patterns.
   There are different regularization techniques including L1 and L2 regularization, dropout, and early stopping.
   Fundamentally, they involve applying some mathematical function to prevent the model from over-learning from the data.
   L1 and L2 regularization: add absolute values or squares of weights to the loss function
   Dropout: Randomly set some fraction of outputs in the layer to zero during training (prevents single neuron overlearning)
 */
#define SKIP_GRAM_REGULARIZATION_STRENGTH 0.1
/*
    EPOCH?    
    In the context of training a machine learning model, an epoch is defined as a complete pass over the entire training dataset during training.
    One epoch is completed when the model has made one update to the weights based on each training sample in the dataset.
    In other words, during one epoch, the model has seen every example in the dataset once and has made one update to the model parameters for each example.

    Use ifdef, undef define preprocessor directives
 */
#define SKIP_GRAM_DEFAULT_EPOCH 100
/*
   The learning rate controls the step size at each iteration of the optimization process. 
   There's no single "best" learning rate. It depends on your specific problem and model architecture.

   A learning rate of 0.1 can be a good starting point for some problems, but it might be too high for your Skip-gram model
   if you're encountering NaN loss in every iteration.
   0.01 to 0.001: This is a common starting point for many deep learning tasks.
   Even lower (e.g., 0.0001 or less): Depending on your specific dataset and network architecture, you might need an even smaller learning rate.

   CBOW:-   
   For Word2Vec, where typical values range from 0.01 to 0.05 (often starting at 0.025 for CBOW in the original implementation).
   This high rate could cause the model to make large updates,
   leading to the observed loss spikes and potentially preventing smooth convergence.
 */
#define SKIP_GRAM_DEFAULT_LEARNING_RATE 0.00001
/*
    Number of neurons in the hidden layer and this represents the size of the hidden layer in the neural network.
    10 neurons is small size, suitable for small vocabulary.
    However, for larger vocabularies and more complex tasks, a larger hidden layer size may be required to capture more intricate relationships 
    between the input and output. 
    
    A 100-dimensional vector is relatively high for such a small dataset (e.g. 14 lines of text training data). 
    Word2Vec models often use 50–300 dimensions, 
    but with limited data, a smaller dimension (e.g., 50 or even 20) might help the model focus on more meaningful features and reduce overfitting risk.

    Use ifdef, undef define preprocessor directives
 */
#define SKIP_GRAM_EMBEDDNG_VECTOR_SIZE 20
/*
   Size of window of context words around a target/center word, and use the context words to predict the target word(in CBOW/Skip-Gram model) 
   In the Skip-gram model, the model predicts the context words given a target word
 */ 
#ifndef SKIP_GRAM_WINDOW_SIZE
#define SKIP_GRAM_WINDOW_SIZE 2
#endif

/*
   The number of negative samples is a hyperparameter. 
   It's often denoted as k, where k is the number of negative samples generated for each positive sample.

   Positive Samples:
   A positive sample is the context words as input and the center word as the target.
   So, the number of positive samples is determined by the word-context pairs in your corpus.
   These come directly from your training data (i.e., actual word-context relationships). 
   You don't control their quantity directly since they are determined by the corpus and the context window size.

   Negative Samples:
   Negative sampling involves randomly selecting words from the vocabulary/corpus that are neither the center word nor the context words of one single positive sample.
   You typically generate multiple negative samples for each positive sample.
   It is common to generate more negative samples than positive ones, for example, 5 or 10 negative samples for each positive sample.   
 */
#define CBOW_NEGAIVE_SAMPLE_SIZE 10

#endif