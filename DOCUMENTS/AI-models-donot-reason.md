# AI Models: Pattern Matching, Not Reasoning
Artificial Intelligence models like **CBOW (Continuous Bag of Words)** or even **Large Language Models (LLMs)** aren't capable of **reasoning in the way humans do**. Their **"**intelligence**"** is grounded in `mathematical operations`, like `vector space transformations`, that rely heavily on pattern recognition rather than logical thought or reasoning.

For instance, consider a simple **CBOW** implementation with two weight matrices, **W1** and **W2**:

```cpp
/*
    W1, This weight matrix maps the input (context words) to the embedding space.
    That is, each word in the vocabulary is represented as a vector/array of random numbers.
 */
Collective<double> W1 = Numcy::Random::randn(DIMENSIONS{SKIP_GRAM_EMBEDDNG_VECTOR_SIZE, vocab.numberOfUniqueTokens(), NULL, NULL});

/*
    W2, This weight matrix maps the hidden layer (after summing the word embeddings) back to the output layer, which corresponds to the same vocabulary as the input matrix W1.
 */
Collective<double> W2 = Numcy::Random::randn(DIMENSIONS{vocab.numberOfUniqueTokens(), SKIP_GRAM_EMBEDDNG_VECTOR_SIZE, NULL, NULL});
```
In this **CBOW** model, the process is entirely **mechanical**:
1. **W1*** converts input context words into a vector representation, which the model uses to understand the relationships between words.
2. **W2** takes that hidden layer and maps it back to the output layer, aiming to predict a target word based on the context.

However, it's crucial to recognize that **AI** models do not understand or reason about these word relationships. They rely purely on [statistical correlations](https://github.com/KHAAdotPK/skip-gram/blob/main/DOCUMENTS/GradientDescent.md) learned from their training data.

### AI Models and Cosine Similarity: The Core Mechanism
When **AI** models, including advanced **LLMs**, generate output, they’re not reasoning through a problem. Instead, they’re **comparing vector representations** of words and sentences, typically using metrics like **cosine similarity**. Cosine similarity measures the **closeness** between vectors—words or sentences with similar meanings end up close together in vector space.

Consider this code block that calculates cosine similarity between two vectors `u` and `v`, extracted from the weight matrix `W1`:
```C++
/*
    This line directly prints the cosine similarity between two vectors.
    The cosine similarity value ranges between -1 (completely opposite vectors) and 1 (identical vectors).
    A value of 0 would indicate orthogonality (no similarity).
 */
Collective<double> u = Collective<double>{W1.slice(ptr->i*SKIP_GRAM_EMBEDDNG_VECTOR_SIZE, SKIP_GRAM_EMBEDDNG_VECTOR_SIZE), DIMENSIONS{SKIP_GRAM_EMBEDDNG_VECTOR_SIZE, 1, NULL, NULL}};
Collective<double> v = Collective<double>{W1.slice(next_ptr->i*SKIP_GRAM_EMBEDDNG_VECTOR_SIZE, SKIP_GRAM_EMBEDDNG_VECTOR_SIZE), DIMENSIONS{1, SKIP_GRAM_EMBEDDNG_VECTOR_SIZE, NULL, NULL}};
std::cout<< "Cosine Similarity = " << Numcy::Spatial::Distance::cosine(u, v) << std::endl;

/*
    This line calculates the cosine distance, which is 1 - cosine similarity.
    Cosine distance transforms the similarity measure into a metric where 0 represents identical vectors,
    and 1 represents vectors that are completely dissimilar (at 90° or 180°).
 */
Collective<double> u1 = Collective<double>{W1.slice(ptr->i*SKIP_GRAM_EMBEDDNG_VECTOR_SIZE, SKIP_GRAM_EMBEDDNG_VECTOR_SIZE), DIMENSIONS{SKIP_GRAM_EMBEDDNG_VECTOR_SIZE, 1, NULL, NULL}};
Collective<double> v1 = Collective<double>{W1.slice(next_ptr->i*SKIP_GRAM_EMBEDDNG_VECTOR_SIZE, SKIP_GRAM_EMBEDDNG_VECTOR_SIZE), DIMENSIONS{1, SKIP_GRAM_EMBEDDNG_VECTOR_SIZE, NULL, NULL}};
/*
     If the cosine similarity is negative (which can happen when two vectors point in opposite directions),
     then deducting it from 1 (i.e., 1 - cosine similarity) would result in a value greater than 1.
     This is problematic if you're expecting a distance or similarity metric that should be constrained 
     between 0 and 1.
     If you don't want negative values to affect your metric,
     you can consider taking the absolute value of cosine similarity for a meaningful distance metric.
     This way, you're ensuring that even opposite vectors are treated in a range that makes sense for your application
 */
std::cout<< "Cosine Distance = " << 1 -  std::abs(Numcy::Spatial::Distance::cosine(u1, v1)) << std::endl;
```
This code demonstrates how cosine similarity and cosine distance help AI models determine how close two word vectors are. **Cosine similarity** ranges between `-1` (completely dissimilar) and `1` (identical). Meanwhile, **cosine distance** provides a more intuitive metric by subtracting cosine similarity from `1`, offering a value between `0` and `1` where `0` indicates identical vectors and `1` represents complete dissimilarity.
### Lack of Awareness:
Despite these complex mathematical calculations, models are not engaged in reasoning. Instead, they rely on **statistical relationships** between words based on their training data. There is no **awareness** of the actual meaning behind the words, and these systems operate purely on **pattern recognition**.

AI models simply follow **rules** embedded in their architecture—using operations like **cosine similarity**—to make predictions based on past data. However, there is no **self-awareness**, **logical reasoning**, or **conscious understanding** of their outputs. In essence, they perform **sophisticated mathematical pattern matching**, not **reasoning**.


