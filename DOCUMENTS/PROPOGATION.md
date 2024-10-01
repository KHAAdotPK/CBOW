## __Forward__ ~~and __Backward__~~ propogation in `CBOW` and `Skip-gram` 
**Word2Vec** has two **word embedding algorithms**, the `Skip-gram` and `CBOW`. `Skip-gram` predicts context words for a center/target word, while `CBOW` predicts the center/target word for given context words.
#### Steps involved in Forward propogation are more or less similar in `CBOW` and `Skip-gram`
1. **Context Extraction**:
- Both `Skip-gram` and `CBOW` require identifying the context words. For `CBOW`, this involves averaging the embeddings of all context words, whereas, in `Skip-gram`, this would typically involve extracting the embedding for a single target word.
- __Example of context extraction in__ `Skip-gram`
```C++
/*
    Extract the corresponding word embedding from the weight matrix 𝑊1.
    The embedding for the center word is stored in the hidden layer vector h.
*/
double* h_ptr = cc_tokenizer::allocator<double>().allocate(W1.getShape().getNumberOfColumns());
// Loop through the columns of W1 to extract the embedding for the center word.
for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < W1.getShape().getNumberOfColumns(); i++)
{
    *(h_ptr + i) = W1[W1.getShape().getNumberOfColumns()*(pair->getCenterWord() - INDEX_ORIGINATES_AT_VALUE) + i];
}
Collective<double> h = Collective<double>{h_ptr, DIMENSIONS{W1.getShape().getNumberOfColumns(), 1, NULL, NULL}};
```
- __Example of context extraction in__ `CBOW`
```C++
/*
    For CBOW, h is the average of the embeddings of the context words. 
    This involves accessing the embeddings for all context words and averaging them.
 */
cc_tokenizer::string_character_traits<char>::size_type j = 0;
cc_tokenizer::string_character_traits<char>::size_type* ptr = cc_tokenizer::allocator<cc_tokenizer::string_character_traits<char>::size_type>().allocate(CBOW_WINDOW_SIZE*2);
for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < SKIP_GRAM_WINDOW_SIZE; i++)
{
    if ((*(pair->getLeft()))[i] >= INDEX_ORIGINATES_AT_VALUE)
    {
        ptr[j] = (*(pair->getLeft()))[i] - INDEX_ORIGINATES_AT_VALUE;
        j = j + 1;
    }
}
for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < SKIP_GRAM_WINDOW_SIZE; i++)
{
    if ((*(pair->getRight()))[i] >= INDEX_ORIGINATES_AT_VALUE)
    {
        ptr[j] = (*(pair->getRight()))[i] - INDEX_ORIGINATES_AT_VALUE;
        j = j + 1;
    }
}
Collective<cc_tokenizer::string_character_traits<char>::size_type> context = Collective<cc_tokenizer::string_character_traits<char>::size_type>{ptr, DIMENSIONS{CBOW_WINDOW_SIZE*2, 1, NULL, NULL}};
/*
    Extract the corresponding word embeddings from the weight matrix 𝑊1, 
    average them and then store it in hidden layer.
 */
Collective<double> h = Numcy::mean(W1, context);
```
2. **Dot Product with Output Weights**:
- Both algorithms then perform a dot product between the hidden layer representation (h) and the output weight matrix (W2).
```C++
/*
    The dot product gives us the logits or unnormalized probabilities (u), which can then be transformed into probabilities using a softmax function
 */
Collective<E> u = Numcy::dot(h, W2);
```
This transformation step(`the dot product`) is crucial in both algorithms to map the hidden representation to the vocabulary space.

3. **Positive predicted probablities**:
```C++
        /*
            The resulting vector (u) is passed through a softmax function to obtain the predicted probabilities (y_pred). 
            The softmax function converts the raw scores into probabilities.
         */
        Collective<E> y_pred = softmax<E>(u);
```
    - In `Skip-gram`, this output represents the likelihood of each word being one of the context words for the given center word.
    - In `CBOW`, this output represents the likelihood of each word being the target word, given the context words.

