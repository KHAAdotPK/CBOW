```
/*
	NegativeSampling.md
	Written by, Sohail Qayum Malik.
 */
```

`"Readers should be aware that this article represents an ongoing project. The information and code contained herein are preliminary and will be expanded upon in future revisions."`


#### Implementing Negative Sampling in a CBOW Model
---
**What are Positive Samples**?
- For a given center word, you have a window of context words around it.
- In the CBOW model, the context words are the input, and the center word is the output.
- These naturally occurring word-context pairs from your text are considered positive samples as they represent true word associations.

**What are Negative Samples**?
Negative samples are randomly selected words from the vocabulary/corpus that are not related to the current context words or the current center/target word. For each positive sample, multiple negative samples are generated. Negative sampling helps the model learn by contrasting the positive samples with these randomly selected negative samples. By providing random words as negative samples, you're showing the model that those words should not be predicted as the center word given the current context.

In the CBOW training loop, Negative Sampling can be implemented within the training loop, specifically in the section where the forward and backward propagation occur. Here’s how it can be integrated
1. **Before Forward Propagation**: After retrieving the current word pair(`a positive sample`), you can introduce a method to generate negative samples. This would involve selecting a set of negative words (i.e., words that are not related to the current context) based on your vocabulary. **In this method, make sure to avoid selecting the center/target word itself or any of the context words as negative samples**.

```C++
/**
 * @brief Generates negative samples for the CBOW model using negative sampling
 *
 * This function selects a set of negative samples (words) that are not related to the current target word (central word) being predicted.
 * Negative samples are randomly drawn from the vocabulary,
 * and the function ensures that the generated negative samples do not include the target word (central word) of the current context.
 * These negative samples are used to train the model to differentiate between the correct (target) word and unrelated (negative) words
 *
 * @tparam E - Type of the elements, typically used for indexing and storing sizes.
 *             Defaults to the size type of cc_tokenizer::string_character_traits<char>
 *
 * @param vocab - Reference to the corpus vocabulary from which to generate negative samples. 
 *                It must contain the method numberOfUniqueTokens(), which returns the total number 
 *                of unique words in the vocabulary
 *
 * @param pair - Pointer to the word pair, representing the center/target word, left and right context words in the CBOW model.
 *               The pair object should have methods getLeft() and getRight() that return arrays of context words.
 *             - This is positive sample
 *
 * @param n - The number of negative samples to generate. Defaults to the size of the skip-gram window (CBOW_NEGATIVE_SAMPLE_SIZE).
 *          - Typically, a small number of negative samples (5-20 per positive sample) are selected to avoid too many unnecessary computations
 *
 * @throws ala_exception - Throws this exception in case of memory allocation errors (`std::bad_alloc`) 
 *                         or length issues (`std::length_error`) during dynamic memory allocation
 *
 * @return Collective<E> - Returns an instance of templated composite type.   
 *                         The composite encapsulates an array of negative samples and the size of the array of negative samples
 *                          
 */
template <typename E = cc_tokenizer::string_character_traits<char>::size_type>
Collective<E> generateNegativeSamples_cbow(CORPUS_REF vocab, WORDPAIRS_PTR pair, E n = CBOW_NUMBER_OF_NEGATIVE_SAMPLES) throw (ala_exception)
{     
    if (n == 0)
    {
        return Collective<E>();
    }
    
    E lowerbound = 0 + INDEX_ORIGINATES_AT_VALUE;
    E higherbound = PAIRS_VOCABULARY_TRAINING_SPLIT((vocab.numberOfUniqueTokens() + INDEX_ORIGINATES_AT_VALUE - 1));
    /*    
        For documentation purposes. 
        Ensure valid distribution bounds, we know our bounds can't generate negative random numbers
        if (lowerBound > upperBound) 
        {
            throw ala_exception(cc_tokenizer::String<char>("generateNegativeSamples() Error: Invalid bounds for random distribution."));
        }
     */

    /*
        Vocabulary Size Check.
        Ensure there are enough unique tokens to generate the required number of negative samples without redundancy. 
        This is important...
        1. When you don't want the model to be trained on duplicate negative examples.
        2. And prevents the loop from running indefinitely when the vocabulary is too small.
     */    
    if (n > vocab.numberOfUniqueTokens())
    {
        throw ala_exception(cc_tokenizer::String<char>("generateNegativeSamples() Error: Vocabulary size too small."));
    }

    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<E> distrib(lowerbound, higherbound);
    
    E* ptr = NULL;

    try 
    {
        ptr = cc_tokenizer::allocator<E>().allocate(n);
    }
    catch (std::bad_alloc& e)
    {
        throw ala_exception(cc_tokenizer::String<char>("generateNegativeSamples() Error: std::bad_alloc caught. ") + cc_tokenizer::String<char>(e.what()));
    }    
    catch (std::length_error& e)
    {
        throw ala_exception(cc_tokenizer::String<char>("generateNegativeSamples() Error: std::length_error caught. ") + cc_tokenizer::String<char>(e.what()));
    }

    E i = 0;

    /*
        TODO,
        Instead of using while(1) with a break condition, you can use while(i < n) to clarify the intent and reduce potential confusion
     */
    while (1)
    {
        E index = distrib(gen);

        if (pair->getCenterWord() != index)
        {
            E j = 0;

            for (; j < SKIP_GRAM_WINDOW_SIZE;)
            {
                if ((*(pair->getRight()))[j] == index || (*(pair->getLeft()))[j] == index)
                {
                    break;
                }

                j++;
            }

            /*
                This block is handling the result of the first inner loop, which checks whether the randomly selected `index` 
                is found in the right or left context of the word pair (i.e., if it's part of the context window).

                - If `j == SKIP_GRAM_WINDOW_SIZE`, it means that the loop completed without finding the `index` in the context,
                so `index` is not part of the context and we are free to proceed further. 
                We reset `j` to 0 to ensure it's ready for the next loop, which checks for redundancy in the negative samples array.

                - If `j != SKIP_GRAM_WINDOW_SIZE`, it means the loop found that `index` is part of the context (either left or right),
                and therefore this word should not be selected as a negative sample. Setting `j = n` forces the outer loop
                to skip adding this word to the negative samples array by failing the subsequent redundancy check.
             */
            if (j == SKIP_GRAM_WINDOW_SIZE)
            {
                j = 0;
            }
            else 
            {
                j = n;
            }
            
            /* 
                Negative sampling with no redundancy.            
                The inner loop checks for redundancy by scanning the previously selected indices in the ptr array.
                This ensures that no duplicate samples are selected
             */            
            for (; j < i;)
            {
                if (ptr[j] == index)
                {
                    break;
                }

                j = j + 1;
            } 
            // True if no redundancy is found           
            if (j == i)
            {    
                ptr[i] = index;

                i = i + 1;
            }
        }

        // If true then negative sampling is completed, come out of universal loop
        if (i == n)
        {
            break;
        }
    }

    return Collective<E> {ptr, DIMENSIONS{n, 1, NULL, NULL}};    
}
```

#### Older version of above given function (for documentation purposes)

```C++
/*    
    In CBOW, we are predicting the central word, not the context.
    So, the negative samples should be words that are not the central word (target) but could be drawn from the entire vocabulary

    A better way to phrase it:
    We need to randomly select words from the vocabulary that are not the target (central) word when given a context of surrounding words.
    These negative samples help the model learn to distinguish between the true target word and unrelated words

    Is it ok to not find a single vocabulary word which is not in the center of context window?
    In a large enough corpus, every word in the vocabulary could potentially be the central word for some context window at some point
 */
/**
 * @brief Generates negative samples for the CBOW model using negative sampling
 *
 * This function selects a set of negative samples (words) that are not related to the current target word (central word) being predicted.
 * Negative samples are randomly drawn from the vocabulary,
 * and the function ensures that the generated negative samples do not include the target word (central word) of the current context.
 * These negative samples are used to train the model to differentiate between the correct (target) word and unrelated (negative) words
 *
 * @tparam E - Type of the elements, typically used for indexing and storing sizes.
 *             Defaults to the size type of cc_tokenizer::string_character_traits<char>
 *
 * @param vocab - Reference to the corpus vocabulary from which to generate negative samples. 
 *                It must contain the method numberOfUniqueTokens(), which returns the total number 
 *                of unique words in the vocabulary
 *
 * @param pair - Pointer to the word pair, representing the center/target word, left and right context words in the CBOW model.
 *               The pair object should have methods getLeft() and getRight() that return arrays of context words.
 *             - This is positive sample
 *
 * @param n - The number of negative samples to generate. Defaults to the size of the skip-gram window (CBOW_NEGATIVE_SAMPLE_SIZE).
 *          - Typically, a small number of negative samples (5-20 per positive sample) are selected to avoid too many unnecessary computations
 *
 * @throws ala_exception - Throws this exception in case of memory allocation errors (`std::bad_alloc`) 
 *                         or length issues (`std::length_error`) during dynamic memory allocation
 *
 * @return cc_tokenizer::string_character_traits<char>::size_type* - Returns a pointer to an array of negative samples.
 *                                                                   The array contains `n` negative samples, where 
 *                                                                   each sample is an index into the vocabulary.
 *                                                                   The caller is responsible for deallocating this memory 
 *                                                                   to avoid memory leaks
 */
template <typename E = cc_tokenizer::string_character_traits<char>::size_type>
E* generateNegativeSamples_cbow(CORPUS_REF vocab, WORDPAIRS_PTR pair, E n = CBOW_NEGAIVE_SAMPLE_SIZE) throw (ala_exception)
{    
    E lowerbound = 0 + INDEX_ORIGINATES_AT_VALUE;
    E higherbound = vocab.numberOfTokens() + INDEX_ORIGINATES_AT_VALUE - 1;
    /*    
        For documentation purposes. 
        Ensure valid distribution bounds, we know our bounds can't generate negative random numbers
        if (lowerBound > upperBound) 
        {
            throw ala_exception(cc_tokenizer::String<char>("generateNegativeSamples() Error: Invalid bounds for random distribution."));
        }
     */

    /*
        Vocabulary Size Check.
        Ensure there are enough unique tokens to generate the required number of negative samples without redundancy. 
        This is important...
        1. When you don't want the model to be trained on duplicate negative examples.
        2. And prevents the loop from running indefinitely when the vocabulary is too small.
     */    
    if (n > vocab.numberOfUniqueTokens())
    {
        throw ala_exception(cc_tokenizer::String<char>("generateNegativeSamples() Error: Vocabulary size too small."));
    }

    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<E> distrib(lowerbound, higherbound);
    
    E* ptr = NULL;

    try 
    {
        ptr = cc_tokenizer::allocator<E>().allocate(n);
    }
    catch (std::bad_alloc& e)
    {
        throw ala_exception(cc_tokenizer::String<char>("generateNegativeSamples() Error: std::bad_alloc caught. ") + cc_tokenizer::String<char>(e.what()));
    }    
    catch (std::length_error& e)
    {
        throw ala_exception(cc_tokenizer::String<char>("generateNegativeSamples() Error: std::length_error caught. ") + cc_tokenizer::String<char>(e.what()));
    }

    E i = 0;

    /*
        TODO,
        Instead of using while(1) with a break condition, you can use while(i < n) to clarify the intent and reduce potential confusion
     */
    while (1)
    {
        E index = distrib(gen);

        if (pair->getCenterWord() != index)
        {
            E j = 0;

            for (; j < SKIP_GRAM_WINDOW_SIZE;)
            {
                if ((*(pair->getRight()))[j] == index || (*(pair->getLeft()))[j] == index)
                {
                    break;
                }

                j++;
            }

            /*
                This block is handling the result of the first inner loop, which checks whether the randomly selected `index` 
                is found in the right or left context of the word pair (i.e., if it's part of the context window).

                - If `j == SKIP_GRAM_WINDOW_SIZE`, it means that the loop completed without finding the `index` in the context,
                so `index` is not part of the context and we are free to proceed further. 
                We reset `j` to 0 to ensure it's ready for the next loop, which checks for redundancy in the negative samples array.

                - If `j != SKIP_GRAM_WINDOW_SIZE`, it means the loop found that `index` is part of the context (either left or right),
                and therefore this word should not be selected as a negative sample. Setting `j = n` forces the outer loop
                to skip adding this word to the negative samples array by failing the subsequent redundancy check.
             */
            if (j == SKIP_GRAM_WINDOW_SIZE)
            {
                j = 0;
            }
            else 
            {
                j = n;
            }
            
            /* 
                Negative sampling with no redundancy.            
                The inner loop checks for redundancy by scanning the previously selected indices in the ptr array.
                This ensures that no duplicate samples are selected
             */            
            for (; j < i;)
            {
                if (ptr[j] == index)
                {
                    break;
                }

                j = j + 1;
            } 
            // True if no redundancy is found           
            if (j == i)
            {    
                ptr[i] = index;

                i = i + 1;
            }
        }

        // If true then negative sampling is completed, come out of universal loop
        if (i == n)
        {
            break;
        }
    }

    return ptr;
}
```
**Training Loop**, how and where to call the function `generateNegativeSamples_cbow()`?

```C++
#define CBOW_TRAINING_LOOP(el, epoch, lr, rs, ns, pairs, t, verbose, vocab, W1, W2, W1_best, W2_best)\
{\
    ------------
    --------------
    ----------------
    /* Epoch loop: Main loop for epochs */\
    for (cc_tokenizer::string_character_traits<char>::size_type i = 1; i <= epoch; i++)\
    {\
        while (pairs.go_to_next_word_pair() != cc_tokenizer::string_character_traits<char>::eof())\
        {\
            /* Get Current Word Pair: We've a pair, a pair is LEFT_CONTEXT_WORD/S CENTER_WORD and RIGHT_CONTEXT_WORD/S */\
            /* This is one Positive Sample */\
            WORDPAIRS_PTR pair = pairs.get_current_word_pair();\    
            Collective<cc_tokenizer::string_character_traits<char>::size_type> negative_samples = generateNegativeSamples_cbow(vocab, pair, ns);\
            ------------
            --------------
            ----------------
```
2. **During Forward Propagation**: Modify the forward propagation function to accept the negative samples as additional input. This way, the model can compute the probabilities(between 0 and 1) not just for the positive samples but also for the negative samples. One more thing is that thes computed probabilities are between 0 and 1 for both positive and negative samples.
    - `Positive predicted probabilities` (for positive samples) should be close to 1.
    - `Negative predicted probabilities` (for negative samples) should be close to 0.
```C++
template<typename E>
struct forward_propagation 
{
    --------------------------
    ------------------------------
    ---------------------------------

    forward_propagation<E>(Collective<E>& h, Collective<E>& y_pred, Collective<E>& u, Collective<E>& h_negative, Collective<E>& y_pred_negative, Collective<E>& u_negative) throw (ala_exception) 
    { 
        try
        {
            this->hidden_layer_vector = h;
            this->predicted_probabilities = y_pred;
            this->intermediate_activation = u;

            this->negative_hidden_layer_vector = h_negative;        
            this->negative_predicted_probabilities = y_pred_negative;
            this->negative_intermediate_activation = u_negative;
        }
        catch(ala_exception& e)
        {
            // Propagate existing ala_exception with additional context
            // NO cleanup performed assuming this is also a critical error
            throw ala_exception(cc_tokenizer::String<char>("forward_propagaton<E>::forward_propagation(Collective<E>&, Collective<E>&, Collective<E>&, Collective<E>&, Collective<E>&, Collective<E>&) -> ") + e.what());
        }                
    }

    --------------------------
    ------------------------------
    ---------------------------------

    /*
        Positive predicted probabilities accessor methods
    */
    E pb(cc_tokenizer::string_character_traits<char>::size_type i) throw (ala_exception)
    {
        if (i >= predicted_probabilities.getShape().getN())
        {
            throw ala_exception("forward_propagation::pb() Error: Provided index value is out of bounds.");
        }

        return predicted_probabilities[((i/predicted_probabilities.getShape().getNumberOfColumns())*predicted_probabilities.getShape().getNumberOfColumns() + i%predicted_probabilities.getShape().getNumberOfColumns())];
    }

    /*
        Negative predicted probabilities accessor methods
    */
    E npp(cc_tokenizer::string_character_traits<char>::size_type i) throw (ala_exception)
    {
        if (i >= negative_predicted_probabilities.getShape().getN())
        {
            throw ala_exception("forward_propagation::ppb() Error: Provided index value is out of bounds.");
        }

        return negative_predicted_probabilities[((i/negative_predicted_probabilities.getShape().getNumberOfColumns())*negative_predicted_probabilities.getShape().getNumberOfColumns() + i%negative_predicted_probabilities.getShape().getNumberOfColumns())];
    }    

    --------------------------
    ------------------------------
    ---------------------------------

    private:           
        Collective<E> hidden_layer_vector; // E* h        
        Collective<E> predicted_probabilities; // E* y_pred           
        Collective<E> intermediate_activation; // E* u
        Collective<E> negative_hidden_layer_vector;  // E* h_negative
        Collective<E> negative_predicted_probabilities;  // E* y_pred_negative               
        Collective<E> negative_intermediate_activation; // E* u_negative;
};
```
```C++
template <typename E = double>
forward_propagation<E> forward(Collective<E>& W1, Collective<E>& W2, Collective<cc_tokenizer::string_character_traits<char>::size_type>& negative_context, CORPUS_REF vocab, WORDPAIRS_PTR pair = NULL) throw (ala_exception)
{
    if (pair == NULL)
    {
        throw ala_exception("forward() Error: Null pointer passed, expected a valid WORDPAIRS_PTR.");
    }

    --------------------------
    ------------------------------
    ---------------------------------

    try
    {
        --------------------------
        ------------------------------
        ---------------------------------

        Collective<E> y_pred = Numcy::sigmoid<E>(u) /*softmax<E>(u)*/;

        --------------------------
        ------------------------------
        ---------------------------------

        Collective<E> h_negative, u_negative, y_pred_negative;

        if (negative_context.getShape().getN())
        {
            h_negative = Numcy::mean(W1, negative_context);
            u_negative = Numcy::dot(h_negative, W2);
            y_pred_negative = Numcy::sigmoid(u_negative) /*softmax(u_negative)*/; 
        }

        return forward_propagation<E>{h, y_pred, u, h_negative, y_pred_negative, u_negative,};
    }
    catch(std::bad_alloc& e)
    {
        // CRITICAL: Memory allocation failure - system should terminate immediately
        // NO cleanup performed - this is a fatal error requiring process exit
        throw ala_exception(cc_tokenizer::String<char>("forward(Collective<E>&, Collective<E>&, Collective<cc_tokenizer::string_character_traits<char>::size_type>&, CORPUS_REF, WORDPAIRS_PTR) Error: ") + cc_tokenizer::String<char>(e.what()));
    }
    catch(std::length_error& e)
    {
        // CRITICAL: Length constraint violation - system should terminate immediately
        // NO cleanup performed - this is a fatal error requiring process exit
        throw ala_exception(cc_tokenizer::String<char>("forward(Collective<E>&, Collective<E>&, Collective<cc_tokenizer::string_character_traits<char>::size_type>&, CORPUS_REF, WORDPAIRS_PTR) Error: ") + cc_tokenizer::String<char>(e.what()));
    }
    catch(ala_exception& e)
    {
        // Propagate existing ala_exception with additional context
        // NO cleanup performed assuming this is also a critical error
        throw ala_exception(cc_tokenizer::String<char>("forward(Collective<E>&, Collective<E>&, Collective<cc_tokenizer::string_character_traits<char>::size_type>&, CORPUS_REF, WORDPAIRS_PTR) -> ") + cc_tokenizer::String<char>(e.what()));
    }          
}
```

3. **During Loss Calculation**: Update the loss function to consider both the positive and negative samples. Instead of only using **Negative Log Likelihood** (`NLL`) for the positive samples, it will also include terms for the negative samples, which should contribute to a lower loss when correctly identified as non-context words.
    - **Positive Loss Calculation** :  `−log(p)` where `p` is the predicted probability that positive sample is postive. That is when `p` is close to `1` then, `-log(p)` is very close to `0` which means a small loss.
    - **Negative Loss Calculation** :  `-log(1 − q)` where `q` is the predicted probability that negative sample is positive. That is when `q` is close to `0` then `-log(1 - q)` is very close to `0` as well, which means a small loss.
        - If the model incorrectly predicts that a negative sample is actually a positive sample (i.e., `𝑞` is high, close to `1`), then `1 − 𝑞` becomes very small, and the `−log(1 − 𝑞)` becomes large. This penalizes the model for making such mistakes.

The total loss (or cost function) you compute is a sum of the positive and negative losses: Total Loss = Positive Loss + ∑Negative Loss

```C++
// Initialize the loss
E el = 0.0;

// Calculate positive loss (for the center word)
E positive_loss = -1 * log(fp.ppp(pair->getCenterWord() - INDEX_ORIGINATES_AT_VALUE));
el += positive_loss;

// Calculate negative loss (for negative samples)
if (negative_samples != NULL && num_negative_samples > 0) 
{
    for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < num_negative_samples; ++i)
    {
        // Assuming y_pred_negative is a Collective<E> that holds probabilities for negative samples
        E negative_loss = -1 * log(1 - fp.npp(i)); // Calculate loss for each negative sample
        el += negative_loss;
    }
}

// el now contains the total loss including both positive and negative samples
```

​
   
