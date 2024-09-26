#### Implementation of Negative Sampling in this CBOW model.
---
In the CBOW training loop, Negative Sampling can be implemented within the training loop, specifically in the section where the forward and backward propagation occur. Here’s how it can be integrated
1. **Before Forward Propagation**: After retrieving the current word pair, you can introduce a method to generate negative samples. This would involve selecting a set of negative words (i.e., words that are not related to the current context) based on your vocabulary. **In this method, make sure to avoid selecting the center/target word itself or any of the context words as negative samples**.
 ```C++
/*    
    In CBOW, we are predicting the central word, not the context.
    So, the negative samples should be words that are not the central word (target) but could be drawn from the entire vocabulary.

    A better way to phrase it:
    We need to randomly select words from the vocabulary that are not the target (central) word when given a context of surrounding words.
    These negative samples help the model learn to distinguish between the true target word and unrelated words.

    Is it ok to not find a single vocabulary word which is not in the center of context window?
    In a large enough corpus, every word in the vocabulary could potentially be the central word for some context window at some point.                           
 */
/**
 * @brief Generates negative samples for the CBOW model using negative sampling.
 *
 * This function selects a set of negative samples (words) that are not related to the current target word (central word) being predicted.
 * Negative samples are randomly drawn from the vocabulary,
 * and the function ensures that the generated negative samples do not include the target word (central word) of the current context.
 * These negative samples are used to train the model to differentiate between the correct (target) word and unrelated (negative) words.
 *
 * @tparam E - Type of the elements, typically used for indexing and storing sizes.
 *             Defaults to the size type of cc_tokenizer::string_character_traits<char>.
 *
 * @param vocab - Reference to the corpus vocabulary from which to generate negative samples. 
 *                It must contain the method numberOfUniqueTokens(), which returns the total number 
 *                of unique words in the vocabulary.
 *
 * @param pair - Pointer to the word pair, representing the center/target word, left and right context words in the CBOW model.
 *               The pair object should have methods getLeft() and getRight() that return arrays of context words.
 *
 * @param n - The number of negative samples to generate. Defaults to the size of the skip-gram window (SKIP_GRAM_WINDOW_SIZE).
 *          - Typically, a small number of negative samples (5-20 per positive sample) are selected to avoid too many unnecessary computations.
 *
 * @throws ala_exception - Throws this exception in case of memory allocation errors (`std::bad_alloc`) 
 *                         or length issues (`std::length_error`) during dynamic memory allocation.
 *
 * @return cc_tokenizer::string_character_traits<char>::size_type* - Returns a pointer to an array of negative samples.
 *                                                                   The array contains `n` negative samples, where 
 *                                                                   each sample is an index into the vocabulary.
 *                                                                   The caller is responsible for deallocating this memory 
 *                                                                   to avoid memory leaks.
 */
template <typename E = cc_tokenizer::string_character_traits<char>::size_type>
cc_tokenizer::string_character_traits<char>::size_type* generateNegativeSamples_cbow(CORPUS_REF vocab, WORDPAIRS_PTR pair, E n = SKIP_GRAM_WINDOW_SIZE) throw (ala_exception)
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
                Negative sampling with no redundency.            
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
            // True if no redundency is found           
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