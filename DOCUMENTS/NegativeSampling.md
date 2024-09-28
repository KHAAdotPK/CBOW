#### Implementation of Negative Sampling in this CBOW model.
---
**What are Positive Samples**?
- For a given center word, you have a window of context words around it.
- In the CBOW model, the context words are the input, and the center word is the output.
- The word-context pairs you extract from your corpus are the positive samples because they represent the actual relationships between words that occur in your text.

**What are Negative Samples**?
Negative samples are randomly selected words from the vocabulary that are not related to the current context words or the current center/target word. For each positive sample, multiple negative samples are generated. Negative sampling helps the model learn by contrasting the positive samples with these randomly selected negative samples. By providing random words as negative samples, you're showing the model that those words should not be predicted as the center word given the current context.

In the CBOW training loop, Negative Sampling can be implemented within the training loop, specifically in the section where the forward and backward propagation occur. Here’s how it can be integrated
1. **Before Forward Propagation**: After retrieving the current word pair(`a positive sample`), you can introduce a method to generate negative samples. This would involve selecting a set of negative words (i.e., words that are not related to the current context) based on your vocabulary. **In this method, make sure to avoid selecting the center/target word itself or any of the context words as negative samples**.
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
**Training Loop**, how and where to call the function `generateNegativeSamples_cbow()`?
```C++
while (pairs.go_to_next_word_pair() != cc_tokenizer::string_character_traits<char>::eof())\
{\
    /* Get Current Word Pair: We've a pair, a pair is LEFT_CONTEXT_WORD/S CENTER_WORD and RIGHT_CONTEXT_WORD/S */\
    /* This is one Positive Sample */\
    WORDPAIRS_PTR pair = pairs.get_current_word_pair();\
    cc_tokenizer::string_character_traits<char>::size_type* negative_samples_ptr = generateNegativeSamples_cbow(vocab, pair, static_cast<cc_tokenizer::string_character_traits<char>::size_type>(CBOW_NEGATIVE_SAMPLE_SIZE));\
    ------------
    -------------
    --------------
    /* Deallocate Negative Samples array. The function generateNegativeSamples_cbow() returns a pointer to array of negative samples. The calling function is responsible for deallocation */\
    cc_tokenizer::allocator<cc_tokenizer::string_character_traits<char>::size_type>().deallocate(negative_samples_ptr);\
```
2. **During Forward Propagation**: Modify the forward function to accept the negative samples as additional input. This way, the model can compute the probabilities not just for the positive context words but also for the negative samples.
```C++            
    struct forward_propogation<E>(Collective<E>& h, Collective<E>& y_pred_positive, Collective<E>& u_positive, Collective<E>& y_pred_negative, Collective<E>& u_negative)
    {           
        E* ptr = NULL;

        try 
        {                    
            ptr = cc_tokenizer::allocator<E>().allocate(h.getShape().getN());
            for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < h.getShape().getN(); i++)
            {
                ptr[i] = h[i];                
            }
        }
        catch (std::length_error& e)
        {
            throw ala_exception(cc_tokenizer::String<char>("forward_propogation() Error: ") + cc_tokenizer::String<char>(e.what())); 
        }
        catch (std::bad_alloc& e)
        {
           throw ala_exception(cc_tokenizer::String<char>("forward_propogation() Error: ") + cc_tokenizer::String<char>(e.what())); 
        }
        catch (ala_exception& e)
        {
            throw ala_exception(cc_tokenizer::String<char>("forward_propogation() Error: ") + cc_tokenizer::String<char>(e.what()));
        }
        hidden_layer_vector = Collective<E>{ptr, h.getShape().copy()};

        try
        {                 
            ptr = cc_tokenizer::allocator<E>().allocate(y_pred_positive.getShape().getN());
            for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < y_pred_positive.getShape().getN(); i++)
            {
                ptr[i] = y_pred_positive[i];
            }
        } 
        catch (std::length_error& e)
        {
            throw ala_exception(cc_tokenizer::String<char>("forward_propogation() Error: ") + cc_tokenizer::String<char>(e.what())); 
        }
        catch (std::bad_alloc& e)
        {
            throw ala_exception(cc_tokenizer::String<char>("forward_propogation() Error: ") + cc_tokenizer::String<char>(e.what())); 
        }      
        catch (ala_exception& e)
        {
            throw ala_exception(cc_tokenizer::String<char>("forward_propogation() Error: ") + cc_tokenizer::String<char>(e.what()));
        }
        positive_predicted_probabilities = Collective<E>{ptr, y_pred_positive.getShape().copy()};

        try
        {        
            ptr = cc_tokenizer::allocator<E>().allocate(u_positive.getShape().getN());
            for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < u_positive.getShape().getN(); i++)
            {
                ptr[i] = u_positive[i];
            }
        }
        catch (std::length_error& e)
        {
            throw ala_exception(cc_tokenizer::String<char>("forward_propogation() Error: ") + cc_tokenizer::String<char>(e.what())); 
        }
        catch (std::bad_alloc& e)
        {
            throw ala_exception(cc_tokenizer::String<char>("forward_propogation() Error: ") + cc_tokenizer::String<char>(e.what())); 
        }      
        catch (ala_exception& e)
        {
            throw ala_exception(cc_tokenizer::String<char>("forward_propogation() Error: ") + cc_tokenizer::String<char>(e.what()));
        }
        positive_intermediate_activation = Collective<E>{ptr, u_positive.getShape().copy()};

        try
        {                 
            ptr = cc_tokenizer::allocator<E>().allocate(y_pred_negative.getShape().getN());
            for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < y_pred_negative.getShape().getN(); i++)
            {
                ptr[i] = y_pred_negative[i];
            }
        } 
        catch (std::length_error& e)
        {
            throw ala_exception(cc_tokenizer::String<char>("forward_propogation() Error: ") + cc_tokenizer::String<char>(e.what())); 
        }
        catch (std::bad_alloc& e)
        {
            throw ala_exception(cc_tokenizer::String<char>("forward_propogation() Error: ") + cc_tokenizer::String<char>(e.what())); 
        }      
        catch (ala_exception& e)
        {
            throw ala_exception(cc_tokenizer::String<char>("forward_propogation() Error: ") + cc_tokenizer::String<char>(e.what()));
        }
        negative_predicted_probabilities = Collective<E>{ptr, y_pred_negative.getShape().copy()};

        try
        {        
            ptr = cc_tokenizer::allocator<E>().allocate(u_negative.getShape().getN());
            for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < u_negative.getShape().getN(); i++)
            {
                ptr[i] = u_negative[i];
            }
        }
        catch (std::length_error& e)
        {
            throw ala_exception(cc_tokenizer::String<char>("forward_propogation() Error: ") + cc_tokenizer::String<char>(e.what())); 
        }
        catch (std::bad_alloc& e)
        {
            throw ala_exception(cc_tokenizer::String<char>("forward_propogation() Error: ") + cc_tokenizer::String<char>(e.what())); 
        }      
        catch (ala_exception& e)
        {
            throw ala_exception(cc_tokenizer::String<char>("forward_propogation() Error: ") + cc_tokenizer::String<char>(e.what()));
        }
        negative_intermediate_activation = Collective<E>{ptr, u_negative.getShape().copy()};
    }

    forward_propogation<E>(forward_propogation<E>& other) 
    {   
        try
        {
            E* ptr = cc_tokenizer::allocator<E>().allocate(other.hidden_layer_vector.getShape().getN());
            for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < other.hidden_layer_vector.getShape().getN(); i++)
            {
                ptr[i] = other.hidden_layer_vector[i];
            }
            hidden_layer_vector = Collective<E>{ptr, other.hidden_layer_vector.getShape().copy()};

            ptr = cc_tokenizer::allocator<E>().allocate(other.positive_predicted_probabilities.getShape().getN());
            for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < other.positive_predicted_probabilities.getShape().getN(); i++)
            {
                ptr[i] = other.positive_predicted_probabilities[i];
            }
            positive_predicted_probabilities = Collective<E>{ptr, other.positive_predicted_probabilities.getShape().copy()};

            ptr = cc_tokenizer::allocator<E>().allocate(other.positive_intermediate_activation.getShape().getN());
            for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < other.positive_intermediate_activation.getShape().getN(); i++)
            {
                ptr[i] = other.positive_intermediate_activation[i];
            }
            positive_intermediate_activation = Collective<E>{ptr, other.positive_intermediate_activation.getShape().copy()};

            ptr = cc_tokenizer::allocator<E>().allocate(other.negative_predicted_probabilities.getShape().getN());
            for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < other.negative_predicted_probabilities.getShape().getN(); i++)
            {
                ptr[i] = other.negative_predicted_probabilities[i];
            }
            negative_predicted_probabilities = Collective<E>{ptr, other.negative_predicted_probabilities.getShape().copy()};

            ptr = cc_tokenizer::allocator<E>().allocate(other.negative_intermediate_activation.getShape().getN());
            for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < other.negative_intermediate_activation.getShape().getN(); i++)
            {
                ptr[i] = other.negative_intermediate_activation[i];
            }
            negative_intermediate_activation = Collective<E>{ptr, other.negative_intermediate_activation.getShape().copy()};
        }
        catch (std::length_error& e)
        {
            throw ala_exception(cc_tokenizer::String<char>("forward_propogation() Error: ") + cc_tokenizer::String<char>(e.what())); 
        }
        catch (std::bad_alloc& e)
        {
            throw ala_exception(cc_tokenizer::String<char>("forward_propogation() Error: ") + cc_tokenizer::String<char>(e.what())); 
        }      
        catch (ala_exception& e)
        {
            throw ala_exception(cc_tokenizer::String<char>("forward_propogation() Error: ") + cc_tokenizer::String<char>(e.what()));
        }       
    }

    forward_propogation<E>& operator= (forward_propogation<E>& other)    
    { 
        if (this == &other)
        {
            return *this;
        }

        E* ptr = NULL;          

        try 
        {
            ptr = cc_tokenizer::allocator<E>().allocate(other.hidden_layer_vector.getShape().getN());
            for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < other.hidden_layer_vector.getShape().getN(); i++)
            {
                ptr[i] = other.hidden_layer_vector[i];
            }        
        }
        catch (std::length_error& e)
        {
            throw ala_exception(cc_tokenizer::String<char>("forward_propogation::operator=() Error: ") + cc_tokenizer::String<char>(e.what())); 
        }
        catch (std::bad_alloc& e)
        {
           throw ala_exception(cc_tokenizer::String<char>("forward_propogation::operator=() Error: ") + cc_tokenizer::String<char>(e.what())); 
        }
        catch (ala_exception& e)
        {
            throw ala_exception(cc_tokenizer::String<char>("forward_propogation::operator=() Error: ") + cc_tokenizer::String<char>(e.what()));
        }
        hidden_layer_vector = Collective<E>{ptr, other.hidden_layer_vector.getShape().copy()};

        try
        {                
            ptr = cc_tokenizer::allocator<E>().allocate(other.positive_predicted_probabilities.getShape().getN());
            for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < other.positive_predicted_probabilities.getShape().getN(); i++)
            {
                ptr[i] = other.positive_predicted_probabilities[i];
            }
        }
        catch (std::length_error& e)
        {
            throw ala_exception(cc_tokenizer::String<char>("forward_propogation::operator=() Error: ") + cc_tokenizer::String<char>(e.what())); 
        }
        catch (std::bad_alloc& e)
        {
           throw ala_exception(cc_tokenizer::String<char>("forward_propogation::operator=() Error: ") + cc_tokenizer::String<char>(e.what())); 
        }
        catch (ala_exception& e)
        {
            throw ala_exception(cc_tokenizer::String<char>("forward_propogation::operator=() Error: ") + cc_tokenizer::String<char>(e.what()));
        }
        positive_predicted_probabilities = Collective<E>{ptr, other.predicted_probabilities.getShape().copy()};

        try
        {                
            ptr = cc_tokenizer::allocator<E>().allocate(other.negative_predicted_probabilities.getShape().getN());
            for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < other.negative_predicted_probabilities.getShape().getN(); i++)
            {
                ptr[i] = other.negative_predicted_probabilities[i];
            }
        }
        catch (std::length_error& e)
        {
            throw ala_exception(cc_tokenizer::String<char>("forward_propogation::operator=() Error: ") + cc_tokenizer::String<char>(e.what())); 
        }
        catch (std::bad_alloc& e)
        {
           throw ala_exception(cc_tokenizer::String<char>("forward_propogation::operator=() Error: ") + cc_tokenizer::String<char>(e.what())); 
        }
        catch (ala_exception& e)
        {
            throw ala_exception(cc_tokenizer::String<char>("forward_propogation::operator=() Error: ") + cc_tokenizer::String<char>(e.what()));
        }
        negative_predicted_probabilities = Collective<E>{ptr, other.negative_predicted_probabilities.getShape().copy()};

        try
        {        
            ptr = cc_tokenizer::allocator<E>().allocate(other.positive_intermediate_activation.getShape().getN());
            for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < other.positive_intermediate_activation.getShape().getN(); i++)
            {
                ptr[i] = other.positive_intermediate_activation[i];
            }
        }
        catch (std::length_error& e)
        {
            throw ala_exception(cc_tokenizer::String<char>("forward_propogation::operator=() Error: ") + cc_tokenizer::String<char>(e.what())); 
        }
        catch (std::bad_alloc& e)
        {
           throw ala_exception(cc_tokenizer::String<char>("forward_propogation::operator=() Error: ") + cc_tokenizer::String<char>(e.what())); 
        }
        catch (ala_exception& e)
        {
            throw ala_exception(cc_tokenizer::String<char>("forward_propogation::operator=() Error: ") + cc_tokenizer::String<char>(e.what()));
        }
        positive_intermediate_activation = Collective<E>{ptr, other.positive_intermediate_activation.getShape().copy()};

        try
        {        
            ptr = cc_tokenizer::allocator<E>().allocate(other.negative_intermediate_activation.getShape().getN());
            for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < other.negative_intermediate_activation.getShape().getN(); i++)
            {
                ptr[i] = other.negative_intermediate_activation[i];
            }
        }
        catch (std::length_error& e)
        {
            throw ala_exception(cc_tokenizer::String<char>("forward_propogation::operator=() Error: ") + cc_tokenizer::String<char>(e.what())); 
        }
        catch (std::bad_alloc& e)
        {
           throw ala_exception(cc_tokenizer::String<char>("forward_propogation::operator=() Error: ") + cc_tokenizer::String<char>(e.what())); 
        }
        catch (ala_exception& e)
        {
            throw ala_exception(cc_tokenizer::String<char>("forward_propogation::operator=() Error: ") + cc_tokenizer::String<char>(e.what()));
        }
        negative_intermediate_activation = Collective<E>{ptr, other.negative_intermediate_activation.getShape().copy()};
        
        return *this;
    }
    
    /*
        Hidden Layer Vector accessor methods
     */
    E hlv(cc_tokenizer::string_character_traits<char>::size_type i) throw (ala_exception)
    {
        if (i >= hidden_layer_vector.getShape().getN())
        {
            throw ala_exception("forward_propogation::hlv() Error: Provided index value is out of bounds.");
        }

        return hidden_layer_vector[((i/hidden_layer_vector.getShape().getNumberOfColumns())*hidden_layer_vector.getShape().getNumberOfColumns() + i%hidden_layer_vector.getShape().getNumberOfColumns())];
    }
    DIMENSIONS hlvShape(void)
    {
        return *(hidden_layer_vector.getShape().copy());
    }

    /*
        Positive Predicted Probabilities accesssor methods
     */
    E ppb(cc_tokenizer::string_character_traits<char>::size_type i) throw (ala_exception)
    {
        if (i >= positive_predicted_probabilities.getShape().getN())
        {
            throw ala_exception("forward_propogation::ppb() Error: Provided index value is out of bounds.");
        }

        return positive_predicted_probabilities[((i/positive_predicted_probabilities.getShape().getNumberOfColumns())*positive_predicted_probabilities.getShape().getNumberOfColumns() + i%positive_predicted_probabilities.getShape().getNumberOfColumns())];
    }    
    DIMENSIONS ppbShape(void)
    {
        return *(positive_predicted_probabilities.getShape().copy());
    }

    /*
        Negative Predicted Probabilities accesssor methods
     */
    E npb(cc_tokenizer::string_character_traits<char>::size_type i) throw (ala_exception)
    {
        if (i >= negative_predicted_probabilities.getShape().getN())
        {
            throw ala_exception("forward_propogation::ppb() Error: Provided index value is out of bounds.");
        }

        return negative_predicted_probabilities[((i/negative_predicted_probabilities.getShape().getNumberOfColumns())*negative_predicted_probabilities.getShape().getNumberOfColumns() + i%negative_predicted_probabilities.getShape().getNumberOfColumns())];
    }    
    DIMENSIONS npbShape(void)
    {
        return *(negative_predicted_probabilities.getShape().copy());
    }

    /*
        Positive Intermediate Activation accesssor methods
     */
    E pia(cc_tokenizer::string_character_traits<char>::size_type i) throw (ala_exception)
    {
        if (i >= positve_intermediate_activation.getShape().getN())
        {
            throw ala_exception("forward_propogation::ia() Error: Provided index value is out of bounds.");
        }

        return positive_intermediate_activation[((i/positive_intermediate_activation.getShape().getNumberOfColumns())*positive_intermediate_activation.getShape().getNumberOfColumns() + i%positive_intermediate_activation.getShape().getNumberOfColumns())];
    }
    DIMENSIONS piaShape(void)
    {
        return *(positive_intermediate_activation.getShape().copy());
    }

    /*
        Negative Intermediate Activation accesssor methods
     */
    E nia(cc_tokenizer::string_character_traits<char>::size_type i) throw (ala_exception)
    {
        if (i >= negative_intermediate_activation.getShape().getN())
        {
            throw ala_exception("forward_propogation::ia() Error: Provided index value is out of bounds.");
        }

        return negative_intermediate_activation[((i/negative_intermediate_activation.getShape().getNumberOfColumns())*negative_intermediate_activation.getShape().getNumberOfColumns() + i%negative_intermediate_activation.getShape().getNumberOfColumns())];
    }
    DIMENSIONS piaShape(void)
    {
        return *(negative_intermediate_activation.getShape().copy());
    }


    /*
        Declare forward as a friend function within the struct. It is templated, do we need it like this.
     */    
    /*
        Documentation Note:
        -------------------
        The default argument for the template parameter is causing the following error during compilation:
    
        D:\ML\Embedding-Algorithms\Word2Vec\skip-gram\ML\Embedding-Algorithms\Word2Vec\skip-gram\skip-gram.hh(263): warning C4348: 'forward': redefinition of default parameter: parameter 1
        D:\ML\Embedding-Algorithms\Word2Vec\skip-gram\ML\Embedding-Algorithms\Word2Vec\skip-gram\skip-gram.hh(355): note: see declaration of 'forward'
        D:\ML\Embedding-Algorithms\Word2Vec\skip-gram\ML\Embedding-Algorithms\Word2Vec\skip-gram\skip-gram.hh(272): note: the template instantiation context (the oldest one first) is
        main.cpp(169): note: see reference to class template instantiation 'forward_propagation<double>' being compiled

        This error occurs at compile time because the friend declaration and the actual definition of the function both use the default argument for the template parameter. 
        To resolve this error, remove the default argument from either the friend declaration or the definition. 

        Example problematic friend declaration:
    
        template <typename T = double>
        friend forward_propagation<T> forward(Collective<T>&, Collective<T>&, CORPUS_REF, WORDPAIRS_PTR, bool) throw (ala_exception);

        Additional details about the friend declaration:
        The above friend declaration is ineffective because no instance of the vector/composite class is being passed to this function as an argument.
        Therefore, the function cannot access the private or protected members of the vector/composite class it is declared as a friend of.
     */
        
    template <typename T>
    friend backward_propogation<T> backward(Collective<T>&, Collective<T>&, CORPUS_REF, forward_propogation<T>&, WORDPAIRS_PTR, bool) throw (ala_exception);
        
    /*
        TODO, uncomment the following statement and make all variables/properties of this vector private.
     */                       
    /*private:*/
        /*
            In the context of our CBOW/Skip-Gram model, h refers to the hidden layer vector obtained by averaging the embeddings of the context words.
            It is used in both the forward and backward passes of the neural network.

            The shape of this array is (1, SKIP_GRAM_EMBEDDNG_VECTOR_SIZE), 
            a single row vector with the size of the embedding dimension.
         */
        //E* h;
        Collective<E> hidden_layer_vector;
        /*
            y_pred/y_pred_positive is a Numcy array of predicted probabilities of the output word given the input context. 
            In our implementation, it is the output of the forward propagation step.

            The shape of this array is (1, len(vocab)), indicating a single row vector with the length of the vocabulary and 
            where each element corresponds to the predicted probability of a specific word.
         */
        //E* y_pred;
        Collective<E> positive_predicted_probabilities;  

        /*	
            Represents an intermediat gradient.	 
            This vector has shape (1, len(vocab)), similar to y_pred. 
            It represents the result of the dot product operation between the center or target word vector "h" and the weight matrix W2.
            The result stored in "u” captures the combined influence of hidden neurons on predicting context words. It provides a
            numerical representation of how likely each word in the vocabulary is to be a context word of a given target 
            word (within the skip-gram model).

            The variable "u or positive_u" serves as an intermediary step in the forward pass, representing the activations before applying 
            the “softmax” function to generate the predicted probabilities. 

            It represents internal state in the neural network during the working of "forward pass".
            This intermediate value is used in calculations involving gradients in "backward pass" or "back propogation"(the function backward).
         */
        //E* u;
        Collective<E> positive_intermediate_activation; 

        /*
            Predicted probabilities for the negative samples, i.e., words that are not related to the context or target word.
        
            Shape: (N, len(vocab)), where N is the number of negative samples and each element represents the probability of
            a specific word being incorrectly predicted as context.
         */
        Collective<E> negative_predicted_probabilities;

        /*	
            Intermediate activations from the dot product of the hidden layer and the weight matrix for the negative samples.
            This vector represents the influence of hidden neurons on predicting negative samples.
        
            Shape: (N, len(vocab)), where N is the number of negative samples.
         */
        Collective<E> negative_intermediate_activation;
};
```
```C++
template <typename E = double>
forward_propogation<E> forward(Collective<E>& W1, Collective<E>& W2, CORPUS_REF vocab, WORDPAIRS_PTR pair = NULL, cc_tokenizer::string_character_traits<char>::size_type* negative_samples = NULL, cc_tokenizer::string_character_traits<char>::size_type num_negative_samples = 0) throw (ala_exception)
{    
    if (pair == NULL)
    {
        throw ala_exception("forward() Error: Null pointer passed, expected a valid WORDPAIRS_PTR.");
    }

    // Positive samples (context words) initialization
    cc_tokenizer::string_character_traits<char>::size_type* ptr = NULL;
    cc_tokenizer::string_character_traits<char>::size_type n = 0, j = 0;

    // Calculate the number of valid context words
    for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < SKIP_GRAM_WINDOW_SIZE; i++)
    {
        if ((*(pair->getLeft()))[i] >= INDEX_ORIGINATES_AT_VALUE)
        {
            n = n + 1;
        }

        if ((*(pair->getRight()))[i] >= INDEX_ORIGINATES_AT_VALUE)
        {
            n = n + 1;
        }        
    }
    
    try
    {
        ptr = cc_tokenizer::allocator<cc_tokenizer::string_character_traits<char>::size_type>().allocate(n);

        // Store context words in ptr
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

        Collective<cc_tokenizer::string_character_traits<char>::size_type> context = Collective<cc_tokenizer::string_character_traits<char>::size_type>{ptr, DIMENSIONS{n, 1, NULL, NULL}};

        // Compute hidden layer (mean of context words' embeddings)
        Collective<E> h = Numcy::mean(W1, context);

        // Compute predictions for positive samples
        Collective<E> u_positive = Numcy::dot(h, W2);
        Collective<E> y_pred_positive = softmax<E>(u_positive);

        // If negative samples exist, compute their probabilities
        Collective<E> u_negative, y_pred_negative;
        if (negative_samples != NULL && num_negative_samples > 0)
        {
            // Convert negative samples to Collective format
            Collective<cc_tokenizer::string_character_traits<char>::size_type> neg_samples = Collective<cc_tokenizer::string_character_traits<char>::size_type>{negative_samples, DIMENSIONS{num_negative_samples, 1, NULL, NULL}};

            // Compute the dot product for negative samples
            u_negative = Numcy::dot(h, W2, neg_samples);
            y_pred_negative = softmax<E>(u_negative);
        }

        // Return the results (you may want to store both positive and negative predictions for later use)
        return forward_propogation<E>{h, y_pred_positive, u_positive, y_pred_negative, u_negative};
    }
    catch(std::bad_alloc& e)
    {
        throw ala_exception(cc_tokenizer::String<char>("forward() Error: ") + cc_tokenizer::String<char>(e.what()));
    }
    catch(std::length_error& e)
    {
        throw ala_exception(cc_tokenizer::String<char>("forward() Error: ") + cc_tokenizer::String<char>(e.what()));
    }
    catch(ala_exception& e)
    {
        throw ala_exception(cc_tokenizer::String<char>("forward() Error: ") + cc_tokenizer::String<char>(e.what()));
    }                    
}
```