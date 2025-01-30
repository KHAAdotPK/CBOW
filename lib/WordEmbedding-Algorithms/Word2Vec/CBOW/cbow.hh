/*
    lib/WordEmbedding-Algorithms/Word2Vec/CBOW/cbow.hh
    Q@khaa.pk
 */

#ifndef KHAA_DOT_PK_WORD_EMBEDDING_ALGORITHMS_CBOW_HH
#define KHAA_DOT_PK_WORD_EMBEDDING_ALGORITHMS_CBOW_HH

#include "header.hh"

/*
    Without the following declaration, this error occures at compile time... 
    D:\KHAAdotPK\CBOW\lib\WordEmbedding-Algorithms\Word2Vec\CBOW\cbow.hh(287): error C2988: unrecognizable template declaration/definition
    D:\KHAAdotPK\CBOW\lib\WordEmbedding-Algorithms\Word2Vec\CBOW\cbow.hh(17): note: while compiling class template 'forward_propogation'
 */
template<typename E>
struct backward_propogation; 

template<typename E>
struct forward_propogation 
{
    /*
        In the first constructor, forward_propagation(),
        member variables hidden_layer_vector, predicted_probabilities, and intermediate_activation
        are initialized directly in the initialization list.
        This approach is cleaner and more efficient than assigning them inside the constructor body.
     */
    forward_propogation(void) : hidden_layer_vector(Collective<E>{NULL, DIMENSIONS{0, 0, NULL, NULL}}), predicted_probabilities(Collective<E>{NULL, DIMENSIONS{0, 0, NULL, NULL}}), intermediate_activation(Collective<E>{NULL, DIMENSIONS{0, 0, NULL, NULL}})
    {        
    }

    /*
        TODO, 
        Use of Initialization Lists: Utilize constructor initialization lists to initialize
        member variables rather than assigning them inside the constructor body. This improves efficiency and readability...
        implemented but still commented out from the implementation of function.
     */
    //forward_propogation<E>(Collective<E>& h, Collective<E>& y_pred, Collective<E>& u) : hidden_layer_vector(h), predicted_probabilities(y_pred), intermediate_activation(u)
    forward_propogation<E>(Collective<E>& h, Collective<E>& y_pred, Collective<E>& u) /*: hidden_layer_vector(h), predicted_probabilities(y_pred), intermediate_activation(u) */
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
            ptr = cc_tokenizer::allocator<E>().allocate(y_pred.getShape().getN());
            for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < y_pred.getShape().getN(); i++)
            {
                ptr[i] = y_pred[i];
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
        predicted_probabilities = Collective<E>{ptr, y_pred.getShape().copy()};

        try
        {        
            ptr = cc_tokenizer::allocator<E>().allocate(u.getShape().getN());
            for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < u.getShape().getN(); i++)
            {
                ptr[i] = u[i];
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
        intermediate_activation = Collective<E>{ptr, u.getShape().copy()};
    }

    forward_propogation<E>(forward_propogation<E>& other) 
    {           
        E* ptr = cc_tokenizer::allocator<E>().allocate(other.hidden_layer_vector.getShape().getN());
        for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < other.hidden_layer_vector.getShape().getN(); i++)
        {
            ptr[i] = other.hidden_layer_vector[i];
        }
        hidden_layer_vector = Collective<E>{ptr, other.hidden_layer_vector.getShape().copy()};

        ptr = cc_tokenizer::allocator<E>().allocate(other.predicted_probabilities.getShape().getN());
        for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < other.predicted_probabilities.getShape().getN(); i++)
        {
            ptr[i] = other.predicted_probabilities[i];
        }
        predicted_probabilities = Collective<E>{ptr, other.predicted_probabilities.getShape().copy()};

        ptr = cc_tokenizer::allocator<E>().allocate(other.intermediate_activation.getShape().getN());
        for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < other.intermediate_activation.getShape().getN(); i++)
        {
            ptr[i] = other.intermediate_activation[i];
        }
        intermediate_activation = Collective<E>{ptr, other.intermediate_activation.getShape().copy()};
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
            ptr = cc_tokenizer::allocator<E>().allocate(other.predicted_probabilities.getShape().getN());
            for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < other.predicted_probabilities.getShape().getN(); i++)
            {
                ptr[i] = other.predicted_probabilities[i];
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
        predicted_probabilities = Collective<E>{ptr, other.predicted_probabilities.getShape().copy()};

        try
        {        
            ptr = cc_tokenizer::allocator<E>().allocate(other.intermediate_activation.getShape().getN());
            for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < other.intermediate_activation.getShape().getN(); i++)
            {
                ptr[i] = other.intermediate_activation[i];
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
        intermediate_activation = Collective<E>{ptr, other.intermediate_activation.getShape().copy()};
        
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
        Predicted Probabilities accesssor methods
     */
    E pb(cc_tokenizer::string_character_traits<char>::size_type i) throw (ala_exception)
    {
        if (i >= predicted_probabilities.getShape().getN())
        {
            throw ala_exception("forward_propogation::pb() Error: Provided index value is out of bounds.");
        }

        return predicted_probabilities[((i/predicted_probabilities.getShape().getNumberOfColumns())*predicted_probabilities.getShape().getNumberOfColumns() + i%predicted_probabilities.getShape().getNumberOfColumns())];
    }
    DIMENSIONS pbShape(void)
    {
        return *(predicted_probabilities.getShape().copy());
    }

     /*
        Intermediate Activation accesssor methods
     */
    E ia(cc_tokenizer::string_character_traits<char>::size_type i) throw (ala_exception)
    {
        if (i >= intermediate_activation.getShape().getN())
        {
            throw ala_exception("forward_propogation::ia() Error: Provided index value is out of bounds.");
        }

        return intermediate_activation[((i/intermediate_activation.getShape().getNumberOfColumns())*intermediate_activation.getShape().getNumberOfColumns() + i%intermediate_activation.getShape().getNumberOfColumns())];
    }
    DIMENSIONS iaShape(void)
    {
        return *(intermediate_activation.getShape().copy());
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
            y_pred is a Numcy array of predicted probabilities of the output word given the input context. 
            In our implementation, it is the output of the forward propagation step.

            The shape of this array is (1, len(vocab)), indicating a single row vector with the length of the vocabulary and 
            where each element corresponds to the predicted probability of a specific word.
         */
        //E* y_pred;
        Collective<E> predicted_probabilities;  

        /*	
            Represents an intermediat gradient.	 
            This vector has shape (1, len(vocab)), similar to y_pred. 
            It represents the result of the dot product operation between the center or target word vector "h" and the weight matrix W2.
            The result stored in "u” captures the combined influence of hidden neurons on predicting context words. It provides a
            numerical representation of how likely each word in the vocabulary is to be a context word of a given target 
            word (within the skip-gram model).

            The variable "u" serves as an intermediary step in the forward pass, representing the activations before applying 
            the “softmax” function to generate the predicted probabilities. 

            It represents internal state in the neural network during the working of "forward pass".
            This intermediate value is used in calculations involving gradients in "backward pass" or "back propogation"(the function backward).
         */
        //E* u;
        Collective<E> intermediate_activation; 

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

/*
    The following structure is a container designed to hold gradients calculated during backpropagation
    in a two-layer neural network used for word embeddings. The presence of grad_W1 and grad_W2 implies a 
    neural network with two layers. W1 represents the weights between the input layer and the first hidden layer, 
    and W2 represents the weights between the first hidden layer and the output layer.
    - The gradients (partial derivatives of the loss function) with respect to the network's weights
    - Backpropagation, this structure plays a crucial role in backpropagation, 
      an algorithm used to train neural networks. Backpropagation calculates the 
      gradients (partial derivatives of the loss function) with respect to the network's weights.
      These gradients are then used to update the weights in a way that minimizes the loss function

    In summary, the backward_propogation<E> structure is a container designed to hold gradients calculated during 
                backpropagation in a two-layer neural network used for word embeddings.  
 */
template<typename E>
struct backward_propogation 
{  
    /*
        In the first constructor, forward_propagation(),
        member variables hidden_layer_vector, predicted_probabilities, and intermediate_activation
        are initialized directly in the initialization list.
        This approach is cleaner and more efficient than assigning them inside the constructor body.
     */         
    backward_propogation() : grad_weights_input_to_hidden(Collective<E>{NULL, DIMENSIONS{0, 0, NULL, NULL}}), grad_weights_hidden_to_output(Collective<E>{NULL, DIMENSIONS{0, 0, NULL, NULL}}), grad_hidden_with_respect_to_center_word(Collective<E>{NULL, DIMENSIONS{0, 0, NULL, NULL}})
    {
        
    }

    /*
        TODO, 
        Use of Initialization Lists: Utilize constructor initialization lists to initialize
        member variables rather than assigning them inside the constructor body. This improves efficiency and readability...
        implemented but still commented out from the implementation of function.
     */
    backward_propogation(Collective<E>& grad_W1, Collective<E>& grad_W2, Collective<E>& grad_center_word) /*: grad_weights_input_to_hidden(grad_W1), grad_weights_hidden_to_output(grad_W2), grad_hidden_with_respect_to_center_word(Collective<E>{NULL, DIMENSIONS{0, 0, NULL, NULL}})*/
    {
        E* ptr = NULL;

        try 
        {                    
            ptr = cc_tokenizer::allocator<E>().allocate(grad_W1.getShape().getN());
            for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < grad_W1.getShape().getN(); i++)
            {
                ptr[i] = grad_W1[i];                
            }
        }
        catch (std::length_error& e)
        {
            throw ala_exception(cc_tokenizer::String<char>("backward_propogation() Error: ") + cc_tokenizer::String<char>(e.what())); 
        }
        catch (std::bad_alloc& e)
        {
           throw ala_exception(cc_tokenizer::String<char>("backward_propogation() Error: ") + cc_tokenizer::String<char>(e.what())); 
        }
        catch (ala_exception& e)
        {
            throw ala_exception(cc_tokenizer::String<char>("backward_propogation() Error: ") + cc_tokenizer::String<char>(e.what()));
        }
        grad_weights_input_to_hidden = Collective<E>{ptr, grad_W1.getShape().copy()};

        try 
        {                    
            ptr = cc_tokenizer::allocator<E>().allocate(grad_W2.getShape().getN());
            for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < grad_W2.getShape().getN(); i++)
            {
                ptr[i] = grad_W2[i];                
            }
        }
        catch (std::length_error& e)
        {
            throw ala_exception(cc_tokenizer::String<char>("backward_propogation() Error: ") + cc_tokenizer::String<char>(e.what())); 
        }
        catch (std::bad_alloc& e)
        {
           throw ala_exception(cc_tokenizer::String<char>("backward_propogation() Error: ") + cc_tokenizer::String<char>(e.what())); 
        }
        catch (ala_exception& e)
        {
            throw ala_exception(cc_tokenizer::String<char>("backward_propogation() Error: ") + cc_tokenizer::String<char>(e.what()));
        }
        grad_weights_hidden_to_output = Collective<E>{ptr, grad_W2.getShape().copy()};

        //grad_hidden_with_respect_to_center_word = Collective<E>{NULL, DIMENSIONS{0, 0, NULL, NULL}};

        try 
        {                    
            ptr = cc_tokenizer::allocator<E>().allocate(grad_center_word.getShape().getN());
            for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < grad_center_word.getShape().getN(); i++)
            {
                ptr[i] = grad_center_word[i];                
            }
        }
        catch (std::length_error& e)
        {
            throw ala_exception(cc_tokenizer::String<char>("backward_propogation() Error: ") + cc_tokenizer::String<char>(e.what())); 
        }
        catch (std::bad_alloc& e)
        {
           throw ala_exception(cc_tokenizer::String<char>("backward_propogation() Error: ") + cc_tokenizer::String<char>(e.what())); 
        }
        catch (ala_exception& e)
        {
            throw ala_exception(cc_tokenizer::String<char>("backward_propogation() Error: ") + cc_tokenizer::String<char>(e.what()));
        }
        grad_hidden_with_respect_to_center_word = Collective<E>{ptr, grad_hidden_with_respect_to_center_word.getShape().copy()};
    }

    backward_propogation<E>& operator= (backward_propogation<E>& other)    
    { 
        if (this == &other)
        {
            return *this;
        }

        E* ptr = NULL;

        try 
        {
            ptr = cc_tokenizer::allocator<E>().allocate(other.grad_weights_input_to_hidden.getShape().getN());
            for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < other.grad_weights_input_to_hidden.getShape().getN(); i++)
            {
                ptr[i] = other.grad_weights_input_to_hidden[i];
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
        grad_weights_input_to_hidden = Collective<E>{ptr, other.grad_weights_input_to_hidden.getShape().copy()};

        try 
        {
            ptr = cc_tokenizer::allocator<E>().allocate(other.grad_weights_hidden_to_output.getShape().getN());
            for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < other.grad_weights_hidden_to_output.getShape().getN(); i++)
            {
                ptr[i] = other.grad_weights_hidden_to_output[i];
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
        grad_weights_hidden_to_output = Collective<E>{ptr, other.grad_weights_hidden_to_output.getShape().copy()};

         try 
        {
            ptr = cc_tokenizer::allocator<E>().allocate(other.grad_hidden_with_respect_to_center_word.getShape().getN());
            for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < other.grad_hidden_with_respect_to_center_word.getShape().getN(); i++)
            {
                ptr[i] = other.grad_hidden_with_respect_to_center_word[i];
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
        grad_hidden_with_respect_to_center_word = Collective<E>{ptr, other.grad_hidden_with_respect_to_center_word.getShape().copy()};

        return *this;
    }

    /*
       Gradiant Weights Input to Hidden accessor methods
     */
    E gw1(cc_tokenizer::string_character_traits<char>::size_type i) throw (ala_exception)
    {
        if (i >= grad_weights_input_to_hidden.getShape().getN())
        {
            throw ala_exception("forward_propogation::gw1() Error: Provided index value is out of bounds.");
        }

        return grad_weights_input_to_hidden[((i/grad_weights_input_to_hidden.getShape().getNumberOfColumns())*grad_weights_input_to_hidden.getShape().getNumberOfColumns() + i%grad_weights_input_to_hidden.getShape().getNumberOfColumns())];
    }
    DIMENSIONS gw1Shape(void)
    {
        return *(grad_weights_input_to_hidden.getShape().copy());
    }

    /*        
        Gradiant Weights Hidden to Output accessor methods
     */
    E gw2(cc_tokenizer::string_character_traits<char>::size_type i) throw (ala_exception)
    {
        if (i >= grad_weights_hidden_to_output.getShape().getN())
        {
            throw ala_exception("forward_propogation::gw2() Error: Provided index value is out of bounds.");
        }

        return grad_weights_hidden_to_output[((i/grad_weights_hidden_to_output.getShape().getNumberOfColumns())*grad_weights_hidden_to_output.getShape().getNumberOfColumns() + i%grad_weights_hidden_to_output.getShape().getNumberOfColumns())];
    }
    DIMENSIONS gw2Shape(void)
    {
        return *(grad_weights_hidden_to_output.getShape().copy());
    }

     /*        
        Gradiant Hidden with respect_to Center Word accessor methods
     */
    E ghcw(cc_tokenizer::string_character_traits<char>::size_type i) throw (ala_exception)
    {
        if (i >= grad_hidden_with_respect_to_center_word.getShape().getN())
        {
            throw ala_exception("forward_propogation::ghcw() Error: Provided index value is out of bounds.");
        }

        return grad_hidden_with_respect_to_center_word[((i/grad_hidden_with_respect_to_center_word.getShape().getNumberOfColumns())*grad_hidden_with_respect_to_center_word.getShape().getNumberOfColumns() + i%grad_hidden_with_respect_to_center_word.getShape().getNumberOfColumns())];
    }
    DIMENSIONS ghcwShape(void)
    {
        return *(grad_hidden_with_respect_to_center_word.getShape().copy());
    }

    /*        
        Declare backward as a friend function within the struct. It is templated, do we need it like this.
     */    
    template <typename T>
    friend backward_propogation<T> backward(Collective<T>&, Collective<T>&, CORPUS_REF, forward_propogation<T>&, WORDPAIRS_PTR, bool) throw (ala_exception);
        
    /*
        TODO, uncomment the following statement and make all variables/properties of this vector private.
     */
    /*private:*/
        /*
            Both arrays has shape which is (corpus::len(), REPLIKA_HIDDEN_SIZE) and (REPLIKA_HIDDEN_SIZE, corpus::len()) respectovely
         */
        //E* grad_W1;
        /*
            Stores the gradients(The gradients (partial derivatives of the loss function) with respect to the network's weights)
            for the first layer weights (W1)
         */
        //Collective<E> grad_W1;
        /*
         * grad_weights_input_to_hidden: This collective object stores the gradients with respect to the weights between the input layer and the hidden layer (W1).
         * It has a shape of (corpus::len(), REPLIKA_HIDDEN_SIZE).
         */
        Collective<E> grad_weights_input_to_hidden;
        //E* grad_W2;
        /*
            Similar to grad_W1, this member stores the gradients for the second layer weights (W2)
         */
        //Collective<E> grad_W2;
        /*
         * grad_weights_hidden_to_output: This collective object stores the gradients with respect to the weights between the hidden layer and the output layer (W2).
         * It has a shape of (REPLIKA_HIDDEN_SIZE, corpus::len()).
         */
        Collective<E> grad_weights_hidden_to_output;
        /*
            Which are the gradients of the loss function with respect to the first layer weights, second layer weights, and the center word input, respectively.
            (REPLIKA_VOCABULARY_LENGTH,, SKIP_GRAM_HIDDEN_SIZE)
         */
        /*
            This member stores the gradients with respect to the center word input (likely the word used as a reference in the word embedding task)
         */
        //E* grad_h_with_respect_to_center_or_target_word;
        //Collective<E> grad_h_with_respect_to_center_or_target_word;
        /*
         * grad_hidden_with_respect_to_center_word: This collective object stores the gradients with respect to the center word input (the word used as a reference in the word embedding task).
         * It has a shape of (REPLIKA_VOCABULARY_LENGTH, SKIP_GRAM_HIDDEN_SIZE).
         */
        Collective<E> grad_hidden_with_respect_to_center_word;
};

/*
    // Use stochastic gradient descent (SGD) or a variant (e.g., Adam) to optimize the model.
 */

/*
    The softmax function is a mathematical transformation that converts a vector of real numbers 
    into a probability distribution. This ensures:
      - All output values lie between 0 and 1.
      - The sum of all output values is 1.

    In the context of CBOW (Continuous Bag of Words):
      - The input to softmax is the **predicted output scores** (logits) from the hidden layer.
      - These scores are obtained after taking the average of word embeddings from the context words 
        and multiplying them with the weight matrix W2.
      - The softmax function then converts these scores into probabilities, representing the likelihood 
        of each word in the vocabulary being the correct target word.

    Parameters:
      - a: Collective<T> 
        A vector of real-valued numbers representing the unnormalized logits (raw scores) 
        from the output layer of the CBOW model.

      - verbose: bool (optional, default = false)
        If true, prints intermediate steps (useful for debugging).

    Returns:
      - Collective<T>
        A probability distribution over the vocabulary, where each value represents the probability 
        of the corresponding word being the correct target word.

    The computation follows these steps:
      1. Subtract the maximum value from all elements for numerical stability.
      2. Apply the exponential function.
      3. Normalize by dividing each element by the sum of all exponentiated values.

    This ensures that the output probabilities do not suffer from floating-point precision issues 
    and remain numerically stable.
*/
template <typename T>
Collective<T> softmax(Collective<T>& a, bool verbose = false) throw (ala_exception)
{
    Collective<T> m; // max
    Collective<T> a_m; // a minus m 
    Collective<T> e_a_m; // exp over a_m
    Collective<T> s_e_a_m; // sum of e_a_m
    Collective<T> e_a_minus_max_divided_by_e_a_minus_max_sum;    

    try
    {
        m = Numcy::max(a); // Max value for numerical stability
        a_m = Numcy::subtract(a, m); // a - max(a)
        e_a_m = Numcy::exp(a_m); // exp(a - max(a))  
        s_e_a_m = Numcy::sum(e_a_m); // sum(exp(a - max(a)))
        /*
            m is max
            a_m, a minus m
            e_a_m, exp over a_m
            s_e_a_m, sum of e_a_m
         */
        /*
            Normalization step:
            Each element is divided by the sum of all exponentiated values 
            to ensure that the sum of the output probabilities is exactly 1.
         */
        e_a_minus_max_divided_by_e_a_minus_max_sum = Numcy::divide(e_a_m, s_e_a_m);     
    }
    catch(ala_exception& e)
    {        
        throw ala_exception(cc_tokenizer::String<char>("softmax() -> ") + cc_tokenizer::String<char>(e.what()));
    }
    
    return e_a_minus_max_divided_by_e_a_minus_max_sum;
}

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



template <typename E = cc_tokenizer::string_character_traits<char>::size_type>
cc_tokenizer::string_character_traits<char>::size_type* generateNegativeSamples_skip_gram(CORPUS_REF vocab, WORDPAIRS_PTR pair, E n = (SKIP_GRAM_WINDOW_SIZE*2 + 1)) throw (ala_exception)
{
    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<E> distrib(0, vocab.numberOfUniqueTokens() - 1);

    /*
    for (int i = 0; i < SKIP_GRAM_WINDOW_SIZE; i++)
    {        
        std::cout<< (*(pair->getRight()))[i] << ", ";
    }
    for (int i = 0; i < SKIP_GRAM_WINDOW_SIZE; i++)
    {
        std::cout<< (*(pair->getLeft()))[i] << ", ";
    }    
    std::cout<< std::endl;
     */

    E i = 0;
    E* ptr = NULL;
    try {

        ptr = cc_tokenizer::allocator<E>().allocate(SKIP_GRAM_WINDOW_SIZE*2);
    }
    catch (std::bad_alloc& e)
    {
        throw ala_exception(cc_tokenizer::String<char>("generateNegativeSamples() Error: std::bad_alloc caught. ") + cc_tokenizer::String<char>(e.what()));
    }    
    catch (std::length_error& e)
    {
        throw ala_exception(cc_tokenizer::String<char>("generateNegativeSamples() Error: std::length_error caught. ") + cc_tokenizer::String<char>(e.what()));
    }

    while (1)
    {
        E index = distrib(gen), j = 0;

        for (; j < SKIP_GRAM_WINDOW_SIZE;)
        {
            if ((*(pair->getRight()))[j] == (index + INDEX_ORIGINATES_AT_VALUE) || (*(pair->getLeft()))[j] == (index + INDEX_ORIGINATES_AT_VALUE))
            {
                break;
            }

            j++;
        }

        if (j == SKIP_GRAM_WINDOW_SIZE)
        {
            ptr[i] = index + INDEX_ORIGINATES_AT_VALUE;
            i = i + 1;
        }
        
        if (i == n)
        {
            break;
        }
    }

    return ptr;
}

template <typename E = double>
forward_propogation<E> forward(Collective<E>& W1, Collective<E>& W2, CORPUS_REF vocab, WORDPAIRS_PTR pair = NULL) throw (ala_exception)
{    
    if (pair == NULL)
    {
        throw ala_exception("forward() Error: Null pointer passed, expected a valid WORDPAIRS_PTR.");
    }

    cc_tokenizer::string_character_traits<char>::size_type* ptr = NULL;
    cc_tokenizer::string_character_traits<char>::size_type n = 0, j = 0;

    /*
        Counting Valid Context Words:
        --------------------------------
        The following loop calculates the number of context words in the pair that are not padding tokens.
        This information is used to determine the size of the context array (ptr) that stores the indices of the context words.
        The context array is used to compute the hidden layer vector (h) by averaging the embeddings of the context words

        INDEX_ORIGINATES_AT_VALUE IS a threshold to determine if a word index is valid.  
        "n" keeps track of the total number of valid context words
     */    
    for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < SKIP_GRAM_WINDOW_SIZE; i++)
    {
        if ((*(pair->getLeft()))[i] >= INDEX_ORIGINATES_AT_VALUE)
        {
            n = n + 1;      
        }
        else
        {
            // Unnecessary
        }

        if ((*(pair->getRight()))[i] >= INDEX_ORIGINATES_AT_VALUE)
        {
            n = n + 1;            
        }
        else 
        {
            // Unnecessary
        }        
    }
    
    try
    {   /*
            Allocating Memory for Context Words:
            ---------------------------------------
            Allocate memory for the context array (ptr) based on the number of valid context words (n).
            The context array stores the indices of the context words, which are used to compute the hidden layer vector (h).
            The size of the context array is determined by the number of valid context words in the pair.
         */
        ptr = cc_tokenizer::allocator<cc_tokenizer::string_character_traits<char>::size_type>().allocate(/* At most it can be SKIP_GRAM_WINDOW_SIZE*2 */ n);
        
        /*
            Storing Context Words:
            -------------------------
            The following loop populates the context array (ptr) with the indices of the valid context words from the pair.
            The indices are stored in the context array to compute the hidden layer vector (h) by averaging the embeddings of the context words.
            The loop iterates over the context words in the pair and stores their indices in the context array.
         */
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

        /*
            Creating a Collective Object for Context Words:
            --------------------------------------------------
            Create a Collective object (context) to store the context words (ptr) with the appropriate dimensions.
            The context object is used to compute the hidden layer vector (h) by averaging the embeddings of the context words.
            The Collective is a container that holds the context words and their dimensions for further processing           
         */    
        Collective<cc_tokenizer::string_character_traits<char>::size_type> context = Collective<cc_tokenizer::string_character_traits<char>::size_type>{ptr, DIMENSIONS{/*SKIP_GRAM_WINDOW_SIZE*2*/ n, 1, NULL, NULL}};

         /*
            Computing the Hidden Layer Vector (h):
            -----------------------------------------
            Computes the hidden layer vector h by averaging the embeddings of the context words.
            The hidden layer vector h is used in both the forward and backward passes of the neural network             
         */
         /*
            In the context of our CBOW, h refers to the hidden layer vector obtained by averaging the embeddings of the context words.
            It is used in both the forward and backward passes of the neural network.

            The shape of this array is (1, SKIP_GRAM_EMBEDDNG_VECTOR_SIZE), 
            a single row vector with the size of the embedding dimension.
         */
         /*
            In both Skip-gram and CBOW, the hidden layer vector (h) is computed. 
            For CBOW, h is the average of the embeddings of the context words. 
            This involves accessing the embeddings for all context words and averaging them.
        */
        Collective<E> h = Numcy::mean(W1, context);
        
        /*	
            Represents an intermediat gradient.	 
            This vector has shape (1, len(vocab)), similar to y_pred. 
            It represents the result of the dot product operation between the center or target word vector "h" and the weight matrix W2.
            The result stored in "u” captures the combined influence of hidden neurons on predicting context words. It provides a
            numerical representation of how likely each word in the vocabulary is to be a context word of a given target 
            word (within the skip-gram model).

            The variable "u" serves as an intermediary step in the forward pass, representing the activations before applying 
            the “softmax” function to generate the predicted probabilities. 

            It represents internal state in the neural network during the working of "forward pass".
            This intermediate value is used in calculations involving gradients in "backward pass" or "back propogation"(the function backward).
         */
        /*
            Both algorithms then perform a dot product between the hidden layer representation (h) and the output weight matrix (W2). 
            This step is essential to transform the hidden layer activations into the vocabulary space for prediction.
        */
        Collective<E> u = Numcy::dot(h, W2);

        /*
            y_pred is a Numcy array of predicted probabilities of the output word given the input context. 
            In our implementation, it is the output of the forward propagation step.

            The shape of this array is (1, len(vocab)), indicating a single row vector with the length of the vocabulary and 
            where each element corresponds to the predicted probability of a specific word.
         */
        /*
            The resulting vector (u) is passed through a softmax function to obtain the predicted probabilities (y_pred). 
            The softmax function converts the raw scores into probabilities.
         */
        Collective<E> y_pred = softmax<E>(u);

        return forward_propogation<E>{h, y_pred, u};
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

template <typename T = double>
backward_propogation<T> backward(Collective<T>& W1, Collective<T>& W2, CORPUS_REF vocab, forward_propogation<T>& fp, WORDPAIRS_PTR pair, bool verbose = false) throw (ala_exception)
{
    /* The hot one array is row vector, and has shape (1, vocab.len = REPLIKA_VOCABULARY_LENGTH a.k.a no redundency) */
    Collective<T> oneHot;
    /* The shape of grad_u is the same as y_pred (fp.predicted_probabilities) which is (1, len(vocab) without redundency) */
    Collective<T> grad_u;
    /*          
     */
    Collective<T> grad_u_T;
    /*
        Dimensions of grad_u is (1, len(vocab) without redundency)
        Dimensions of fp.intermediate_activation (1, len(vocab) without redundency)

        Dimensions of grad_W2 is (len(vocab) without redundency, len(vocab) without redundency)        
     */
    Collective<T> grad_W2;    
    /*
       Dimensions of grad_u is (1, len(vocab) without redundency)
       Dimensions of W2_T is (len(vocab) without redundency, SKIP_GRAM_EMBEDDNG_VECTOR_SIZE)

       Dimensions of grad_h is (1, SKIP_GRAM_EMBEDDNG_VECTOR_SIZE)
     */
    Collective<T> grad_h;
    /*
        Dimensions of grad_W1 is (len(vocab) without redundency, SKIP_GRAM_EMBEDDNG_VECTOR_SIZE)
     */
    Collective<T> grad_W1;
    
    /*
        Creating a One-Hot Vector, using Numcy::zeros with a shape of (1, vocab.numberOfUniqueTokens()).
        This creates a zero-filled column vector with a length equal to the vocabulary size
     */
    try 
    {       
        oneHot = Numcy::zeros(DIMENSIONS{vocab.numberOfUniqueTokens(), 1, NULL, NULL});
 
        oneHot[pair->getCenterWord() - INDEX_ORIGINATES_AT_VALUE] = 1;

        grad_u = Numcy::subtract<double>(fp.predicted_probabilities, oneHot);

        grad_W2 = Numcy::outer(fp.hidden_layer_vector, grad_u);
        
        grad_u_T = Numcy::transpose(grad_u);
        
        grad_h = Numcy::dot(W2, grad_u_T);
                
        grad_W1 = Numcy::zeros<T>(DIMENSIONS{SKIP_GRAM_EMBEDDNG_VECTOR_SIZE, vocab.numberOfUniqueTokens(), NULL, NULL});
        
       /*
            The following code block iterates through the context word indices (left and right) from the pair object.
            For each valid context word index (i), it adds the gradient values from transpose_outer_grad_h_context_ones to 
            the corresponding entries in grad_W1, updating the specific columns of the respective rows.
        */
        // Iterate through the left context word indices in reverse order.
        for (int i = SKIP_GRAM_WINDOW_SIZE - 1; i >= 0; i--)
        {   
            // Check if the current left context word index is within the valid range of unique tokens in the vocabulary.
            if (((*(pair->getLeft()))[i] - INDEX_ORIGINATES_AT_VALUE) < vocab.numberOfUniqueTokens())
            {
                // Iterate through the columns of the gradient matrix.
                for (cc_tokenizer::string_character_traits<char>::size_type j = 0; j < grad_W1.getShape().getNumberOfColumns(); j++)
                {
                    // Update the specific column of the specific row in grad_W1 by adding the corresponding value from transpose_outer_grad_h_context_ones.
                    grad_W1[((*(pair->getLeft()))[i] - INDEX_ORIGINATES_AT_VALUE)*grad_W1.getShape().getNumberOfColumns() + j] += (grad_h[j] / SKIP_GRAM_WINDOW_SIZE);
                }
            }
        }
        // Iterate through the right context word indices in order.
        for (int i = 0; i < SKIP_GRAM_WINDOW_SIZE; i++)
        {
            // Check if the current right context word index is within the valid range of unique tokens in the vocabulary.
            if (((*(pair->getRight()))[i] - INDEX_ORIGINATES_AT_VALUE) < vocab.numberOfUniqueTokens())
            {
                // Iterate through the columns of the gradient matrix.
                for (cc_tokenizer::string_character_traits<char>::size_type j = 0; j < grad_W1.getShape().getNumberOfColumns(); j++)
                {
                    // Update the specific column of the specific row in grad_W1 by adding the corresponding value from transpose_outer_grad_h_context_ones.
                    grad_W1[((*(pair->getRight()))[i] - INDEX_ORIGINATES_AT_VALUE)*grad_W1.getShape().getNumberOfColumns() + j] += (grad_h[j] / SKIP_GRAM_WINDOW_SIZE);
                }
            } 
        }
    }
    catch (ala_exception& e)
    {
        throw ala_exception(cc_tokenizer::String<char>("backward() Error: ") + cc_tokenizer::String<char>(e.what()));
    }

    /*
        Dimensions of grad_W1 is (len(vocab) without redundency, SKIP_GRAM_EMBEDDNG_VECTOR_SIZE)
        Dimensions of grad_W2 is (len(vocab) without redundency, len(vocab) without redundency)
     */ 
    return backward_propogation<T>{grad_W1, grad_W2, Collective<T>{NULL, DIMENSIONS{0, 0, NULL, NULL}}};
}

/**
 * @param el: (Type: float or double)
 *      A variable to accumulate and store the error or loss for each epoch. This tracks the model's performance.
 *
 * @param epoch: (Type: size_t or int)
 *      The total number of epochs (iterations) for which the training loop will run. Each epoch processes the entire dataset once.
 *
 * @param lr: (Type: float or double)
 *      The learning rate for the model. This controls the step size during weight updates and determines how much the model should adjust its weights at each step.
 *
 * @param pairs: (Type: PAIRS or similar data structure)
 *      The training data, which contains word pairs. Each pair consists of a center word and its surrounding context words. The model learns to predict the center word from its context.
 *
 * @param t: (Template Type)
 *      A generic template type for numerical operations. This allows the macro to work with different numerical types (e.g., float, double) depending on precision requirements.
 *
 * @param verbose: (Type: bool)
 *      If true, prints detailed logs of the training process, such as the current epoch number and other information for debugging or tracking purposes.
 *
 * @param vocab: (Type: VOCAB or similar data structure)
 *      The vocabulary data structure that stores all the unique words used in training. It is required to index and retrieve word embeddings during forward and backward propagation.
 *
 * @param W1: (Type: Collective<t>)
 *      The input-to-hidden weight matrix. It holds the word embeddings for the context words and is updated during backpropagation.
 *
 * @param W2: (Type: Collective<t>)
 *      The hidden-to-output weight matrix. This matrix is used to predict the center word from the context word embeddings and is also updated during training.
 */
#define CBOW_TRAINING_LOOP(el, epoch, lr, pairs, t, verbose, vocab, W1, W2)\
{\
    /* Epoch loop */\
    for (cc_tokenizer::string_character_traits<char>::size_type i = 1; i <= epoch; i++)\
    {\
        if (verbose)\
        {\
            std::cout<< "Epoch# " << i << " of " << epoch << " epochs." << std::endl;\
        }\
        /* Shuffle Word Pairs: Shuffles the training data (word pairs) before each epoch to avoid biases in weight updates */\
        Numcy::Random::shuffle<PAIRS>(pairs, pairs.get_number_of_word_pairs());\
        /* Iterates through each word pair in the training data  */\
        while (pairs.go_to_next_word_pair() != cc_tokenizer::string_character_traits<char>::eof())\
        {\
            /* Get Current Word Pair: We've a pair, a pair is LEFT_CONTEXT_WORD/S CENTER_WORD and RIGHT_CONTEXT_WORD/S */\
            WORDPAIRS_PTR pair = pairs.get_current_word_pair();\
            try\
            {\
                forward_propogation<t> fp = forward (W1, W2, vocab, pair);\
                backward_propogation<t> bp = backward (W1, W2, vocab, fp, pair);\
                /* Update weights */\
                W1 -= bp.grad_weights_input_to_hidden * lr;\
                W2 -= bp.grad_weights_hidden_to_output * lr;\
                /* Loss Function: The CBOW model typically uses negative log-likelihood (NLL) as the loss function.\
                   In NLL, lower values indicate better performance. */\
                el = el + (-1*log(fp.pb(pair->getCenterWord() - INDEX_ORIGINATES_AT_VALUE)));\
            }\
            catch (ala_exception& e)\
            {\
                std::cout<< "CBOW_TRAINING_LOOP() -> " << e.what() << std::endl;\
            }\
        }\
        std::cout<< "epoch_loss = " << el/pairs.get_number_of_word_pairs() << std::endl;\
        el = 0;\
    }\
}\

#endif
