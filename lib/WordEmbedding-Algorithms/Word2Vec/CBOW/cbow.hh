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
    D:\KHAAdotPK\CBOW\lib\WordEmbedding-Algorithms\Word2Vec\CBOW\cbow.hh(17): note: while compiling class template 'forward_propagation'
 */
template<typename E>
struct backward_propagation; 

template<typename E>
struct forward_propagation 
{
    /*
        In the first constructor, forward_propagation(),
        member variables hidden_layer_vector, predicted_probabilities, and intermediate_activation
        are initialized directly in the initialization list.
        This approach is cleaner and more efficient than assigning them inside the constructor body.
     */
    forward_propagation(void) : hidden_layer_vector(Collective<E>{NULL, DIMENSIONS{0, 0, NULL, NULL}}), predicted_probabilities(Collective<E>{NULL, DIMENSIONS{0, 0, NULL, NULL}}), intermediate_activation(Collective<E>{NULL, DIMENSIONS{0, 0, NULL, NULL}}),/* negative_hidden_layer_vector(Collective<E>{NULL, DIMENSIONS{0, 0, NULL, NULL}}),*/ negative_predicted_probabilities(Collective<E>{NULL, DIMENSIONS{0, 0, NULL, NULL}}), negative_intermediate_activation(Collective<E>{NULL, DIMENSIONS{0, 0, NULL, NULL}})
    {        
    }

    /*
        TODO, 
        Use of Initialization Lists: Utilize constructor initialization lists to initialize
        member variables rather than assigning them inside the constructor body. This improves efficiency and readability...
        implemented but still commented out from the implementation of function.
     */
    //forward_propagation<E>(Collective<E>& h, Collective<E>& y_pred, Collective<E>& u) : hidden_layer_vector(h), predicted_probabilities(y_pred), intermediate_activation(u)
    forward_propagation<E>(Collective<E>& h, Collective<E>& y_pred, Collective<E>& u, /*Collective<E>& h_negative,*/ Collective<E>& y_pred_negative, Collective<E>& u_negative) throw (ala_exception) /*: hidden_layer_vector(h), predicted_probabilities(y_pred), intermediate_activation(u) */
    { 
        try
        {
            this->hidden_layer_vector = h;
            this->predicted_probabilities = y_pred;
            this->intermediate_activation = u;

            /*this->negative_hidden_layer_vector = h_negative;*/        
            this->negative_predicted_probabilities = y_pred_negative;
            this->negative_intermediate_activation = u_negative;
        }
        catch(ala_exception& e)
        {
            // Propagate existing ala_exception with additional context
            // NO cleanup performed assuming this is also a critical error
            throw ala_exception(cc_tokenizer::String<char>("forward_propagaton<E>::forward_propagation(Collective<E>&, Collective<E>&, Collective<E>&, Collective<E>&, Collective<E>&, Collective<E>&) -> ") + e.what());
        }
        
        
        //E* ptr = NULL;

        /*try 
        {                    
            ptr = cc_tokenizer::allocator<E>().allocate(h.getShape().getN());
            for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < h.getShape().getN(); i++)
            {
                ptr[i] = h[i];                
            }
        }
        catch (std::length_error& e)
        {
            throw ala_exception(cc_tokenizer::String<char>("forward_propagation() Error: ") + cc_tokenizer::String<char>(e.what())); 
        }
        catch (std::bad_alloc& e)
        {
           throw ala_exception(cc_tokenizer::String<char>("forward_propagation() Error: ") + cc_tokenizer::String<char>(e.what())); 
        }
        catch (ala_exception& e)
        {
            throw ala_exception(cc_tokenizer::String<char>("forward_propagation() Error: ") + cc_tokenizer::String<char>(e.what()));
        }*/
        //hidden_layer_vector = Collective<E>{ptr, h.getShape()/*.copy()*/};
        /*this->hidden_layer_vector = h;*/

        /*try
        {                 
            ptr = cc_tokenizer::allocator<E>().allocate(y_pred.getShape().getN());
            for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < y_pred.getShape().getN(); i++)
            {
                ptr[i] = y_pred[i];
            }
        } 
        catch (std::length_error& e)
        {
            throw ala_exception(cc_tokenizer::String<char>("forward_propagation() Error: ") + cc_tokenizer::String<char>(e.what())); 
        }
        catch (std::bad_alloc& e)
        {
            throw ala_exception(cc_tokenizer::String<char>("forward_propagation() Error: ") + cc_tokenizer::String<char>(e.what())); 
        }      
        catch (ala_exception& e)
        {
            throw ala_exception(cc_tokenizer::String<char>("forward_propagation() Error: ") + cc_tokenizer::String<char>(e.what()));
        }*/
        //predicted_probabilities = Collective<E>{ptr, y_pred.getShape()/*.copy()*/};
        /*this->predicted_probabilities = y_pred;*/

        /*try
        {        
            ptr = cc_tokenizer::allocator<E>().allocate(u.getShape().getN());
            for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < u.getShape().getN(); i++)
            {
                ptr[i] = u[i];
            }
        }
        catch (std::length_error& e)
        {
            throw ala_exception(cc_tokenizer::String<char>("forward_propagation() Error: ") + cc_tokenizer::String<char>(e.what())); 
        }
        catch (std::bad_alloc& e)
        {
            throw ala_exception(cc_tokenizer::String<char>("forward_propagation() Error: ") + cc_tokenizer::String<char>(e.what())); 
        }      
        catch (ala_exception& e)
        {
            throw ala_exception(cc_tokenizer::String<char>("forward_propagation() Error: ") + cc_tokenizer::String<char>(e.what()));
        }*/
        //intermediate_activation = Collective<E>{ptr, u.getShape()/*.copy()*/};
        /*this->intermediate_activation = u;*/

        /*this->negative_hidden_layer_vector = h_negative;        
        this->negative_predicted_probabilities = negative_predicted_probabilities;
        this->negative_intermediate_activation = negative_intermediate_activation;*/


        /*std::cout<< "Get Reference Count = " << this->negative_hidden_layer_vector.getReferenceCount() << std::endl;*/
        
        /*try 
        {                    
            ptr = cc_tokenizer::allocator<E>().allocate(h_negative.getShape().getN());
            for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < h_negative.getShape().getN(); i++)
            {
                ptr[i] = h_negative[i];                
            }
        }
        catch (std::length_error& e)
        {
            throw ala_exception(cc_tokenizer::String<char>("forward_propagation() Error: ") + cc_tokenizer::String<char>(e.what())); 
        }
        catch (std::bad_alloc& e)
        {
           throw ala_exception(cc_tokenizer::String<char>("forward_propagation() Error: ") + cc_tokenizer::String<char>(e.what())); 
        }
        catch (ala_exception& e)
        {
            throw ala_exception(cc_tokenizer::String<char>("forward_propagation() Error: ") + cc_tokenizer::String<char>(e.what()));
        }*/
        //negative_hidden_layer_vector = Collective<E>{ptr, h_negative.getShape()/*.copy()*/};
    }

    forward_propagation<E>(forward_propagation<E>& other) throw (ala_exception)
    {          
        E* ptr = NULL;

        try
        {        
            ptr = cc_tokenizer::allocator<E>().allocate(other.hidden_layer_vector.getShape().getN());
            for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < other.hidden_layer_vector.getShape().getN(); i++)
            {
                ptr[i] = other.hidden_layer_vector[i];
            }
            hidden_layer_vector = Collective<E>{ptr, other.hidden_layer_vector.getShape()/*.copy()*/};

            ptr = cc_tokenizer::allocator<E>().allocate(other.predicted_probabilities.getShape().getN());
            for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < other.predicted_probabilities.getShape().getN(); i++)
            {
                ptr[i] = other.predicted_probabilities[i];
            }
            predicted_probabilities = Collective<E>{ptr, other.predicted_probabilities.getShape()/*.copy()*/};

            ptr = cc_tokenizer::allocator<E>().allocate(other.intermediate_activation.getShape().getN());
            for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < other.intermediate_activation.getShape().getN(); i++)
            {
                ptr[i] = other.intermediate_activation[i];
            }
            intermediate_activation = Collective<E>{ptr, other.intermediate_activation.getShape()/*.copy()*/};

            /*ptr = cc_tokenizer::allocator<E>().allocate(other.negative_hidden_layer_vector.getShape().getN());
            for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < other.negative_hidden_layer_vector.getShape().getN(); i++)
            {
                ptr[i] = other.negative_hidden_layer_vector[i];
            }*/
            //negative_hidden_layer_vector = Collective<E>{ptr, other.negative_hidden_layer_vector.getShape()/*.copy()*/};

            ptr = cc_tokenizer::allocator<E>().allocate(other.negative_predicted_probabilities.getShape().getN());
            for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < other.negative_predicted_probabilities.getShape().getN(); i++)
            {
                ptr[i] = other.negative_predicted_probabilities[i];
            }
            negative_predicted_probabilities = Collective<E>{ptr, other.negative_predicted_probabilities.getShape()/*.copy()*/};

            ptr = cc_tokenizer::allocator<E>().allocate(other.negative_intermediate_activation.getShape().getN());
            for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < other.negative_intermediate_activation.getShape().getN(); i++)
            {
                ptr[i] = other.negative_intermediate_activation[i];
            }
            negative_intermediate_activation = Collective<E>{ptr, other.negative_intermediate_activation.getShape()/*.copy()*/};
        }
        catch (std::bad_alloc& e)
        {
            // CRITICAL: Memory allocation failure - system should terminate immediately
            // NO cleanup performed - this is a fatal error requiring process exit
            throw ala_exception(cc_tokenizer::String<char>("forward_propagation<E>::forward_propagation(Collective<E>&) Error: ") + cc_tokenizer::String<char>(e.what())); 
        }
        catch (std::length_error& e)
        {
            // CRITICAL: Length constraint violation - system should terminate immediately
            // NO cleanup performed - this is a fatal error requiring process exit
            throw ala_exception(cc_tokenizer::String<char>("forward_propagation<E>::forward_propagation(Collective<E>&) Error: ") + cc_tokenizer::String<char>(e.what())); 
        } 
        catch (ala_exception& e)
        {
            // Propagate existing ala_exception with additional context
            // NO cleanup performed assuming this is also a critical error
            throw ala_exception(cc_tokenizer::String<char>("forward_propagation<E>::forward_propagation(Collective<E>&) -> ") + cc_tokenizer::String<char>(e.what())); 
        }
    }

    forward_propagation<E>& operator= (forward_propagation<E>& other)
    {
        // Self assignment check
        if (this == &other)
        {
            return *this;            
        }
        
        try
        {
            this->hidden_layer_vector = other.hidden_layer_vector;
            this->predicted_probabilities = other.predicted_probabilities;
            this->intermediate_activation = other.intermediate_activation;

            //this->negative_hidden_layer_vector = other.negative_hidden_layer_vector;        
            this->negative_predicted_probabilities = other.negative_predicted_probabilities;
            this->negative_intermediate_activation = other.negative_intermediate_activation;

            return *this;
        }
        catch(ala_exception& e)
        {
            // Propagate existing ala_exception with additional context
            // NO cleanup performed assuming this is also a critical error
            throw ala_exception(cc_tokenizer::String<char>("forward_propagaton<E>::operator= (Collective<E>&) -> ") + e.what());
        }
    }

    /*forward_propagation<E>& operator= (forward_propagation<E>& other)    
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
            throw ala_exception(cc_tokenizer::String<char>("forward_propagation<E>::operator=() Error: ") + cc_tokenizer::String<char>(e.what())); 
        }
        catch (std::bad_alloc& e)
        {
           throw ala_exception(cc_tokenizer::String<char>("forward_propagation<E>::operator=() Error: ") + cc_tokenizer::String<char>(e.what())); 
        }
        catch (ala_exception& e)
        {
            throw ala_exception(cc_tokenizer::String<char>("forward_propagation<E>::operator=() -> ") + cc_tokenizer::String<char>(e.what()));
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
            throw ala_exception(cc_tokenizer::String<char>("forward_propagation::operator=() Error: ") + cc_tokenizer::String<char>(e.what())); 
        }
        catch (std::bad_alloc& e)
        {
           throw ala_exception(cc_tokenizer::String<char>("forward_propagation::operator=() Error: ") + cc_tokenizer::String<char>(e.what())); 
        }
        catch (ala_exception& e)
        {
            throw ala_exception(cc_tokenizer::String<char>("forward_propagation::operator=() Error: ") + cc_tokenizer::String<char>(e.what()));
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
            throw ala_exception(cc_tokenizer::String<char>("forward_propagation::operator=() Error: ") + cc_tokenizer::String<char>(e.what())); 
        }
        catch (std::bad_alloc& e)
        {
           throw ala_exception(cc_tokenizer::String<char>("forward_propagation::operator=() Error: ") + cc_tokenizer::String<char>(e.what())); 
        }
        catch (ala_exception& e)
        {
            throw ala_exception(cc_tokenizer::String<char>("forward_propagation::operator=() Error: ") + cc_tokenizer::String<char>(e.what()));
        }
        intermediate_activation = Collective<E>{ptr, other.intermediate_activation.getShape().copy()};
        
        return *this;
    }*/
    
    /*
        Hidden Layer Vector accessor methods
     */
    E hlv(cc_tokenizer::string_character_traits<char>::size_type i) throw (ala_exception)
    {
        if (i >= hidden_layer_vector.getShape().getN())
        {
            throw ala_exception("forward_propagation::hlv() Error: Provided index value is out of bounds.");
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
            throw ala_exception("forward_propagation::pb() Error: Provided index value is out of bounds.");
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
            throw ala_exception("forward_propagation::ia() Error: Provided index value is out of bounds.");
        }

        return intermediate_activation[((i/intermediate_activation.getShape().getNumberOfColumns())*intermediate_activation.getShape().getNumberOfColumns() + i%intermediate_activation.getShape().getNumberOfColumns())];
    }
    DIMENSIONS iaShape(void)
    {
        return *(intermediate_activation.getShape().copy());
    }

    /*
        Negative Predicted Probabilities accesssor methods
     */
    E npp(cc_tokenizer::string_character_traits<char>::size_type i) throw (ala_exception)
    {
        if (i >= negative_predicted_probabilities.getShape().getN())
        {
            // NO cleanup performed assuming this is also a critical error
            throw ala_exception("forward_propagation::npp(cc_tokenizer::string_character_traits<char>::size_type) Error: Provided index value is out of bounds.");
        }

        return negative_predicted_probabilities[((i/negative_predicted_probabilities.getShape().getNumberOfColumns())*negative_predicted_probabilities.getShape().getNumberOfColumns() + i%negative_predicted_probabilities.getShape().getNumberOfColumns())];
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
    friend backward_propagation<T> backward(Collective<T>&, Collective<T>&, CORPUS_REF, forward_propagation<T>&, WORDPAIRS_PTR, bool) throw (ala_exception);
        
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
            This intermediate value is used in calculations involving gradients in "backward pass" or "back propagation"(the function backward).
         */
        //E* u;
        Collective<E> intermediate_activation; 

        /*
            In the context of our CBOW/Skip-Gram model, h_negative refers to the hidden layer vector obtained by averaging the embeddings of the negative_context words.
            It is used in both the forward and backward passes of the neural network.

            The shape of this array is (1, SKIP_GRAM_EMBEDDNG_VECTOR_SIZE), a single row vector with the size of the embedding dimension.
         */
        /*//E* h_negative
        Collective<E> negative_hidden_layer_vector;*/

        /*
            Predicted probabilities for the negative samples, i.e., words that are not related to the context or target word.
        
            Shape: (N, len(vocab)), where N is the number of negative samples and each element represents the probability of
            a specific word being incorrectly predicted as context.
         */
        //E* y_pred_negative
        Collective<E> negative_predicted_probabilities;

        /*	
            Intermediate activations from the dot product of the hidden layer and the weight matrix for the negative samples.
            This vector represents the influence of hidden neurons on predicting negative samples.
        
            Shape: (N, len(vocab)), where N is the number of negative samples.
         */
        //E* u_negative;
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

    In summary, the backward_propagation<E> structure is a container designed to hold gradients calculated during 
                backpropagation in a two-layer neural network used for word embeddings.  
 */
template<typename E>
struct backward_propagation 
{  
    /*
        In the first constructor, forward_propagation(),
        member variables hidden_layer_vector, predicted_probabilities, and intermediate_activation
        are initialized directly in the initialization list.
        This approach is cleaner and more efficient than assigning them inside the constructor body.
     */         
    backward_propagation() : grad_weights_input_to_hidden(Collective<E>{NULL, DIMENSIONS{0, 0, NULL, NULL}}), grad_weights_hidden_to_output(Collective<E>{NULL, DIMENSIONS{0, 0, NULL, NULL}}), grad_hidden_with_respect_to_center_word(Collective<E>{NULL, DIMENSIONS{0, 0, NULL, NULL}})
    {
        
    }

    backward_propagation(Collective<E>& grad_W1, Collective<E>& grad_W2) /*: grad_weights_input_to_hidden(grad_W1), grad_weights_hidden_to_output(grad_W2), grad_hidden_with_respect_to_center_word(Collective<E>{NULL, DIMENSIONS{0, 0, NULL, NULL}})*/
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
            throw ala_exception(cc_tokenizer::String<char>("backward_propagation() Error: ") + cc_tokenizer::String<char>(e.what())); 
        }
        catch (std::bad_alloc& e)
        {
           throw ala_exception(cc_tokenizer::String<char>("backward_propagation() Error: ") + cc_tokenizer::String<char>(e.what())); 
        }
        catch (ala_exception& e)
        {
            throw ala_exception(cc_tokenizer::String<char>("backward_propagation() Error: ") + cc_tokenizer::String<char>(e.what()));
        }
        grad_weights_input_to_hidden = Collective<E>{ptr, grad_W1.getShape()/*.copy()*/};

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
            throw ala_exception(cc_tokenizer::String<char>("backward_propagation() Error: ") + cc_tokenizer::String<char>(e.what())); 
        }
        catch (std::bad_alloc& e)
        {
           throw ala_exception(cc_tokenizer::String<char>("backward_propagation() Error: ") + cc_tokenizer::String<char>(e.what())); 
        }
        catch (ala_exception& e)
        {
            throw ala_exception(cc_tokenizer::String<char>("backward_propagation() Error: ") + cc_tokenizer::String<char>(e.what()));
        }
        grad_weights_hidden_to_output = Collective<E>{ptr, grad_W2.getShape()/*.copy()*/};

        //grad_hidden_with_respect_to_center_word = Collective<E>{NULL, DIMENSIONS{0, 0, NULL, NULL}};

        /*try 
        {                    
            ptr = cc_tokenizer::allocator<E>().allocate(grad_center_word.getShape().getN());
            
            for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < grad_center_word.getShape().getN(); i++)
            {
                ptr[i] = grad_center_word[i];                
            }
        }
        catch (std::length_error& e)
        {
            throw ala_exception(cc_tokenizer::String<char>("backward_propagation() Error: ") + cc_tokenizer::String<char>(e.what())); 
        }
        catch (std::bad_alloc& e)
        {
           throw ala_exception(cc_tokenizer::String<char>("backward_propagation() Error: ") + cc_tokenizer::String<char>(e.what())); 
        }
        catch (ala_exception& e)
        {
            throw ala_exception(cc_tokenizer::String<char>("backward_propagation() Error: ") + cc_tokenizer::String<char>(e.what()));
        }
        std::cout<< "SONI-> problem area..." << grad_center_word.getShape().getN() << std::endl;
        if (grad_center_word.getShape().getN() > 0)
        {
            std::cout<< "----------------------------------------->>>>>>>> SONI-> problem area..." << grad_center_word.getShape().getN() << std::endl;*/
            //grad_hidden_with_respect_to_center_word = Collective<E>{ptr, grad_center_word.getShape() /*grad_hidden_with_respect_to_center_word.getShape()*//*.copy()*/};
        //}
    }

    /*
        TODO, 
        Use of Initialization Lists: Utilize constructor initialization lists to initialize
        member variables rather than assigning them inside the constructor body. This improves efficiency and readability...
        implemented but still commented out from the implementation of function.
     */
    backward_propagation(Collective<E>& grad_W1, Collective<E>& grad_W2, Collective<E>& grad_center_word) /*: grad_weights_input_to_hidden(grad_W1), grad_weights_hidden_to_output(grad_W2), grad_hidden_with_respect_to_center_word(Collective<E>{NULL, DIMENSIONS{0, 0, NULL, NULL}})*/
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
            throw ala_exception(cc_tokenizer::String<char>("backward_propagation() Error: ") + cc_tokenizer::String<char>(e.what())); 
        }
        catch (std::bad_alloc& e)
        {
           throw ala_exception(cc_tokenizer::String<char>("backward_propagation() Error: ") + cc_tokenizer::String<char>(e.what())); 
        }
        catch (ala_exception& e)
        {
            throw ala_exception(cc_tokenizer::String<char>("backward_propagation() Error: ") + cc_tokenizer::String<char>(e.what()));
        }
        grad_weights_input_to_hidden = Collective<E>{ptr, grad_W1.getShape()/*.copy()*/};

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
            throw ala_exception(cc_tokenizer::String<char>("backward_propagation() Error: ") + cc_tokenizer::String<char>(e.what())); 
        }
        catch (std::bad_alloc& e)
        {
           throw ala_exception(cc_tokenizer::String<char>("backward_propagation() Error: ") + cc_tokenizer::String<char>(e.what())); 
        }
        catch (ala_exception& e)
        {
            throw ala_exception(cc_tokenizer::String<char>("backward_propagation() Error: ") + cc_tokenizer::String<char>(e.what()));
        }
        grad_weights_hidden_to_output = Collective<E>{ptr, grad_W2.getShape()/*.copy()*/};

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
            throw ala_exception(cc_tokenizer::String<char>("backward_propagation() Error: ") + cc_tokenizer::String<char>(e.what())); 
        }
        catch (std::bad_alloc& e)
        {
           throw ala_exception(cc_tokenizer::String<char>("backward_propagation() Error: ") + cc_tokenizer::String<char>(e.what())); 
        }
        catch (ala_exception& e)
        {
            throw ala_exception(cc_tokenizer::String<char>("backward_propagation() Error: ") + cc_tokenizer::String<char>(e.what()));
        }
        //std::cout<< "SONI-> problem area..." << grad_center_word.getShape().getN() << std::endl;
        //if (grad_center_word.getShape().getN() > 0)
        //{
            //std::cout<< "----------------------------------------->>>>>>>> SONI-> problem area..." << grad_center_word.getShape().getN() << std::endl;
        grad_hidden_with_respect_to_center_word = Collective<E>{ptr, grad_center_word.getShape() /*grad_hidden_with_respect_to_center_word.getShape()*//*.copy()*/};
        //}
    }

    backward_propagation(const backward_propagation<E>& other) throw (ala_exception)
    {
        this->grad_hidden_with_respect_to_center_word = other.grad_hidden_with_respect_to_center_word;   
        this->grad_weights_hidden_to_output = other.grad_weights_hidden_to_output;
        this->grad_weights_input_to_hidden = other.grad_weights_input_to_hidden;
    }
    
    backward_propagation<E>& operator= (backward_propagation<E>& other)
    {
        // Self assignment check
        if (this == &other)
        {
            return *this;            
        }
        
        try
        {
            this->grad_weights_input_to_hidden = other.grad_weights_input_to_hidden;
            this->grad_weights_hidden_to_output = other.grad_weights_hidden_to_output;
            this->grad_hidden_with_respect_to_center_word = grad_hidden_with_respect_to_center_word;

            return *this;
        }
        catch(ala_exception& e)
        {
            // Propagate existing ala_exception with additional context
            // NO cleanup performed assuming this is also a critical error
            throw ala_exception(cc_tokenizer::String<char>("backward_propagaton<E>::operator= (Collective<E>&) -> ") + e.what());
        }
    }

    /*
       Gradiant Weights Input to Hidden accessor methods
     */
    E gw1(cc_tokenizer::string_character_traits<char>::size_type i) throw (ala_exception)
    {
        if (i >= grad_weights_input_to_hidden.getShape().getN())
        {
            throw ala_exception("forward_propagation::gw1() Error: Provided index value is out of bounds.");
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
            throw ala_exception("forward_propagation::gw2() Error: Provided index value is out of bounds.");
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
            throw ala_exception("forward_propagation::ghcw() Error: Provided index value is out of bounds.");
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
    friend backward_propagation<T> backward(Collective<T>&, Collective<T>&, CORPUS_REF, forward_propagation<T>&, WORDPAIRS_PTR, bool) throw (ala_exception);
        
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

    return Collective<E> {ptr, DIMENSIONS{n, 1, NULL, NULL}};    
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
forward_propagation<E> forward(Collective<E>& W1, Collective<E>& W2, Collective<cc_tokenizer::string_character_traits<char>::size_type>& negative_context, CORPUS_REF vocab, WORDPAIRS_PTR pair = NULL) throw (ala_exception)
{
    if (pair == NULL)
    {
        throw ala_exception("forward(Collective<E>&, Collective<E>&, Collective<cc_tokenizer::string_character_traits<char>::size_type>&, CORPUS_REF, WORDPAIRS_PTR) Error: Null pointer passed, expected a valid WORDPAIRS_PTR. Required for context word processing.");
    }

    cc_tokenizer::string_character_traits<char>::size_type* ptr = NULL;
    cc_tokenizer::string_character_traits<char>::size_type n = 0, j = 0;

    /*
     * Counts valid context words by scanning both left and right context windows.
     * Valid words are those with indices >= INDEX_ORIGINATES_AT_VALUE, indicating 
     * they are actual vocabulary entries rather than padding tokens.
     * 
     * The count 'n' determines the size of the context array used for computing
     * the hidden layer vector through embedding averaging.
     */
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

 #ifdef CBOW_DEBUG_PAIR
    for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < SKIP_GRAM_WINDOW_SIZE; i++)
    {
        if ((*(pair->getLeft()))[(SKIP_GRAM_WINDOW_SIZE - 1) - i] >= INDEX_ORIGINATES_AT_VALUE)
        {
            std::cout<< vocab[(*(pair->getLeft()))[/*(SKIP_GRAM_WINDOW_SIZE - 1) -*/ i]].c_str() << " ";
        }
        else 
        {
            std::cout<< "NONE ";
        }
    }
    
    std::cout<< "[ "<< vocab[pair->getCenterWord()].c_str() << " ] ";

    for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < SKIP_GRAM_WINDOW_SIZE; i++)
    {
        if ((*(pair->getRight()))[i] >= INDEX_ORIGINATES_AT_VALUE)
        {
            std::cout<< vocab[(*(pair->getRight()))[i]].c_str() << " ";
        }
        else
        {
            std::cout<< "NONE ";
        }
    }
    std::cout<< std::endl;
 #endif   
        
    try
    {
        /*
         * Allocates memory for context word indices.
         * The array stores indices of valid context words after adjusting for 
         * INDEX_ORIGINATES_AT_VALUE offset. Maximum possible size is 2 * SKIP_GRAM_WINDOW_SIZE
         * (left + right contexts).
         */
        /*
            Allocating Memory for Context Words:
            ---------------------------------------
            Allocate memory for the context array (ptr) based on the number of valid context words (n).
            The context array stores the indices of the context words, which are used to compute the hidden layer vector (h).
            The size of the context array is determined by the number of valid context words in the pair.
         */
        ptr = cc_tokenizer::allocator<cc_tokenizer::string_character_traits<char>::size_type>().allocate(/* At most it will be SKIP_GRAM_WINDOW_SIZE*2 */ n);

        /*
         * Populates context array with adjusted indices of valid context words.
         * Index adjustment: subtracts INDEX_ORIGINATES_AT_VALUE to convert from 
         * corpus indices to embedding matrix indices.
         */
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
         * Creates Collective container for context indices to enable batch operations.
         * Dimensions: {n, 1} where n is the count of valid context words.
         */
        /*
            Creating a Collective Object for Context Words:
            --------------------------------------------------
            Create a Collective object (context) to store the context words (ptr) with the appropriate dimensions.
            The context object is used to compute the hidden layer vector (h) by averaging the embeddings of the context words.
            The Collective is a container that holds the context words and their dimensions for further processing           
         */    
        Collective<cc_tokenizer::string_character_traits<char>::size_type> context = Collective<cc_tokenizer::string_character_traits<char>::size_type>{ptr, DIMENSIONS{n, 1, NULL, NULL}};

        /*
         * Computes hidden layer vector (h) by averaging context word embeddings.
         * For CBOW: h = mean(embeddings of all context words)
         * Shape: (1, EMBEDDING_DIM) - single row vector representing the combined context
         */
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
        /*
         *  In a single CBOW training step, the context words are fixed.
         *  Whether you are checking the real "positive" center word or a fake "negative" one, the context does not change.
         *  h is calculated from the true context words only once per new pair. This single h vector is then used to evaluate the positive word Aand all the negative words.
         *  Shape: (1, EMBEDDING_DIM) or (1, SKIP_GRAM_EMBEDDNG_VECTOR_SIZE) - single row vector representing the combined context. 
         *  From the point of view of Negative Sampling...
         *  The main goal of negative sampling is to avoid calculations involving the entire W2 matrix, and to achieve that...
         *  The correct approach is to perform dot products only between single hidden vector h and the specific vectors for the chosen words (1 positive + k negative)
         */
        Collective<E> h = Numcy::mean(W1, context);

        Collective<E> W2_positive; // Embedding vector for the positive (center) word. Vocabulary vector of one positive sample       
        Collective<E> u_positive; // Vector of scores. Raw score before activation
        Collective<E> y_pred_positive; // Vector of probabilities. Probability after activation 

        // The core idea is moving from one complex question to many simple one, one positive sample and k negative samples
        /*
         * Two operational modes:
         * 1. With negative sampling: Treats as binary classification (positive vs negative samples)
         * 2. Without negative sampling: Uses softmax over entire vocabulary
         */
        if (negative_context.getShape().getN()) // Negative sampling mode - binary classification
        {            
            /*
             *   he Dot Product Is with Specific Word Vectors, Not All of W2.
             *   The main goal of negative sampling is to avoid calculations involving the entire W2 matrix                
             */
            W2_positive = W2.slice(pair->getCenterWord() - INDEX_ORIGINATES_AT_VALUE, DIMENSIONS{1, W2.getShape().getNumberOfRows(), NULL, NULL}, AXIS_ROWS); // Part of vocabulary
            u_positive = Numcy::dot(h, W2_positive); // Scores of series of simple, independent "Yes/No" questions
            // Now, apply the sigmoid function to the single positive score
            y_pred_positive = Numcy::sigmoid<E>(u_positive); // Classify which 
        }
        else
        {
            // Full softmax mode - multi-class classification over vocabulary
            u_positive = Numcy::dot(h, W2); // A huge vector of scores, one for each word in the vocabulary.
            y_pred_positive = /*Numcy::sigmoid<E>(u_positive)*/ Numcy::softmax(u_positive); // Take whole of the vocabulary and find the center/target word of the given context
            // Say your vocabulary is 10,000 words, then the result is a vector where probabilities[i] gives you the chance that word i is the correct one (the target/center word), and all 10,000 probabilities sum to 1.
        }

        
        /*if (negative_context.getShape().getN())
        {                
            W2_positive = W2.slice(pair->getCenterWord() - INDEX_ORIGINATES_AT_VALUE, DIMENSIONS{1, W2.getShape().getNumberOfRows(), NULL, NULL}, AXIS_ROWS);
        }*/
                
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
            This intermediate value is used in calculations involving gradients in "backward pass" or "back propagation"(the function backward).
         */
        /*
            Both algorithms then perform a dot product between the hidden layer representation (h) and the output weight matrix (W2). 
            This step is essential to transform the hidden layer activations into the vocabulary space for prediction.
        */       
                        /*Collective<E> u = Numcy::dot(h, W2);*/
        
        /*Collective<E> u_positive;

        if (negative_context.getShape().getN())
        {
            u_positive = Numcy::dot(h, W2_positive);
        }

        Collective<E> y_pred_positive;

        if (negative_context.getShape().getN())
        {
            y_pred_positive = Numcy::sigmoid<E>(u_positive);
        }*/

        // h, W2_positive, u_positve, y_pred_positive
        
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
                    //Collective<E> y_pred = /*softmax<E>(u)*/ Numcy::sigmoid<E>(u);

        Collective<E> h_negative, u_negative, y_pred_negative, W2_negative;

        /*
         * Processes negative samples when provided.
         * Negative samples are incorrect context words used for contrastive learning.
         */
        if (negative_context.getShape().getN())
        {
            E* ptr = NULL;
            ptr = cc_tokenizer::allocator<E>().allocate(W2.getShape().getNumberOfRows()*negative_context.getShape().getN());

            /*
             *   he Dot Product Is with Specific Word Vectors, Not All of W2.
             *   The main goal of negative sampling is to avoid calculations involving the entire W2 matrix                
             */
            W2_negative = Collective<E>{ptr, DIMENSIONS{negative_context.getShape().getN(), W2.getShape().getNumberOfRows(), NULL, NULL}};

            /*std::cout<< "Shape of W2: Columns = " << W2.getShape().getNumberOfColumns() << ", Rows = " << W2.getShape().getNumberOfRows() << std::endl;
            std::cout<< "Shape of W2_negative: Columns = " << W2_negative.getShape().getNumberOfColumns() << ", Rows = " << W2_negative.getShape().getNumberOfRows() << std::endl;*/

            for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < negative_context.getShape().getN(); i++)
            {
                Collective<E> temp = W2.slice(negative_context[i], DIMENSIONS{1, W2.getShape().getNumberOfRows(), NULL, NULL}, AXIS_ROWS);
                W2_negative.update_column(i, temp);
            }
                        
            /*
             *  ROWSxCOLUMNS
             *  ------------ 
             *   nxm 1x32 
             *     mxp 32x10                
             *   nxp 1x10 
             *   
             *   Because:
             *      // negative_samples is an array of k word indices
             *      Collective<E> u_negative_scores(k); // A vector to hold the k scores
             *   
             *      for (int i = 0; i < k; ++i)
             *      {
             *          Collective<E> negative_word_vec = W2.getRow(negative_samples[i]);
             *          u_negative_scores[i] = Numcy::dot(h, negative_word_vec);
             *      }
             */
            /*u_negative = Numcy::dot(h_negative, W2);*/
            /*std::cout<< "h = Columns: " << h.getShape().getNumberOfColumns() << ", Rows: " << h.getShape().getNumberOfRows() << std::endl;
            std::cout<< "W2_negative = Columns: " << W2_negative.getShape().getNumberOfColumns() << ", Rows: " << W2_negative.getShape().getNumberOfRows() << std::endl;*/
            u_negative = Numcy::dot(h, W2_negative); 
            // Now, apply the sigmoid function to the k negative scores
            y_pred_negative = /*softmax(u_negative)*/ Numcy::sigmoid<E>(u_negative);
        }

        // Return all computed values for use in backpropagation and loss calculation
        return forward_propagation<E>{h, // Hidden layer representation
                                      y_pred_positive, // Positive sample probability
                                      u_positive, // Positive sample raw scores  
                                      /*h_negative,*/  
                                      y_pred_negative, // Negative sample probabilities
                                      u_negative, // Negative sample raw scores
                                    };
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

template <typename E = double>
forward_propagation<E> forward_old(Collective<E>& W1, Collective<E>& W2, CORPUS_REF vocab, WORDPAIRS_PTR pair = NULL) throw (ala_exception)
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
            This intermediate value is used in calculations involving gradients in "backward pass" or "back propagation"(the function backward).
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

        return forward_propagation<E>{h, y_pred, u};
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
backward_propagation<T> backward_new(Collective<T>& W1, Collective<T>& W2, CORPUS_REF vocab, forward_propagation<T>& fp, WORDPAIRS_PTR pair, bool verbose = false) throw (ala_exception)
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
       
        //backward_propagation<T> ret = backward_propagation<T>{grad_W1, grad_W2, Collective<T>{NULL, DIMENSIONS{0, 0, NULL, NULL}}};
        //return backward_propagation<T>{grad_W1, grad_W2, Collective<T>{NULL, DIMENSIONS{0, 0, NULL, NULL}}};
        DIMENSIONS dim = DIMENSIONS{0, 0, NULL, NULL};
        Collective<T> foo = Collective<T>{NULL, dim.copy()};
        //return ret;  
        
        backward_propagation<T> ret = backward_propagation<T>{grad_W1, grad_W2, foo};

        return ret;
}

template <typename T = double>
backward_propagation<T> backward(Collective<T>& W1, Collective<T>& W2, Collective<cc_tokenizer::string_character_traits<char>::size_type>& negative_context, CORPUS_REF vocab, forward_propagation<T>& fp, WORDPAIRS_PTR pair, bool verbose = false) throw (ala_exception)
{
    /* The hot one array is row vector, and has shape (1, vocab.len = REPLIKA_VOCABULARY_LENGTH a.k.a no redundency) */
    Collective<T> oneHot;
    /* The shape of grad_u is the same as y_pred (fp.predicted_probabilities) which is (1, len(vocab) without redundency) */
    Collective<T> grad_u_positive, grad_u_negative;
    /*          
     */
    Collective<T> grad_u, grad_u_T;
    /*
        Dimensions of grad_u is (1, len(vocab) without redundency)
        Dimensions of fp.intermediate_activation (1, len(vocab) without redundency)

        Dimensions of grad_W2 is (len(vocab) without redundency, len(vocab) without redundency)        
     */
    Collective<T> grad_W2_positive, grad_W2_negative;    
    /*
       Dimensions of grad_u is (1, len(vocab) without redundency)
       Dimensions of W2_T is (len(vocab) without redundency, SKIP_GRAM_EMBEDDNG_VECTOR_SIZE)

       Dimensions of grad_h is (1, SKIP_GRAM_EMBEDDNG_VECTOR_SIZE)
     */
    Collective<T> grad_h;
    /*
        Dimensions of grad_W1 is (len(vocab) without redundency, SKIP_GRAM_EMBEDDNG_VECTOR_SIZE)
     */
    Collective<T> grad_W1, grad_W2;
    

    /*
        Creating a One-Hot Vector, using Numcy::zeros with a shape of (1, vocab.numberOfUniqueTokens()).
        This creates a zero-filled column vector with a length equal to the vocabulary size
     */
    try 
    {       
        if (negative_context.getShape().getN() == 0)
        {
            oneHot = Numcy::zeros(DIMENSIONS{vocab.numberOfUniqueTokens(), 1, NULL, NULL});
            /*std::cout<< oneHot.getShape().getNumberOfColumns() << " -- " << oneHot.getShape().getNumberOfRows() << std::endl;*/ 
            oneHot[pair->getCenterWord() - INDEX_ORIGINATES_AT_VALUE] = 1;

            grad_u = Numcy::subtract<double>(fp.predicted_probabilities, oneHot);
            grad_u_T = Numcy::transpose(grad_u);
            grad_W2 = Numcy::outer(fp.hidden_layer_vector, grad_u);
            grad_h = Numcy::dot(W2, grad_u_T);
        }
        else
        {
            /*
             * Calculate the error signal for the true, positive center word. The "target" label for this word is 1.
             * fp.predicted_probabilities is the sigmoid output for the positive word (a single value)
             * The target is 1.0
             */
            grad_u_positive = fp.predicted_probabilities - 1.0;
            /*
             * Calculate the error signals for each of the k negative samples. The "target" label for all these fake words is 0.
             * Since subtracting zero does nothing, the error signals are simply the predicted probabilities themselves   
             */
            grad_u_negative = fp.negative_predicted_probabilities /*- 0.0*/;

            /*
             * Calculate the Gradient for W2 (Sparse Update):
             * Instead of calculating a dense gradient for the entire W2 matrix, we now only calculate updates for the rows we actually used: the one positive word and the k negative words.
             * Action: Initialize grad_W2 as a matrix of zeros. Then, calculate the gradient for each relevant word vector (error * h) and add it to the corresponding row in grad_W2  
             */
            grad_W2 = Numcy::zeros(W2.getShape()); 
            /*
             *  1. Update for the positive word             
             */
            grad_W2_positive = Numcy::outer(fp.hidden_layer_vector, grad_u_positive);                         
            /*std::cout<< "grad_W2 = " << grad_W2.getShape().getNumberOfColumns() << ", " << grad_W2.getShape().getNumberOfRows() << std::endl;
            std::cout<< "grad_u_positive = " << grad_u_positive.getShape().getNumberOfColumns() << ", " << grad_u_positive.getShape().getNumberOfRows() << std::endl;
            std::cout<< "fp.hidden_layer_vector = " << fp.hidden_layer_vector.getShape().getNumberOfColumns() << ", " << fp.hidden_layer_vector.getShape().getNumberOfRows() << std::endl;*/
            /*
             * 1.1. Add this gradient to the correct row in grad_W2
             * Get the scalar error value from the 1x1 Collective grad_u_positive, because "*" is overloaded for Collective instances so you do not explcitly have to do grad_u_positive[0]
             * Perform scalar-vector multiplication. Add this column gradient to the correct column in grad_W2
             */                        
            grad_W2.update_column(pair->getCenterWord() - INDEX_ORIGINATES_AT_VALUE, /*grad_W2_positive*/ fp.hidden_layer_vector*grad_u_positive);
            /*
             * 2. Loop through and update for each negative word  and repeat the above mentioned process for each negative context 
             */            
            for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < negative_context.getShape().getN(); i++)
            {
                /*std::cout<< "--> " << negative_context[i] << std::endl;
                std::cout<< "=======>>>>>>> " << grad_u_negative[negative_context[i]] << std::endl;*/
                grad_W2.update_column(negative_context[i], fp.hidden_layer_vector*grad_u_negative[/*negative_context[*/i/*]*/]); 
           }
           
           /*
            * Calculate the Gradient for the Hidden Layer h
            * The gradient grad_h is the sum of the errors propagated back from all the output neurons we used (1 positive + k negative).
            * 1. Initialize grad_h with zeros
            */
           grad_h = Numcy::zeros(fp.hidden_layer_vector.getShape());

           /*
            * 2. Propagate error from the positive word.
            * 2.1 Start with getting positive word vector
            */
           Collective<T> W2_positive_vec = W2.slice(pair->getCenterWord() - INDEX_ORIGINATES_AT_VALUE, DIMENSIONS{W2.getShape().getNumberOfRows(), 1, NULL, NULL}, AXIS_ROWS);
           Collective<T> temp = W2_positive_vec * grad_u_positive;
           grad_h = grad_h + temp /*(grad_u_positive * W2_positive_vec))*/;

           /*
            * 3. Loop and propagate error from negative words
            * 3.1 Start with get negative word vector/s
            */
           for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < negative_context.getShape().getN(); i++)
           {
                Collective<T> W2_negative_vec = W2.slice(negative_context[i], DIMENSIONS{W2.getShape().getNumberOfRows(), 1, NULL, NULL}, AXIS_ROWS);
                temp = W2_negative_vec * grad_u_negative[i];
                grad_h = grad_h + temp;
           }
        }
       
        /*for (int i = 0; i < oneHot.getShape().getN(); i++)
        {
            std::cout<< oneHot[i] << ", ";
        }
        std::cout<< std::endl;*/
        /*for (int i = 0; i < fp.predicted_probabilities.getShape().getN(); i++)
        {
            std::cout<< fp.predicted_probabilities[i] << ", ";
        }
        std::cout<< std::endl;*/

        //std::cout<< fp.predicted_probabilities.getShape().getNumberOfColumns() << ", " << fp.predicted_probabilities.getShape().getNumberOfRows() << std::endl;
        
                /*if (!negative_context.getShape().getN())
                {
                    grad_u_positive = Numcy::subtract<double>(fp.predicted_probabilities, oneHot);
                }
                else
                {
                    grad_u_positive = fp.predicted_probabilities - 1.0;
                    grad_u_negative = fp.negative_predicted_probabilities - 0.0;
                }*/
        
        //std::cout<< grad_u.getShape().getNumberOfColumns() << " -- " << grad_u.getShape().getNumberOfRows() << std::endl;

        /*std::cout<< "--------------------" << std::endl;*/

        /*for (int i = 0; i < grad_u.getShape().getN(); i++)
        {
            std::cout<< grad_u[i] << ", ";
        }
        std::cout<< std::endl;*/

                    /*grad_u_T = Numcy::transpose(grad_u_positive);
                    grad_W2_positive = Numcy::outer(fp.hidden_layer_vector, grad_u_positive);*/

        //std::cout<< grad_u_T.getShape().getNumberOfColumns() << " -- " << grad_u_T.getShape().getNumberOfRows() << std::endl;
        
        //std::cout<< W2.getShape().getNumberOfColumns() << " -- " << W2.getShape().getNumberOfRows() << std::endl;
        //std::cout<< grad_u_T.getShape().getNumberOfColumns() << " -- " << grad_u_T.getShape().getNumberOfRows() << std::endl;
                    /*grad_h = Numcy::dot(W2, grad_u_T);*/
        //std::cout<< grad_h.getShape().getNumberOfColumns() << " -- " << grad_h.getShape().getNumberOfRows() << std::endl;
        
        /*
         * Calculate the Gradient for W1
         */
        grad_W1 = Numcy::zeros<T>(DIMENSIONS{SKIP_GRAM_EMBEDDING_VECTOR_SIZE, vocab.numberOfUniqueTokens(), NULL, NULL});
        /*std::cout<< grad_W1.getShape().getNumberOfColumns() << " -- " << grad_W1.getShape().getNumberOfRows() << std::endl;
        for (int i = 0; i < grad_W1.getShape().getN(); i++)
        {
            std::cout<< grad_W1[i] << ", ";
        }
        std::cout<< std::endl;*/
        
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
            /*else 
            {
                // It wll be very big number
                std::cout<< "--> " << ((*(pair->getLeft()))[i] - INDEX_ORIGINATES_AT_VALUE) << std::endl;
            }*/
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
            /*else 
            {
                // It wll be very big number
                std::cout<< "--> " << ((*(pair->getRight()))[i] - INDEX_ORIGINATES_AT_VALUE) << std::endl;
            }*/
        }               
    }
    catch (ala_exception& e)
    {
        throw ala_exception(cc_tokenizer::String<char>("backward(Collective<T>&, Collective<T>&, Collective<cc_tokenizer::string_character_traits<char>::size_type>&, CORPUS_REF, forward_propagation<T>&, WORDPAIRS_PTR, bool) -> ") + cc_tokenizer::String<char>(e.what()));
    }

    /*
        Dimensions of grad_W1 is (len(vocab) without redundency, SKIP_GRAM_EMBEDDNG_VECTOR_SIZE)
        Dimensions of grad_W2 is (len(vocab) without redundency, len(vocab) without redundency)
     */ 
    
    DIMENSIONS temp1 = DIMENSIONS{0, 0, NULL, NULL};
    Collective<T> temp2 = Collective<T>{NULL, temp1/*.copy()*/};       
    backward_propagation<T> ret = backward_propagation<T>{grad_W1, grad_W2, temp2};
    
    /*std::cout<< grad_W1.getShape().getNumberOfColumns() << " -- " << grad_W1.getShape().getNumberOfRows() << std::endl;
    for (int i = 0; i < grad_W1.getShape().getN(); i++)
    {
        std::cout<< grad_W1[i] << ", ";
    }
    std::cout<< std::endl;
    std::cout<< "----------------------------------------------------------------------" << std::endl;*/

    /*std::cout<< grad_W2.getShape().getNumberOfColumns() << " -- " << grad_W2.getShape().getNumberOfRows() << std::endl;
    for (int i = 0; i < grad_W2.getShape().getN(); i++)
    {
        std::cout<< grad_W2[i] << ", ";
    }
    std::cout<< std::endl;
    std::cout<< "----------------------------------------------------------------------" << std::endl;*/
    
    return ret;
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
 * @param rs (Type: float or double)
 *      The regularization strength for the model. This parameter helps prevent overfitting by penalizing large weights in the model.
 * 
 * @param ns (Type: cc_tokenizer::string_character_traits<char>::size_type)
 * 
 * 
 * @param pairs: (Type: PAIRS)
 *      The training data, which contains word pairs. Each pair consists of a center word and its surrounding context words. The model learns to predict the center word from its context.
 *  
 * @param t: (Template Type)
 *      A generic template type for numerical operations. This allows the macro to work with different numerical types (e.g., float, double) depending on precision requirements.
 *
 * @param verbose: (Type: bool)
 *      If true, prints detailed logs of the training process, such as the current epoch number and other information for debugging or tracking purposes.
 *
 * @param vocab: (Type: CORPUS)
 *      The vocabulary data structure that stores all the unique words used in training. It is required to index and retrieve word embeddings during forward and backward propagation.
 *
 * @param W1: (Type: Collective<t>)
 *      The input-to-hidden weight matrix. It holds the word embeddings for the context words and is updated during backpropagation.
 *
 * @param W2: (Type: Collective<t>)
 *      The hidden-to-output weight matrix. This matrix is used to predict the center word from the context word embeddings and is also updated during training.
 * @param W1_best: (Type: Collective<t>)
 *      W1 weights with minimum validation loss.
 * 
 * @param W2_best: (Type: Collective<t>)
 *      W2 weights with minimum validation loss.
 * 
 */
#define CBOW_TRAINING_LOOP(el, epoch, lr, rs, ns, pairs, t, verbose, vocab, W1, W2, W1_best, W2_best)\
{\
    cc_tokenizer::string_character_traits<char>::size_type best_epoch = 0;\
    /* Initialize to infinity */\
    t best_validation_loss = std::numeric_limits<t>::infinity();\
    forward_propagation<t> fp;\
    backward_propagation<t> bp;\
    /* Epoch loop: Main loop for epochs */\
    for (cc_tokenizer::string_character_traits<char>::size_type i = 1; i <= epoch; i++)\
    {\
        if (verbose)\
        {\
            std::cout<< "Epoch# " << i << " of " << epoch << " epochs." << std::endl;\
        }\
        /* Shuffle Word Pairs: Shuffles the training data (word pairs) before each epoch to avoid biases in weight updates */\
        Numcy::Random::shuffle<PAIRS>(training_pairs, PAIRS_VOCABULARY_TRAINING_SPLIT(pairs.get_number_of_word_pairs()));\
        /*---------------------------------------------------------*/\
        /*    PHASE 1: Training Weights with the training data     */\
        /*---------------------------------------------------------*/\
        /* Iterates through each word pair in the training data  */\
        while (pairs.go_to_next_word_pair(PAIRS_TRAINING_PHASE) != cc_tokenizer::string_character_traits<char>::eof())\
        {\
            /* Get Current Word Pair: We've a pair, a pair is LEFT_CONTEXT_WORD/S CENTER_WORD and RIGHT_CONTEXT_WORD/S */\
            WORDPAIRS_PTR pair = pairs.get_current_word_pair();\
            Collective<cc_tokenizer::string_character_traits<char>::size_type> negative_samples = generateNegativeSamples_cbow(vocab, pair, ns);\
            try\
            {\
                /*forward_propagation<t>*/ fp = forward<t> (W1, W2, negative_samples, vocab, pair);\
                /*backward_propagation<t>*/ bp = backward (W1, W2, negative_samples, vocab, fp, pair);\
                /*for (int i = 0; i < bp.grad_weights_input_to_hidden.getShape().getN(); i++)*/\
                /*{*/\
                    /*std::cout<< bp.grad_weights_input_to_hidden[i] << ", ";*/\
                /*}*/\
                /*std::cout<< std::endl;*/\
                /*std::cout<< bp.grad_weights_hidden_to_output.getShape().getNumberOfColumns() << " -- " << bp.grad_weights_hidden_to_output.getShape().getNumberOfRows() << std::endl;*/\
                /*for (int i = 0; i < bp.grad_weights_hidden_to_output.getShape().getN(); i++)*/\
                /*{*/\
                    /*std::cout<< bp.grad_weights_hidden_to_output[i] << ", ";*/\
                /*}*/\
                /*std::cout<< std::endl;*/\
                /* Relationship Between Learning Rate (lr) and Regularization Strength (rs) */\
                /* ------------------------------------------------------------------------ */\
                /* - High learning rate (lr): */\
                /*   - Often requires higher regularization strength (rs) to prevent overfitting or unstable updates. */\
                /*   - Large parameter updates can cause the model to overshoot the optimal solution or overfit. */\
                /*   - Increasing rs penalizes large weights, stabilizing training and reducing overfitting. */\
                /*  */\
                /* - Low learning rate (lr): */\
                /*   - Allows for lower or no regularization strength (rs) since smaller updates reduce overfitting risk. */\
                /*   - The model converges more slowly, and heavy regularization may unnecessarily slow training. */\
                /*  */\
                /* L2 Regularization (Weight Decay) */\
                /* -------------------------------- */\
                /* - Regularization strength (rs) controls the penalty applied to large weights. */\
                /* - During weight updates, the gradient is adjusted by adding the regularization term (W1 * rs or W2 * rs). */\
                /* - This penalizes large weights, helping to prevent overfitting. */\
                /* - Key Considerations: */\
                /*   - Avoid setting rs too high, as it may excessively penalize weights and slow convergence. */\
                /*   - Balance rs and lr to achieve stable training and good generalization. */\
                /*   - Adjust rs based on the learning rate (lr) to balance regularization and training speed. */\
                if (rs == 0)\
                {\
                    /*std::cout<< W1.getShape().getNumberOfColumns() << " -- " << W1.getShape().getNumberOfRows() << std::endl;*/\
                    /*std::cout<< "--------------------" << std::endl;*/\
                    /*for (int i = 0; i < W1.getShape().getN(); i++)*/\
                    /*{*/\
                        /*std::cout<< W1[i] << ", ";*/\
                    /*}*/\
                    /*std::cout<< std::endl;*/\
                    /*std::cout<< "*************************************************************************" << std::endl;*/\
                    /*std::cout<< bp.grad_weights_input_to_hidden.getShape().getNumberOfColumns() << " -- " << bp.grad_weights_input_to_hidden.getShape().getNumberOfRows() << std::endl;*/\
                    /*for (int i = 0; i < bp.grad_weights_input_to_hidden.getShape().getN(); i++)*/\
                    /*{*/\
                        /*bp.grad_weights_input_to_hidden[i] = 1;*/\
                        /*std::cout<< bp.grad_weights_input_to_hidden[i] << ", ";*/\
                    /*}*/\
                    /*std::cout<< std::endl;*/\
                    /* Update weights without regularization strength */\
                                /*W1 -= bp.grad_weights_input_to_hidden * lr;*/\
                                /*W2 -= bp.grad_weights_hidden_to_output * lr;*/\
                    /*std::cout<< W1.getShape().getNumberOfColumns() << " -- " << W1.getShape().getNumberOfRows() << std::endl;*/\
                    /*std::cout<< "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" << std::endl;*/\
                    /*for (int i = 0; i < W1.getShape().getN(); i++)*/\
                    /*{*/\
                        /*std::cout<< W1[i] << ", ";*/\
                    /*}*/\
                    /*std::cout<< std::endl;*/\
                    Collective<t> W1_product = bp.grad_weights_input_to_hidden * lr;\
                    Collective<t> W2_product = bp.grad_weights_hidden_to_output * lr;\
                    /*std::cout<< W1_product.getShape().getNumberOfColumns() << " -- " << W1_product.getShape().getNumberOfRows() << std::endl;*/\
                    /*std::cout<< W2_product.getShape().getNumberOfColumns() << " -- " << W2_product.getShape().getNumberOfRows() << std::endl;*/\
                    /*std::cout<< "W1 = " << PAIRS_VOCABULARY_TRAINING_SPLIT(W1.getShape().getNumberOfRows()) << ", W2 = " << PAIRS_VOCABULARY_TRAINING_SPLIT(W2.getShape().getNumberOfRows()) << std::endl;*/\
                    for (cc_tokenizer::string_character_traits<char>::size_type j = 0; j < PAIRS_VOCABULARY_TRAINING_SPLIT(W1.getShape().getNumberOfRows()); j++)\
                    {\
                        for (cc_tokenizer::string_character_traits<char>::size_type k = 0; k < W1.getShape().getNumberOfColumns(); k++)\
                        {\
                            W1[j*W1.getShape().getNumberOfColumns() + k] = W1[j*W1.getShape().getNumberOfColumns() + k] - W1_product[j*W1_product.getShape().getNumberOfColumns() + k];\
                        }\
                    }\
                    \
                    for (cc_tokenizer::string_character_traits<char>::size_type j = 0; j < PAIRS_VOCABULARY_TRAINING_SPLIT(W2.getShape().getNumberOfRows()); j++)\
                    {\
                        for (cc_tokenizer::string_character_traits<char>::size_type k = 0; k < W2.getShape().getNumberOfColumns(); k++)\
                        {\
                            W2[j*W2.getShape().getNumberOfColumns() + k] = W2[j*W2.getShape().getNumberOfColumns() + k] - W2_product[j*W2_product.getShape().getNumberOfColumns() + k];\
                        }\
                    }\
                }\
                else\
                {\
                    /* Update weights with regulariztion strength */\
                    /*W1 -= ((bp.grad_weights_input_to_hidden + (W1 * rs)) * lr);*/\
                    /*W2 -= ((bp.grad_weights_hidden_to_output + (W2 * rs)) * lr);*/\
                    \
                    /* Update weights with regulariztion strength */\
                    Collective<t> rs_w1 = W1 * rs;\
                    Collective<t> rs_w2 = W2 * rs;\
                    Collective<t> grad_w1_plus_rs_w1_to_lr = ((bp.grad_weights_input_to_hidden + rs_w1/*<->(W1 * rs)*/) * lr);\
                    Collective<t> grad_w2_plus_rs_w2_to_lr = ((bp.grad_weights_hidden_to_output + rs_w2 /*<->(W2 * rs)*/) * lr);\
                    for (cc_tokenizer::string_character_traits<char>::size_type j = 0; j < PAIRS_VOCABULARY_TRAINING_SPLIT(W1.getShape().getNumberOfRows()); j++)\
                    {\
                        for (cc_tokenizer::string_character_traits<char>::size_type k = 0; k < W1.getShape().getNumberOfColumns(); k++)\
                        {\
                            W1[j*W1.getShape().getNumberOfColumns() + k] = W1[j*W1.getShape().getNumberOfColumns() + k] - grad_w1_plus_rs_w1_to_lr[j*grad_w1_plus_rs_w1_to_lr.getShape().getNumberOfColumns() + k];\
                        }\
                    }\
                    /*W1 -= ((bp.grad_weights_input_to_hidden + rs_w1*//*<->(W1 * rs)*//*) * lr);*/\
                    for (cc_tokenizer::string_character_traits<char>::size_type j = 0; j < PAIRS_VOCABULARY_TRAINING_SPLIT(W2.getShape().getNumberOfRows()); j++)\
                    {\
                        for (cc_tokenizer::string_character_traits<char>::size_type k = 0; k < W2.getShape().getNumberOfColumns(); k++)\
                        {\
                            W2[j*W2.getShape().getNumberOfColumns() + k] = W2[j*W2.getShape().getNumberOfColumns() + k] - grad_w2_plus_rs_w2_to_lr[j*grad_w2_plus_rs_w2_to_lr.getShape().getNumberOfColumns() + k];\
                        }\
                    }\
                    /*W2 -= ((bp.grad_weights_hidden_to_output + rs_w2*/ /*<->(W2 * rs)*//*) * lr);*/\
                }\
                /* Loss Function: The CBOW model typically uses negative log-likelihood (NLL) as the loss function.\
                   In NLL, lower values indicate better performance. */\
                if (negative_samples.getShape().getN() == 0)\
                {\
                    el = el + (-1*log(fp.pb(pair->getCenterWord() - INDEX_ORIGINATES_AT_VALUE)));\
                }\
                else\
                {\
                    el = el + (-1*log(fp.pb(0)));\
                }\
                /*Calculate negative loss (for negative samples)*/\
                if (negative_samples.getShape().getN() > 0)\
                {\
                    for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < negative_samples.getShape().getN(); i++)\
                    {\
                        /* Assuming y_pred_negative is a Collective<E> that holds probabilities for negative samples */\
                        t negative_loss = -1 * log(1 - fp.npp(i)); /* Calculate loss for each negative sample */\
                        el = el + negative_loss;\
                    }\
                }\
            }\
            catch (ala_exception& e)\
            {\
                std::cout<< "CBOW_TRAINING_LOOP() PHASE 1 -> " << e.what() << std::endl;\
            }\
        }\
        std::cout<< "epoch_loss = " << el/PAIRS_VOCABULARY_TRAINING_SPLIT(pairs.get_number_of_word_pairs()/**PAIRS_VOCABULARY_TRAINING_SPLIT*/) << std::endl;\
        el = 0;\
        /*---------------------------------------------------------*/\
        /*  PHASE 2: VALIDATION Weights with the validation data   */\
        /*---------------------------------------------------------*/\
        /* Now, with the updated weights from this epoch,*/\
        /* see how the model performs on the unseen validation set.*/\
        t validation_loss_accumulator = 0;\
        while (pairs.go_to_next_word_pair(PAIRS_VALIDATION_PHASE) != cc_tokenizer::string_character_traits<char>::eof())\
        {\
            /* Get Current Word Pair: We've a pair, a pair is LEFT_CONTEXT_WORD/S CENTER_WORD and RIGHT_CONTEXT_WORD/S */\
            WORDPAIRS_PTR pair = pairs.get_current_word_pair();\
            /* We are not using negative sampling, that is why the following redeclartion */\
            Collective<cc_tokenizer::string_character_traits<char>::size_type> negative_samples /*= generateNegativeSamples_cbow(vocab, pair, static_cast<cc_tokenizer::string_character_traits<char>::size_type>(CBOW_NEGATIVE_SAMPLE_SIZE))*/;\
            try\
            {\
                /* Perform only a forward pass to see how the model performs on data it hasn't trained on */\
                /* Crucially, you do not perform backpropagation or update any weights (W1, W2). The model's weights are effectively frozen during this phase */\
                forward_propagation<t> fp = forward (W1, W2, negative_samples, vocab, pair);\
                validation_loss_accumulator = validation_loss_accumulator + (-1*log(fp.pb(pair->getCenterWord() - INDEX_ORIGINATES_AT_VALUE)));\
            }\
            catch (ala_exception& e)\
            {\
                std::cout<< "CBOW_TRAINING_LOOP() PHASE 2 -> " << e.what() << std::endl;\
            }\
        }\
        /*--------------------------------------------------*/\
        /*      PHASE 3: LOGGING AND DECISION MAKING        */\
        /*--------------------------------------------------*/\
        t avg_validation_loss = validation_loss_accumulator / PAIRS_VOCABULARY_VALIDATION_SPLIT(pairs.get_number_of_word_pairs());\
        if (avg_validation_loss < best_validation_loss)\
        {\
            best_validation_loss = avg_validation_loss;\
            best_epoch = i;\
            for (cc_tokenizer::string_character_traits<char>::size_type j = 0; j < W1.getShape().getN(); j++)\
            {\
                W1_best[j] = W1[j];\
                W2_best[j] = W2[j];\
            }\
        }\
        std::cout<< "Accumulated validation loss: " << validation_loss_accumulator << ", ";\
        std::cout<< "Average validation loss: " << avg_validation_loss << std::endl;\
        std::cout << "--- Best validation loss so far: " << best_validation_loss << " (at epoch " << best_epoch << ") ---" << std::endl;\
        \
        \
        /*----------------------------------------------------*/\
        /*   Optional: Check for early stopping conditions    */\
        /*----------------------------------------------------*/\
    }\
    /* After the entire loop finishes */\
    std::cout << "*\nBest overall validation loss (Final Validation Loss): " << best_validation_loss << " (at epoch " << best_epoch << ")" << std::endl;\
}\

#endif
