## Gradient Descent Explained (in Layman's Terms)

### Introduction
Gradient descent is like a hiker trying to find the quickest way down a mountain. Imagine standing on a rugged slope, with the goal of reaching the lowest point—the valley. You can't see the entire landscape, but you can feel the steepness of the terrain around you. So, how do you decide which step to take next?

### Steps of Gradient Descent
1. **Starting Point**: Begin somewhere on the mountain. In our case, this starting point represents the initial guess for the solution to a problem (like finding word vectors in the Skip-gram model).

2. **Direction of Steepest Descent**: You want to move downhill, right? So, check the slope around you. Gradient descent calculates the slope (gradient) of a function at your current position. This gradient points in the direction of steepest decrease.

3. **Taking a Step**: Take a step in the opposite direction of the gradient. If the slope is steep, take a big step; if it's gentle, take a smaller step. This step size is determined by a parameter called the learning rate.

4. **Repeat**: Keep repeating steps 2 and 3 until you reach a point where the slope is nearly flat (i.e., you're close to the valley). At that point, you've found an approximate minimum—a solution to your problem.

### Applying Gradient Descent to Skip-gram NLP Model
- The Skip-gram model is like our hiker, exploring the landscape of words in a text corpus.
- Instead of mountains, it navigates through word vectors—dense numerical representations of words.
- The goal? To learn meaningful word embeddings that capture semantic relationships between words.
- Gradient descent helps adjust these word vectors during training. It tweaks them based on how well the model predicts context words given a center word (or vice versa).

In a nutshell, gradient descent guides the Skip-gram model toward better word vectors, just like our hiker descends toward the valley. 🏔️🚶‍♂️

### Gradient Descent in the context of Skip-gram
Gradient Descent is a fundamental optimization technique used to minimize the loss (or error) function in machine learning models by iteratively adjusting the model's parameters (or weights). In the context of following Skip-gram training code, it involves updating the weight matrices (input `(W1)` and output `(W2)` weights) so that the error between predicted and actual context words is minimized.
```C++    
    /*
        Gradient Descent as an Optimization Technique.
        ----------------------------------------------

        Epoch loop: Iterative Adjustment of Parameters (Weights) 
     */
    for (cc_tokenizer::string_character_traits<char>::size_type i = 1; i <= epoch; i++)\
    {\
        /* 
            MORE CODE 
            MORE CODE 
            .........
            ......
            ...
            .
        */            
        /* 
            Iterates through each word pair in the training data.
            For every word pair processed, the weights are updated to minimize the error (loss) gradually
         */  
        while (pairs.go_to_next_word_pair() != cc_tokenizer::string_character_traits<char>::eof())
        {
            /* 
                MORE CODE 
                MORE CODE 
                .........
                ......
                ...
                .
             */                       
            // Forward pass to make predictions. The prediction of context words given a center word            
            fp = forward<t>(W1, W2, vocab, pair);\
            // Backward pass to compute the gradient of the loss with respect to the weights 
            bp = backward<t>(W1, W2, vocab, fp, pair);
            /* 
                MORE CODE 
                MORE CODE 
                .........
                ......
                ...
                .
             */ 
           
            /* 
                Update Weights.
                --------------- 
                
                grad_weights_input_to_hidden is grad_W1 and grad_weights_hidden_to_out is grad_W2.
                The code implements gradient descent to adjust the model's parameters (weights W1 and W2).
                The code adjusts the weights using the gradients obtained from the backward pass.
                The gradients are multiplied by learning rate, the learning rate determines step size for weight updates.
                A small learning rate means smaller steps, which could result in slower learning but more precise convergence.
                A large learning rate means larger steps, which could make learning faster but risk overshooting the optimal solution. The learning rate (lr) controls how quickly or slowly a model's weights are updated during gradient descent. 
             */
            W1 -= bp.grad_weights_input_to_hidden * lr;            
            W2_reshaped -= bp.grad_weights_hidden_to_output * lr;\
            // Reconstructing W2 from its reshaped form
            for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < W2.getShape().getDimensionsOfArray().getNumberOfInnerArrays(); i++)
            {
                for (cc_tokenizer::string_character_traits<char>::size_type j = 0; j < W2.getShape().getNumberOfColumns(); j++)
                {
                    W2[i*W2.getShape().getNumberOfColumns() + j] = W2_reshaped[i*W2_reshaped.getShape().getNumberOfColumns() + j];
                }
            }
        }
        /* 
            Loss Function: The Skip-gram model typically uses negative log-likelihood (NLL) as the loss function.
            In NLL, lower values indicate better performance. 
            Minimizing the Loss Function: The gradient descent process minimizes this loss function by adjusting the weights
            during each iteration. 
            The loss function in this skip-gram model is related to negative log-likelihood (NLL), 
            which measures the difference between the predicted probabilities and the actual context word probabilities
         */
        el = el + (-1*log(fp.pb(pair->getCenterWord() - INDEX_ORIGINATES_AT_VALUE)));\
    }\

    /* Applying these updates in a loop, which ensures that the model learns to make better predictions by reducing the error with each iteration. */
```

### Gradient Descent in the Context of CBOW
The training loop for the Continuous Bag of Words (`CBOW`) model is largely similar to that of the `Skip-gram` model. The key difference lies in the prediction task: while `Skip-gram` predicts context words given a center word, `CBOW` does the opposite by predicting the center word from its surrounding context words.

In `CBOW`, the context words are averaged to form a single input vector that represents the context. This process, while crucial for the model's operation, relates more to the specifics of how information is propagated through the network rather than the gradient descent optimization technique itself. Thus, the iterative adjustment of parameters using gradient descent occurs similarly in both models, focusing on minimizing the loss function and improving predictions.

For a deeper understanding of the propagation mechanism used in `Skip-gram` and `CBOW`, check out the related article on **Propagation** [here](PROPOGATION.md).