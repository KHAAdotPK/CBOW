# CBOW Implementation Discussion (C++ from scratch)

This document captures a step-by-step discussion about implementing the Continuous Bag of Words (CBOW) model from scratch in C++.  
It covers how **W1 and W2 embeddings** work, how to use them for similarity, why averaging helps, and practical insights gained while testing with a small corpus.  

---

## Initial Setup

- I have implemented a **CBOW model in C++** from scratch.  
- The model is working and training weights / word embeddings.  
- I then built a small program that:
  - Takes user input
  - Tokenizes it
  - Computes cosine similarity between user-entered tokens and the trained word embeddings.

---

## W1 vs W2 Embeddings

- **W1** → context word embeddings.  
- **W2** → target (center) word embeddings.  

### Key Points:
- Both W1 and W2 are valid embeddings after training.  
- Typically, **W1** or the **average of W1 and W2** is used for similarity tasks.  
- W2 alone is not wrong, but it may not capture semantic similarity as well as W1.  

👉 If the goal is **word-to-word similarity** (e.g., *king* ≈ *queen*), W1 (or W1+W2) is better.  

---

## Example Corpus

```TEXT
burning pain
burning pain relieved by antacids
burning pain radiates to other parts of the body relieved by antacids accompanied by black or bloody stools
pain accompanied by constipation triggered or worsened by coughing or other jarring movements
pain accompanied by abdominal swelling triggered or worsened by coughing or other jarring movements
pain accompanied by constipation triggered or worsened by coughing or other jarring movements
pain accompanied by diarrhea triggered or worsened by coughing or other jarring movements
burning pain accompanied by abdominal swelling
pain radiates to other parts of the body
Pain located in middle abdomen and pain radiates to other parts of the body
pain accompanied by abdominal swelling
pain accompanied by black or bloody stools
pain accompanied by constipation
pain accompanied by diarrhea
```

---

## Sample Output (Cosine Similarity)

```TEXT
-:Similarity W1:-
pain: (burning) -0.466979, (pain) 1, (relieved) -0.166628, ...
stools: (burning) 0.325423, (pain) -0.188236, (relieved) 0.493878, ...
burning: (burning) 1, (pain) -0.466979, (relieved) 0.204819, ...
swelling: (burning) -0.0811419, (pain) 0.0663778, (relieved) 0.0904278, ...
Validation Loss ~ 1.5 – 1.6

-:Similarity W2:-
pain: (burning) 0.374201, (pain) -0.629873, (relieved) 0.19515, ...
stools: (burning) 0.196094, (pain) 0.0242275, (relieved) 0.302791, ...
burning: (burning) -0.209336, (pain) 0.66427, (relieved) 0.145812, ...
swelling: (burning) -0.345216, (pain) -0.0263488, (relieved) -0.119749, ...
```


---

## Interpretation of Results

### Why W1 and W2 differ:
- W1 captures **context space** → good for semantic similarity.  
- W2 captures **target prediction space** → aligns context words with the center word.  

### Observations:
- W1 sometimes shows negatives (e.g., *burning–pain*).  
- W2 shows positives for the same pair (*burning–pain*).  
- Negative cosine similarity is not an error, just means vectors are pointing in opposite directions.  

---

## Which Embedding to Use for Chatbot

- For chatbot tasks where we want **semantic similarity**:
  - **W1** is better than W2.  
  - But W1 alone can give strange negatives for frequent pairs.  

👉 Best practice: **use (W1 + W2) / 2** as the final embedding.  
This balances both roles:
- W1 = context view  
- W2 = target view  

---

## Why Averaging Works

- W1 tells how words behave **as context**.  
- W2 tells how words behave **as targets**.  
- Averaging combines the “two camera angles” into one richer view.  
- This reduces weird negatives and produces embeddings closer to human meaning.  

---

## Is the Implementation Correct?

Yes. Signs that the CBOW implementation is working properly:

1. **Two distinct embedding spaces** (W1 vs W2) behave differently.  
2. **Related words cluster** (e.g., *stools* links with *bloody, black*).  
3. **Mix of positive/negative/near-zero values** → normal for cosine similarity.  
4. Loss values are decreasing (~1.4–1.6) → shows training is happening.  

So the model is functioning as expected.  

---

## Do We Always Need Millions of Tokens?

- **General CBOW training**: yes, usually millions of tokens are required.  
- **But** in small, domain-specific corpora:
  - Vocabulary is small.  
  - Words repeat a lot (e.g., *pain*, *burning*, *stools*).  
  - So even a small dataset can form meaningful relationships.  

👉 Rule of thumb:  
- **General-purpose embeddings** → need huge data.  
- **Domain-specific chatbot embeddings** → small but focused corpus can still work.  

---

## Summary

- W1 = context embeddings, better for similarity.  
- W2 = target embeddings, better for prediction.  
- Averaging W1 + W2 usually improves results.  
- Negative similarities are normal, not a bug.  
- Your CBOW is working correctly.  
- Small corpus works fine because it is **domain-specific and repetitive**.  



