# Part 2: Sentiment Classification using Recurrent Neural Networks (RNN) & Temporal Attention

This folder contains the second phase of our NLP architectural evolution series, transitioning from static feed-forward networks to sequential deep learning models designed to capture temporal dependencies, word order, and context within text sequences.

## Architectural Engineering & Strategy
Unlike bag-of-words or static vector averages, this phase treats text as a chronological sequence of tokens. To solve the complex task of multi-class sentiment classification (Negative, Neutral, Positive) on noisy social media data, an advanced sequential pipeline was designed and compiled via **TensorFlow/Keras**:

- **Bidirectional LSTM (Bi-LSTM):** Processes the text sequences simultaneously from left-to-right and right-to-left, capturing both past and future semantic context layers.
- **Stacked Recurrent Layers:** Deployed multiple recurrent hidden layers to allow the network to construct abstract hierarchical representations of syntactic structures.
- **Temporal Attention Mechanism:** Integrated a custom Attention Layer on top of the LSTM outputs. This enables the network to dynamically assign mathematical weights to specific high-impact emotional tokens (e.g., "amazing", "delayed", "worst"), regardless of their chronological position in the tweet.
- **Regularization & Early Stopping:** Configured automated callback monitors tracking validation loss variance to enforce early termination, restoring the absolute optimal weight matrix and neutralizing representation collapse.

## Comprehensive Performance & Diagnostic Insights

The sequence-based network achieved a strong **Macro Average F1-Score of 69%**, demonstrating highly balanced contextual learning across imbalanced distributions:

1. **Learning Curve Overfitting Diagnostics:** The training loops exposed a classic structural divergence. The network achieved its optimal generalization capacity rapidly at **Epoch 2** (Validation Loss ~0.60). Beyond this threshold, training loss continued to decay towards 0.2 due to weight memorization, while validation loss experienced a steady upward climb, highlighting the critical role of Early Stopping regularizers.
2. **Class Imbalance Resilience:** The network demonstrated exceptional predictive capacity regarding the heavily populated **Negative** class, achieving a **Recall of 89%** (1,624 correctly classified tweets). 
3. **The Semantic Neutral Bottleneck:** The **Neutral** class remained the most structurally challenging category to separate, yielding the lowest local F1-Score (**56%**). Out of 618 ground-truth Neutral tweets, the model successfully isolated 343, frequently experiencing boundary confusion with both Negative and Positive classes due to the conversational ambiguity inherent in short-form text.
4. **Attention Mechanism Validation:** The model handled the **Positive** class reliably, scoring a **Precision of 74%** and an F1-Score of 65%. Visual and metric audits prove that the Attention layer successfully prevented token dilution, locking onto sentiment-driving sub-sequences even within short, heavily constrained character lengths.

## Files
- `Part 2 - Sentiment Classification using RNNs.ipynb` - Production-grade Jupyter Notebook containing the full sequence modeling and attention architecture pipeline.