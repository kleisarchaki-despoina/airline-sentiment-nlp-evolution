# US Airline Sentiment Analysis – The NLP Architecture Evolution

This repository features a comprehensive, multi-part **Natural Language Processing (NLP)** and **Deep Learning** benchmarking suite. The project tracks the historical and technical evolution of text classification architectures by solving the same sentiment analysis problem (14,640 tweets categorized into Negative, Neutral, and Positive) across three distinct evolutionary stages of Machine Learning.

---

## Project Structure & Architecture Evolution

### Part 1: Artificial Neural Networks (ANN)
- **Core Strategy:** Multi-Layer Feed-Forward Network compiled via **TensorFlow/Keras**.
- **Feature Engineering:** Benchmarked baseline text vectorization using High-Dimensional Sparse (**TF-IDF**), Word-Level Context (**Word2Vec**), and Sub-word Morphology (**FastText**).
- **Key Takeaway:** Explored the limits of sparse feature spaces, demonstrating how dense embeddings combined with **Early Stopping** successfully mitigate aggressive validation loss divergence (overfitting).

### Part 2: Recurrent Neural Networks (RNN / LSTM / GRU)
- **Core Strategy:** Sequential modeling designed to capture temporal and word-order dependencies within text layout.
- **Implementation:** Built recurrent pipelines using **LSTM (Long Short-Term Memory)** or **GRU** layers to manage the vanishing gradient problem inherent in standard RNNs.
- **Key Takeaway:** Proved that preserving the sequential chronological order of tokens yields a deeper contextual understanding compared to bag-of-words or averaged static word vectors.

### Part 3: Pretrained State-of-the-Art Transformers
- **Core Strategy:** Modern Transfer Learning utilizing a pretrained Transformer model (e.g., **BERT / DistilBERT** or similar via Hugging Face).
- **Implementation:** Fine-tuned a deep attention-based architecture, leveraging self-attention mechanisms to map bi-directional semantic relationships natively.
- **Key Takeaway:** Demonstrates production-grade state-of-the-art accuracy, showing how large-scale pretrained representations eliminate the engineering bottleneck of training embeddings from scratch.

---

## Advanced Technical Implementations

- **End-to-End Pipeline Comparison:** Solves a single business problem using three completely different architectural paradigms (Feed-Forward, Sequential, and Attention-Based).
- **Regularization Pipelines:** Implemented rigorous overfitting controls, structural dropout, and automated convergence checkpoints via validation loss tracking.
- **Advanced Evaluation Metrics:** Benchmarked models using macro/weighted F1-Scores, Confusion Matrices, and training trajectory learning curves.

## Tech Stack & Libraries
- **Deep Learning Frameworks:** TensorFlow / Keras, PyTorch, Hugging Face Transformers
- **NLP Vectorization:** Gensim, Scikit-Learn
- **Data Engineering & Visualization:** Pandas, NumPy, Seaborn, Matplotlib

## License
This project is licensed under the MIT License - see the LICENSE file for details.