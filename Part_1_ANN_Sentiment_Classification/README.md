# Twitter Airline Sentiment – Deep Learning ANN Classification Pipeline

This repository features an advanced **Deep Learning** and **Natural Language Processing (NLP)** pipeline designed to classify public sentiment (Negative, Neutral, Positive) from the Twitter US Airline Sentiment dataset containing approximately 14,640 social media tweets.

The core architectural focus is to evaluate how distinct text embedding spaces influence the convergence, generalization, and overfitting thresholds of a **Multi-Layer Feed-Forward Artificial Neural Network (ANN)**.

## Advanced Deep Learning Implementations

- **Neural Network Architecture:** Designed and tuned a Multi-Layer Feed-Forward ANN utilizing dense hidden layers, **ReLU** activation functions, and a **Softmax** output layer for multi-class cross-entropy classification.
- **Regularization & Early Stopping:** Integrated an automated callback tracking validation loss variance to enforce early termination, successfully protecting the networks against weight memorization and representation collapse.
- **Multi-Vectorization Benchmarking:** Deployed and evaluated three competitive feature extraction frameworks: High-dimensional Sparse (**TF-IDF**), Continuous Word-Level Context (**Word2Vec**), and Sub-word Level Morphology (**FastText**).
- **Performance Evaluation Framework:** Conducted extensive learning curve trajectory audits to map the structural generalization gap of each embedding paradigm.

## Key Architectural Insights

- **Sparse Feature Risk:** The **TF-IDF Neural Pipeline** triggered Early Stopping rapidly at Epoch 8. Sparse feature spaces cause hidden layers to memorize keyword-specific noise, causing immediate validation divergence without active regularization.
- **Continuous Embedding Stability:** Both **Word2Vec** and **FastText** generated highly stable training trajectories with minimal generalization gaps. Dense semantic vectors prevent structural overfitting, though document-level averaging introduces a performance trade-off that requires careful layer depth adjustments.

## Repository Structure

- `Classification with US Airline Sentiment Tweets_updated.ipynb` - The primary production-grade Jupyter Notebook containing the full Deep Learning pipeline.

## ?? Tech Stack & Libraries
- **Language:** Python
- **Deep Learning Framework:** Keras / TensorFlow
- **Natural Language Processing:** Gensim, Scikit-Learn
- **Data Engineering & Visualization:** Pandas, NumPy, Seaborn, Matplotlib

## License
This project is licensed under the MIT License - see the LICENSE file for details.