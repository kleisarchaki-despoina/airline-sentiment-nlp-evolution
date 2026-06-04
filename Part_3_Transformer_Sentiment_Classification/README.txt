# Part 3: Classification of Airline Tweets using Pretrained Transformers (DistilBERT)

This folder contains the third and final phase of our NLP architectural evolution series, transitioning from custom sequential recurrent networks to modern, state-of-the-art **Transformer-based Transfer Learning**.

## Architectural Engineering & Fine-Tuning Strategy
To achieve maximum semantic depth and solve the multi-class sentiment classification challenge on highly unstructured Twitter data, this phase utilizes **DistilBERT** (a distilled, computationally efficient variant of the Bidirectional Encoder Representations from Transformers - BERT) implemented via **Hugging Face Transformers** and **PyTorch**:

- **Self-Attention Mechanics:** Leverages bi-directional attention mechanisms to compute contextual token representations natively, capturing long-range semantic dependencies and subtle conversational nuances.
- **Advanced Fine-Tuning Pipeline:** Formulated a specialized training loop utilizing the **AdamW optimizer** (weight decay regularized) alongside a **Linear Learning Rate Scheduler** to control gradient descent updates and prevent catastrophic forgetting.
- **Dynamic Tokenization:** Deployed the pretrained `DistilTokenizer` to manage sequence lengths via strict attention masking and padding mechanics.

## Comprehensive Performance & Benchmark Breakthroughs

Fine-tuning DistilBERT yielded a definitive performance breakthrough, pushing the **Macro Average F1-Score to 80%**, outperforming both the ANN baseline and the stacked Bi-LSTM with Attention frameworks:

1. **The Neutral Class Resolution:** In both previous architectures, the **Neutral** class presented a structural bottleneck due to conversational ambiguity. DistilBERT successfully resolved this challenge, driving the Neutral F1-Score up to **69%**. Its deep attention masks allowed the network to correctly distinguish subtle operational statements from active sentiment bias.
2. **Precision & Recall Superiority:** - **Negative Sentiment:** Achieved a near-flawless **Recall of 92%** (1,686 correctly identified tweets), creating an industry-grade containment filter for customer complaints.
   - **Positive Sentiment:** Secured a stellar **Precision of 81%** and an F1-Score of **76%**, proving that large-scale pretrained language representations effectively minimize false-positive sentiment mappings.
3. **Training & Convergence Realism:** The validation diagnostics confirmed rapid semantic convergence. Thanks to massive scale pretraining on vast textual corpora, the model requires minimal downstream epochs to lock onto domain-specific syntax, avoiding representation collapse or local minima limitations.

## Files
- `Part 3 - Sentiment Classification using Pretrained Transformers.ipynb` - Production-grade Jupyter Notebook containing the full Transformer fine-tuning, PyTorch loops, and evaluation matrices.