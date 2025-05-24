### Deep Learning-Based Fake News Detection Using Natural Language Processing

**Author**: Sukanya Ghosh
**GitHub**: [SukanyaGhosh6](https://github.com/SukanyaGhosh6)

---

### 1. Introduction

In the digital era, the spread of misinformation poses a serious threat to informed decision-making and democratic processes. Fake news—articles that deliberately contain false information presented as fact—can go viral within minutes. Recognizing this threat, I designed and implemented an intelligent fake news detection system using deep learning and natural language processing (NLP).

This research project explores how computational techniques can analyze linguistic cues to distinguish false narratives from authentic reporting. The system is designed not only for academic experimentation but also for potential real-world deployment.

---

### 2. Motivation

The project is motivated by the societal impact of misinformation. From pandemic-related rumors to political propaganda, false narratives contribute to fear, confusion, and polarization. Manual fact-checking is slow and inconsistent, creating a demand for scalable, automated systems. I saw this as an opportunity to bridge the gap between deep learning and public service.

As a student and developer deeply interested in the intersection of language and technology, I was curious to explore how machines "understand" news and learn to filter truth from fiction. This project was also an exercise in building explainable AI models for NLP tasks.

---

### 3. Objectives

* Build an end-to-end fake news classifier using a deep learning model.
* Preprocess and vectorize raw textual data efficiently.
* Leverage bidirectional LSTM networks to learn context.
* Use SHAP (SHapley Additive exPlanations) for model interpretability.
* Provide an open-source, modular codebase for reproducibility.

---

### 4. Dataset Description

* **Source**: [Fake and real news dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)
* **Records**: \~40,000 news headlines and articles
* **Features**:

  * `title`: Headline of the news article
  * `text`: Body of the article
  * `subject`: Topic label (e.g., politics, world, tech)
  * `date`: Publication date
  * `label`: Ground truth class (0 = fake, 1 = real)

---

### 5. Methodology

#### 5.1 Data Preprocessing

* Converted all text to lowercase
* Removed punctuation, special characters, stopwords
* Applied tokenization and lemmatization
* Merged `title` and `text` for input features

#### 5.2 Embedding and Vectorization

* Used **Keras Tokenizer** to convert text into sequences
* Applied **padding** to maintain sequence length uniformity
* Created an **embedding matrix** from GloVe for semantic richness

#### 5.3 Model Architecture

* Input Layer → Embedding Layer (GloVe vectors)
* Bidirectional LSTM Layer
* Dropout Layer for regularization
* Dense Output Layer with Sigmoid activation

#### 5.4 Training

* Binary cross-entropy loss
* Adam optimizer
* Early stopping and validation loss monitoring

#### 5.5 Evaluation Metrics

* Accuracy
* Precision, Recall, F1-Score
* Confusion Matrix
* ROC-AUC Curve

#### 5.6 Explainability

* Integrated **SHAP** to visualize feature importance
* Helped identify critical keywords influencing classification (e.g., emotionally charged words, repeated phrases)

---

### 6. Results

| Metric    | Score |
| --------- | ----- |
| Accuracy  | 96.2% |
| F1-Score  | 94.7% |
| Precision | 93.8% |
| Recall    | 95.4% |

* **Observation**: Fake news titles often use hyperbole and political bias. Real news maintains neutral language and references.
* SHAP explainability showed that emotional and polarizing keywords played a strong role in classification.

---

### 7. Challenges Faced

* **Imbalanced data**: Required SMOTE and undersampling
* **Noise in text**: Typos, sarcasm, and clickbait headlines
* **Overfitting**: Addressed using dropout layers and validation monitoring
* **Transparency**: Making model decisions explainable to non-technical users

---

### 8. Future Work

* Deploying the model as a Flask API or web extension
* Incorporating **BERT** for deeper contextual understanding
* Extending support for **multilingual news detection**
* Real-time classification of tweets and social media posts
* Fine-tuning with **domain-specific data** (health, finance, politics)
* Adding user feedback loops for reinforcement learning

---

### 9. Conclusion

This project represents a convergence of deep learning, linguistics, and social responsibility. By building a fake news detection system, I have not only sharpened my technical skills in Python, TensorFlow, and NLP but also contributed to the broader goal of digital media integrity.

The journey has been as enriching as the outcome. It reinforced the value of explainable AI, the importance of high-quality data, and the power of reproducible research. I look forward to extending this work to more languages, domains, and platforms.

---

### 10. References

* [Kaggle Fake News Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)
* [SHAP GitHub Repository](https://github.com/slundberg/shap)
* [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
* [GloVe Embeddings](https://nlp.stanford.edu/projects/glove/)
* [Text Preprocessing Techniques](https://www.analyticsvidhya.com/blog/2021/06/text-cleaning-in-nlp-python/)
* [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
* [NLP for Fake News Detection](https://arxiv.org/abs/2006.00198)
* [Explainable Artificial Intelligence (XAI)](https://www.darpa.mil/program/explainable-artificial-intelligence)

---

**Created by**: Sukanya Ghosh
**Repository**: `fake-news-detector`

