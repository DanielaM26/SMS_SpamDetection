# SMS Spam Detection

This repository contains a machine learning project for **SMS spam detection**, implemented using classical **Natural Language Processing (NLP)** techniques and neural network–based classification.  
The goal of the project is to automatically classify SMS messages written in English as **spam** or **ham (non-spam)**.

---

## Project Overview

SMS spam detection is a **binary text classification** problem with important real-world applications, such as filtering unwanted or fraudulent messages.

In this project, two text representation methods are implemented and compared:
- **TF-IDF (sparse representation)**
- **Word2Vec (dense word embeddings)**

Both representations are used as input features for a **neural network classifier implemented in PyTorch**.  
The project focuses on:
- interpretability,
- robustness on an imbalanced dataset,
- and a clear comparison between classical and embedding-based NLP approaches.

---

## Dataset

The project uses the **SMS Spam Collection** dataset from the UCI Machine Learning Repository.

The dataset contains labeled SMS messages:
- `ham` – legitimate messages
- `spam` – unsolicited or fraudulent messages

Each sample consists of:
- raw SMS text (English),
- a binary label (`0 = ham`, `1 = spam`).

The dataset is **imbalanced**, with a significantly larger number of ham messages than spam messages, which motivates the use of multiple evaluation metrics beyond accuracy.

---

## Methodology

### Text Preprocessing
Before vectorization, the SMS messages are preprocessed using standard NLP techniques:
- lowercasing,
- removal of punctuation and special characters,
- stopword removal (NLTK),
- lemmatization using `WordNetLemmatizer`.

---

### Text Representation

Two vectorization methods are implemented:

#### TF-IDF
- unigrams and bigrams,
- sublinear TF scaling,
- filtering of extremely frequent and rare terms.

TF-IDF is well suited for short and structured texts such as SMS messages.

#### Word2Vec
- pretrained word embeddings,
- message-level vectors obtained by averaging word embeddings,
- dense semantic representation of text.

---

### Classification Model

A **feed-forward neural network (MLP)** is implemented using **PyTorch**, consisting of:
- fully connected layers,
- ReLU activations,
- a final output layer for binary classification.

Training is performed using:
- `CrossEntropyLoss`,
- `AdamW` optimizer.

Both **full-batch** and **mini-batch (DataLoader-based)** training strategies are implemented.

---

## Evaluation

The dataset is split into training and testing sets using a **stratified split** to preserve class distribution.

Model performance is evaluated using:
- accuracy,
- precision, recall, and F1-score,
- confusion matrix,
- training and validation loss curves.

Additionally, **manual SMS examples** are used as a sanity check to validate real-world behavior.

---

## Results

Key observations from the experiments:
- **TF-IDF** achieves faster convergence and slightly better performance on this dataset.
- **Word2Vec** captures semantic information but is more sensitive to message length and formulation.
- Both methods achieve high overall accuracy, despite the dataset imbalance.
- TF-IDF proves more stable and effective for short SMS texts.

The results confirm that classical NLP approaches remain highly competitive for SMS spam detection.

---

## Repository Structure
SMS_SpamDetection/
│
├── SMS_Spam_Detection_Project_Final_Baseline.ipynb   # Main notebook (implementation + experiments)
├── README.md                                         # Project description
---

## How to Run

1. Open the notebook in **Google Colab**
2. Mount **Google Drive** and load the dataset
3. Run the cells sequentially to:
   - preprocess data,
   - train the models,
   - evaluate results,
   - test custom SMS messages

---

## Conclusion

This project demonstrates that a well-designed NLP pipeline combining:
- proper text preprocessing,
- effective vectorization (TF-IDF or Word2Vec),
- and a simple neural network classifier

can achieve excellent performance for **SMS spam detection**, even on an imbalanced dataset.  
The comparison highlights the strengths and limitations of sparse versus dense text representations and provides a solid foundation for future extensions, such as transformer-based models.
