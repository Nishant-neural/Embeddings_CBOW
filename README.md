# Embedding Project

This project implements word embeddings using the Continuous Bag of Words (CBOW) model and a sentiment classifier built on top of the learned embeddings.

## Project Structure

- `data.py`: Data preparation utilities for CBOW training
- `preprocess.py`: Text preprocessing and tokenization functions
- `cbow_model.py`: CBOW model implementation
- `classifier.py`: Sentiment classification model
- `train_cbow.py`: Script to train the CBOW model
- `train_classifier.py`: Script to train the sentiment classifier
- `utils.py`: Utility functions for similarity search
- `requirements.txt`: Python dependencies

## Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Train the CBOW model:
   ```bash
   python train_cbow.py
   ```

2. Train the sentiment classifier:
   ```bash
   python train_classifier.py
   ```

## Description

The CBOW model learns word embeddings by predicting a target word from its context words. The trained embeddings can then be used for downstream tasks like sentiment classification.
