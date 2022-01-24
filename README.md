# Targeted-Sentiment-Analysis

## Introduction

Target-oriented sentiment classification aims at classifying sentiment polarities over individual opinion targets in a sentence.

## Data and Preprocessing

Our dataset consist of a collection of tweets and reviews. The dataset contains 3 columns 'text', 'target', 'label'.

* `text`: the text of the tweet or review.
* `target` : the targeted sentence of the text.
* `label` : the sentiment polarity of the target towards the text.

### Preprocessing

After reading the dataset, we process the data to remove any new line characters or tabs, lowercase the text, decontraction of certain words, remove digits, puctuations and extra spaces.
Then used Tokenizer to convert the text into a sequence of tokens.

Also we need to get the position of the target in the text and the weight of the each word towards the target. So, we achive that using `get_pw()` function.

## Model

Model I have used is called Transformation Network(TNet) it is adapted from the paper [Transformation Networks for Target-Oriented Sentiment Classification](https://arxiv.org/pdf/1805.01086.pdf)

TNet first encodes the context text information into word embeddings and generates the contextualized word representations with LSTMs. It has a Target Specific Transformation (TST) component for generating the target-specific word representations. TST generates different representations of the target conditioned on the individual context words, then it consolidates each context word with its tailor-made target representation to obtain the transformed word representation. For example in the sentence “I am pleased with the fast log on, and the long battery life”, considering the context word “long” and the target “battery life” in the above example, TST firstly measures the associations between “long” and individual target words. Then it uses the association scores to generate the target representation conditioned on “long”. After that, TST transforms the representation of “long” into a target-specific version with the new target representation. 
As the context information carried by the representations from the LSTM layer will be lost after the non-linear TST, a content preserving mechanism to contextualize the generated target-specific word representations is used. Such a mechanism also allows a deep transformation structure to learn abstract features. To help the CNN feature extractor locate sentiment indicators more accurately, we adopt a proximity strategy to scale the input of the convolutional layer with positional relevance between a word and the target.

![tnet-img](https://i.imgur.com/AdhKwnd.png)

### Model Architecture
![model](https://i.imgur.com/PVC2AwZ.png)

## Training

I have used a pre-trained GloVe for word embedding and the dimension is 300 and for out-of-vocabulary words used a random sample of uniform distribution of range(-0.25, 0.25). Also applied dropouts after LSTMs and the ultimate sentence representation. I have used hidden dims of 50 and kernel size of 3, and softmax as the activation function for the output logits, and Adam with label smoothing of 0.3 and learning rate of 0.001.

## Evaluation

For the evaluation I have used the F1 score and accuracy score. And after 15 epochs, the `loss - 0.94`, `accuracy - 0.74` and `f1_score - 0.72`.

## Deployment

Created a streamlit app for the frond end and the API for the backend using FastAPI is containerized using docker and deployed on a kubernetes cluster.

Note: The app is not live due to lack of free credits.