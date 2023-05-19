# ImageCaptionGenrator
In this Project we are going to create a deep learning models using pretrained models, CNN, RNN etc. with the help of this model we can generate the caption for any image.Caption generation is a challenging artificial intelligence problem where a textual description must be generated for a given photograph.

# Dataset

We have used Flickr8k dataset for this project. It contains around 8091 images and 5 captions for each image (5 * 8091 = 40455 captions)
Dataset Source: KAGGLE LINK https://www.kaggle.com/shadabhussain/flickr8k
To access complete code with vgg16 weights and others weights refer:https://huggingface.co/kshitij10000/image-caption-genrator-using-flicker-and-tkinter/tree/main

# Model Evaluation Matrix

BLEU metric is an evaluation matrix that compares a generater sentence with a reference sentence. It compares n-grams (1-gram means one word) and a perfect match and a perfect mismatch results in a score of 1.0 and 0.0 respectively.

Data Preparation

We create a new dataframe called as dfword to visualize distribution of the words. It contains each word and its frequency in the entire tokens in decreasing order.

# image cleaning

Then we find the top 50 most and least frequently appearing words. Here, I found that the stopwords (like a, the) and punctuations are most occuring, we have to remove these from our dataset in order to clean it. I have implemented 3 functions to clean the captions:
remove_punctuation(text)
remove_single_character(text)
remove_numeric(text)
Now we'll be adding start and end tokens in every caption ('startseq ' & ' endseq')

# Image Preparation for VGG16 Model

We will be using pre-trained network VGG16.
We will save that file asvgg16_weights.h5 and then load it for future process.
VGG16 model takes input image of size (224, 224, 3) process it with DEEP CNN layers and output a tensor of shape (?, 1000) i.e. 1000 classes We have removed the last layer from the network as we only want the features (?, 4096).
After this we'll reshape our images to (224, 224, 3) and feed it to the model to get features respective to each image.
Tokenizer We'll now convert our captions into tokens eg (startseq = 1, endseq = 2 etc.) and store these into an array.

# Model Training

Splitting the dataset We'll split the dataset (all 3 datasets i.e. captions_data, images_data and filenames_data) in ratio of 0.6 : 0.2 : 0.2 (train:valid:test) For the captions_data we have to do the padding since not all token_arrays are of same length, so we take the maximum length and do padding for other token_arrays.

Model

The input to the model will be image-features of shape 4096.
First we got 256 unit outputs from images using Dense layer and we used the Embedding and LSTM layer to get 256 unit output from captions.
It uses Categorical cross entropy loss function & adam optimizer.
Then we train the model with our dataset.

# Graphical User Interface

We have used tkinter library and PIL (Python Image Library) to make a GUI with Upload button to upload the images and Classify Image button to get generated caption.

# Performance analysis

The performance of the image caption generator using LSTM, VGG16, Flicker8k dataset, and Tkinter was thoroughly analyzed. The model architecture effectively combined VGG16 for image feature extraction with LSTM for caption generation. The model was tested on the dataset, and both qualitative and quantitative analyses were conducted. The generated captions were visually inspected, and their coherence and accuracy were evaluated. Additionally, the evaluation metrics were calculated to provide a quantitative measure of performance. Based on the analysis, potential areas for further improvement were identified to enhance the overall performance of the model.
