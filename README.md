# Object Detection Based Automatic Image Captioning using Deep Learning

**Python Editor:** Spyder

**Dataset:** Flickr8k Dataset; with approximately 8K images and 5 captions per image.

**Deep Learning models:** YOLO (You Only Look Once) for Object Detection, VGG16 for Feature Extraction, and LSTM (Long Short-term Memory) for Caption Generation

This project takes image as an input and generates relevant captions.
The steps of Caption Generation are:
1. Take input image.
2. Detect Objects from the image using YOLO Object Detection technique.
3. Extract features from the image.
4. Preprocess the training captions by tokenizing, padding, processing with start and end sequence to indicate the beginning and end of the sentence.
5. Use Embedding layer to transform the words into vectors.
6. Apply LSTM model to learn the sequence in the caption text.
7. Generate the captions and evaluate the model using BLEU score.
