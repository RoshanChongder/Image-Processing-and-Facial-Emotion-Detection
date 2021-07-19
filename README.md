# Image-Processing-and-Facial-Emotion-Detection
1. For all the purposes of training, testing and validation, google collaboratory environment is used. 
2. At first, the data set which is collected from Kaggle is loaded from the csv file for training and validation depending on the ‘usage’ tag. 
3. Then the CNN model is built by adding different convolution (2d) and other layers. 
4. As per the extracted data set labeled for training and validation, the built model is trained using hardware acceleration as GPU on a certain number of epochs. 
5. After the training, training and validation accuracy and loss graphs have been plotted. To plot the graph , two separate python files are written .
6. To compute different scores of the trained model ,the confusion matrix is plotted on the prediction of the trained model on some random images collected from google. 
7. At last the trained model and its weights are loaded for testing. Then from a live video stream, frames are captured and fed to the model for prediction and the predicted outputs are shown on the stream.
