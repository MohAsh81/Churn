# Churn
This code performs customer churn prediction using a neural network. It begins by loading a dataset from a CSV file containing customer information. After preprocessing the data—such as handling missing values, encoding categorical variables, and normalizing numerical features—the dataset is split into training and testing sets.

A simple neural network model is built using TensorFlow's Keras API, with two hidden layers. The model is compiled with the Adam optimizer and binary cross-entropy loss function, and it is trained on the training set for 100 epochs.

Once trained, the model makes predictions on the test set, and a confusion matrix is generated to evaluate the model's performance visually, showing the counts of true positive, true negative, false positive, and false negative predictions.
