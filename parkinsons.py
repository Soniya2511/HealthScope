import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
import pickle

# loading the Parkinsons Disease dataset to a pandas DataFrame
df=pd.read_csv('ParkinsonsDisease.csv')

# drop name column
df=df.drop(columns='name', axis=1)
df.head()

# seperating the data and label into X and Y respectively
X=df.drop(columns='status',axis=1)
Y=df['status']


X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2, stratify=Y,random_state=2)

# Building the model using Support Vector Machine classifier
classifier = svm.SVC(kernel='linear')

# training the SVM classifier
classifier.fit(X_train, Y_train)

# Save the trained Logistic Regression model with pickle
pickle.dump(classifier, open('parkinsons.pkl', 'wb'))
  

# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn import svm
# from sklearn.metrics import accuracy_score
# import pickle

# # loading the Parkinson's Disease dataset to a pandas DataFrame
# df = pd.read_csv('ParkinsonsDisease.csv')

# # drop name column
# df = df.drop(columns='name', axis=1)

# # separating the data and label into X and Y respectively
# X = df.drop(columns='status', axis=1)
# Y = df['status']

# # splitting the dataset into training and testing data
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# # Building the model using Support Vector Machine classifier
# classifier = svm.SVC(kernel='linear')

# # training the SVM classifier
# classifier.fit(X_train, Y_train)

# # Calculating accuracy
# train_accuracy = accuracy_score(Y_train, classifier.predict(X_train))
# test_accuracy = accuracy_score(Y_test, classifier.predict(X_test))

# # Print accuracy results
# print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
# print(f"Testing Accuracy: {test_accuracy * 100:.2f}%")

# # Save the trained SVM model with pickle
# pickle.dump(classifier, open('parkinsons.pkl', 'wb'))
