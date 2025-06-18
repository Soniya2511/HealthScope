import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# loading the heart dataset to a pandas DataFrame
df=pd.read_csv('heart.csv')

# seperating the data and label into X and Y respectively
X=df.iloc[:,0:13]
Y=df['target']


X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2, stratify=Y,random_state=1)

# Building the model using Logistic Regression
classifier = LogisticRegression()
classifier.fit(X_train, Y_train)

# Save the trained Logistic Regression model with pickle
pickle.dump(classifier, open('heart.pkl', 'wb'))



# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score
# import pickle

# # loading the heart dataset to a pandas DataFrame
# df = pd.read_csv('heart.csv')

# # separating the data and label into X and Y respectively
# X = df.iloc[:, 0:13]
# Y = df['target']

# # splitting the dataset into training and testing data
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=1)

# # Building the model using Logistic Regression
# classifier = LogisticRegression(max_iter=1000)  # Increased max_iter to prevent convergence warnings
# classifier.fit(X_train, Y_train)

# # Calculating accuracy
# train_accuracy = classifier.score(X_train, Y_train)
# test_accuracy = classifier.score(X_test, Y_test)

# print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
# print(f"Testing Accuracy: {test_accuracy * 100:.2f}%")

# # Save the trained Logistic Regression model with pickle
# pickle.dump(classifier, open('heart.pkl', 'wb'))



