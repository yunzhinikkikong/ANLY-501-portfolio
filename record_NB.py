#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


# ### Data Preparation

# In[7]:


TrainDF = pd.read_csv("/Users/nikkkikong/Desktop/501 Assignment6 /Foodborne_NBtrain.csv")
TrainDF["Etiology"] = pd.Categorical(TrainDF["Etiology"])
# There should be only numeric varaibles when fitting NB and SVM model in Python
TrainDF = TrainDF.drop(["setting"], axis=1)
TrainDF = TrainDF.drop(["Etiology.Status"], axis=1)

TestDF = pd.read_csv("/Users/nikkkikong/Desktop/501 Assignment6 /Foodborne_NBtest.csv")
TestDF["Etiology"] = pd.Categorical(TestDF["Etiology"])
# There should be only numeric varaibles when fitting NB and SVM model in Python
TestDF = TestDF.drop(["setting"], axis=1)
TestDF = TestDF.drop(["Etiology.Status"], axis=1)

## Save labels
TestLabels=TestDF["Etiology"]
## remove labels
## Make a copy of TestDF
CopyTestDF=TestDF.copy()
TestDF = TestDF.drop(["Etiology"], axis=1)
## DF seperate TRAIN SET from the labels
TrainDF_nolabels=TrainDF.drop(["Etiology"], axis=1)
#print(TrainDF_nolabels)
TrainLabels=TrainDF["Etiology"]


# In[8]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.YlOrBr):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import itertools
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()


from sklearn.metrics import classification_report as clsr
def evaluate_model(model, X_test_data, y_test_labels):
    y_predicted_labels = model.predict(X_test_data)
    print(clsr(y_test_labels, y_predicted_labels))


# ### Multinomial Naive Bayes Classifier

# In[9]:


from sklearn.naive_bayes import MultinomialNB
MyModelNB= MultinomialNB()
MyModelNB.fit(TrainDF_nolabels, TrainLabels)
Prediction = MyModelNB.predict(TestDF)
print("\nThe prediction from NB is:")
print(Prediction)
print("\nThe actual labels are:")
print(TestLabels)
## confusion matrix
from sklearn.metrics import confusion_matrix
cnf_matrix = confusion_matrix(TestLabels, Prediction,labels=['Other', 'Escherichia' ,'Salmonella' ,'Norovirus'])
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Other', 'Escherichia' ,'Salmonella' ,'Norovirus'],
                      title='Confusion matrix for Multinomial Naive Bayes Classifier')
print("\nClassification Report:\n")
evaluate_model(MyModelNB, TestDF, TestLabels)


# ### Bernoulli Naive Bayes Classifier

# In[10]:


from sklearn.naive_bayes import BernoulliNB
## Bernoulli uses 0 and 1 data (not counts)
## So - we need to re-format our data first
## Make a COPY of the DF
TrainDF_nolabels_Binary=TrainDF_nolabels.copy()   ## USE .copy()
TrainDF_nolabels_Binary[TrainDF_nolabels_Binary >= 1] = 1
TrainDF_nolabels_Binary[TrainDF_nolabels_Binary < 1] = 0
BernModel = BernoulliNB()
BernModel.fit(TrainDF_nolabels_Binary, TrainLabels)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
print("\nBernoulli prediction:\n", BernModel.predict(TestDF))
print("\nActual:")
print(TestLabels)
bn_matrix = confusion_matrix(TestLabels, BernModel.predict(TestDF),labels=['Other', 'Escherichia' ,'Salmonella' ,'Norovirus'])
plt.figure()
plot_confusion_matrix(bn_matrix, classes=['Other', 'Escherichia' ,'Salmonella' ,'Norovirus'],
                      title='Confusion matrix for Bernoulli Naive Bayes Classifier')
print("\nClassification Report:\n")
evaluate_model(BernModel, TestDF, TestLabels)


# ### SVM Classifier with a Linear Kernel

# In[13]:


from sklearn.svm import LinearSVC
SVM_Model=LinearSVC(C=.1)
SVM_Model.fit(TrainDF_nolabels, TrainLabels)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
print("SVM prediction:\n", SVM_Model.predict(TestDF))
print("Actual:")
print(TestLabels)

SVM_matrix = confusion_matrix(TestLabels, SVM_Model.predict(TestDF),labels=['Other', 'Escherichia' ,'Salmonella' ,'Norovirus'])
plt.figure()
plot_confusion_matrix(SVM_matrix, classes=['Other', 'Escherichia' ,'Salmonella' ,'Norovirus'],
                      title='Confusion matrix for SVM Classifier with a Linear Kernel')
print("\nClassification Report:\n")
evaluate_model(SVM_Model, TestDF, TestLabels)                                    


# ### SVM Classifier with a Polynomial Kernel

# In[17]:


from sklearn.svm import SVC
SVM_Model=sklearn.svm.SVC(C=0.2, kernel='poly',degree=2,
                           gamma="auto", verbose=True)
SVM_Model.fit(TrainDF_nolabels, TrainLabels)
print("SVM prediction:\n", SVM_Model.predict(TestDF))
print("Actual:")
print(TestLabels)

SVM_matrix = confusion_matrix(TestLabels, SVM_Model.predict(TestDF),labels=['Other', 'Escherichia' ,'Salmonella' ,'Norovirus'])
plt.figure()
plot_confusion_matrix(SVM_matrix, classes=['Other', 'Escherichia' ,'Salmonella' ,'Norovirus'],
                      title='Confusion matrix for SVM Classifier with a Poly Kernel')
print("\nClassification Report:\n")
evaluate_model(SVM_Model, TestDF, TestLabels)    


# ### SVM Classifier with a Radial Kernel

# In[25]:


from sklearn.svm import SVC
SVM_Model=sklearn.svm.SVC(C=0.5, kernel='rbf', 
                           verbose=True, gamma="auto")
SVM_Model.fit(TrainDF_nolabels, TrainLabels)
print("SVM prediction:\n", SVM_Model.predict(TestDF))
print("Actual:")
print(TestLabels)

SVM_matrix = confusion_matrix(TestLabels, SVM_Model.predict(TestDF),labels=['Other', 'Escherichia' ,'Salmonella' ,'Norovirus'])
plt.figure()
plot_confusion_matrix(SVM_matrix, classes=['Other', 'Escherichia' ,'Salmonella' ,'Norovirus'],
                      title='Confusion matrix for SVM Classifier with a Radial Kernel')

print("\nClassification Report:\n")
evaluate_model(SVM_Model, TestDF, TestLabels)    


# In[ ]:




