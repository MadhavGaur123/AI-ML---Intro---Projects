import pandas as pd
import numpy as np
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
print("OPENING and LOADING The CSV file")
data = pd.read_csv(r"C:\Users\gaurm\Downloads\loans.csv",encoding="latin-1")
Y = np.array(data["Loan_Status"])
dict = {"Male":0,"Female":1}
dictt = {"No":0,"Yes":1}
dicttt = {"Graduate":1,"Not Graduate":0}
data["Gender"] = data["Gender"].map(dict)
data["Married"] = data["Married"].map(dictt)
data["Education"] = data["Education"].map(dicttt)
features = ["LoanAmount"]
X = np.array(data[features])
Xtrain,Xtest,Ytrain,Ytest = train_test_split(X,Y,test_size=0.2,random_state=42) #random state set to 42 to ensure reproducibility
print("Setting up the Decision Tree Model and Using SK learn Module to train it")
model = DecisionTreeClassifier()
model.fit(Xtrain,Ytrain)
# tree.plot_tree(model,feature_names=features)
# plt.show()
Ypredicted = model.predict_proba(Xtest)[:,1]
print(Ypredicted)
TPRs = []
FPRs = []
for i in range(5):
    Predictions = []
    t = float(input())
    for i in Ypredicted:
        if i<=t:
            Predictions.append("N")
        else:
            Predictions.append("Y")
    cm = confusion_matrix(Ytest,Predictions)
    True_negative = cm[0][0]
    False_positive = cm[0][1]
    False_Negative = cm[1][0]
    True_Positive = cm[1][1]
    if True_Positive + False_positive == 0:
        Precision = 0
    else:
        Precision = True_Positive / (True_Positive + False_positive)
    
    Recall = True_Positive / (True_Positive + False_Negative) if (True_Positive + False_Negative) > 0 else 0
    Specificity = True_negative / (True_negative + False_positive) if (True_negative + False_positive) > 0 else 0
    
    TPR = Recall
    FPR = 1 - Specificity
    
    TPRs.append(TPR)
    FPRs.append(FPR)

print(TPRs)
print(FPRs)
plt.plot(FPRs, TPRs, color='blue', alpha=0.7)
plt.show()


