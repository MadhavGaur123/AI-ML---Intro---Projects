import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

print("OPENING AND READING THE CSV FILE")
data = pd.read_csv(r"C:\Users\gaurm\Downloads\dataset.csv", encoding="latin-1")


Y = np.array(data["target"])

features = ["chol","oldpeak","thalach"]
X = np.array(data[features])


Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=42)
machine = DecisionTreeClassifier()
model = RandomForestClassifier(n_estimators=5,max_features=1,random_state=42)
machine.fit(Xtrain, Ytrain)
model.fit(Xtrain,Ytrain)
Ypredicted = machine.predict(Xtest)
Ypredictd2 = model.predict(Xtest)

cm = confusion_matrix(Ytest, Ypredicted)
CM = confusion_matrix(Ytest,Ypredictd2)
print("Confusion Matrix:")
print(cm)
print(CM)
True_Negative = cm[0][0]
True_Positive = cm[1][1]
False_Positive = cm[0][1]
False_Negative = cm[1][0]
True_Negative2 = CM[0][0]
True_Positive2 = CM[1][1]
False_Positive2 = CM[0][1]
False_Negative2 = CM[1][0]
Precision = True_Positive/(True_Positive + False_Positive )
Recall = True_Positive/(True_Positive + False_Negative)
Precision2 = True_Positive2/(True_Positive2 + False_Positive2 )
Recall2 = True_Positive2/(True_Positive2 + False_Negative2)
print(Recall)
print(Precision)
print(Recall2)
print(Precision2)
probabilities_tree = machine.predict_proba(Xtest)[:, 1]
probabilities_forest = model.predict_proba(Xtest)[:, 1]
ans = True
# while(ans == True):
#     chlorestol = int(input())
#     oldpeak = float(input())
#     thalach = int(input())
#     value = model.predict([[chlorestol,oldpeak,thalach]])
#     print(value)
#     z = str(input("Do you want to enter more: "))
#     if(z == "y"):
#         continue
#     else:
#         ans = False


fpr_tree, tpr_tree, thresholds_tree = roc_curve(Ytest, probabilities_tree)
auc_tree = auc(fpr_tree, tpr_tree)


fpr_forest, tpr_forest, thresholds_forest = roc_curve(Ytest, probabilities_forest)
auc_forest = auc(fpr_forest, tpr_forest)


plt.figure(figsize=(10, 6))
plt.plot(fpr_tree, tpr_tree, color='blue', label=f'Decision Tree (AUC = {auc_tree:.2f})')
plt.plot(fpr_forest, tpr_forest, color='green', label=f'Random Forest (AUC = {auc_forest:.2f})')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')

plt.show()


