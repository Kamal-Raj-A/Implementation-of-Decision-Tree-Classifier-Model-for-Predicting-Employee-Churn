# EXP 6 -Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1.Import the required packages.

2.Read the data set.

3.Apply label encoder to the non-numerical column inoreder to convert into numerical values.

4.Determine training and test data set.

5.Apply decision tree Classifier and get the values of accuracy and data prediction.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: KAMAL RAJ A
RegisterNumber:  212223040082
*/
import pandas as pd
data=pd.read_csv("Employee.csv")

data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])

```

## Output:
## data.head():
![image](https://github.com/Kamal-Raj-A/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/145742556/917a031d-ecb4-4312-bf19-c191cc296498)

## data.info():
![Screenshot 2024-04-02 090004](https://github.com/Kamal-Raj-A/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/145742556/ea8069a2-b5a6-4b26-89fa-a7e708d1b896)

## data.isnull().sum():
![image](https://github.com/Kamal-Raj-A/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/145742556/b7775f17-b96a-4a0c-834a-81bb561e6158)

## data["left"].value_counts():
![image](https://github.com/Kamal-Raj-A/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/145742556/17b1c4d5-938c-40e3-b9ce-df0e3099ef0c)

## data.head():
![image](https://github.com/Kamal-Raj-A/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/145742556/d61181a3-d031-4800-bfde-e7f38be1bf1a)

## x.head():
![image](https://github.com/Kamal-Raj-A/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/145742556/b659d96e-16be-4338-9ccd-558ec900d68f)

## accuracy :
![image](https://github.com/Kamal-Raj-A/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/145742556/923e71e0-ed0c-4204-a078-d3d0015115b8)

## dt.predict([[0.5,0.8,9,260,6,0,1,2]]):
![image](https://github.com/Kamal-Raj-A/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/145742556/2aa4cebf-e692-418c-9e21-9564228f223a)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
