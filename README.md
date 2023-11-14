# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard libraries. 
2.Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
3.Import LabelEncoder and encode the dataset.
4.Import LogisticRegression from sklearn and apply the model on the dataset.
5.Predict the values of array.
6.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
7.Apply new unknown values

## Program:
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: jagadeeshreddy 
RegisterNumber: 212222240059  
```python
import pandas as pd
data=pd.read_csv('/content/Placement_Data (1).csv')
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1) # remove specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size= 0.2,random_state= 0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver = "liblinear") # A Library for Large Linear Classification
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:
Placement Data:

![ml_4 1](https://github.com/jagadeeshreddy561/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120623104/b1acb378-daa7-4301-bbfc-290232371edf)

After Removing Column:

![ml_4 2](https://github.com/jagadeeshreddy561/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120623104/a349b042-2636-4434-8d9f-8c0412bf449f)

Checking the null function():

![ml_4 3](https://github.com/jagadeeshreddy561/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120623104/dde7866d-e991-44e8-9235-5f5f12b999b1)

Data duplicates:

![ml_4 4](https://github.com/jagadeeshreddy561/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120623104/a2b2fd98-b3e8-4b48-898e-11038a8f017e)

Print Data:

![ml_4 5](https://github.com/jagadeeshreddy561/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120623104/927a7d02-100b-4a04-8227-69b99871cb10)

X :

![ml_4 6](https://github.com/jagadeeshreddy561/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120623104/40a807f0-b7c2-4412-8e44-62431c04ce06)

Y :

![ml_4 7](https://github.com/jagadeeshreddy561/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120623104/25737521-f53a-418c-8084-e02c87a894cd)

Y_Prediction Array :

![ml_4 8](https://github.com/jagadeeshreddy561/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120623104/a8a61260-4a00-4b77-b9a4-57ad2f078d33)

Accuracy Value:

![ml_4 9](https://github.com/jagadeeshreddy561/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120623104/4a883b18-08cc-4b70-aac3-95113636287d)

Confusion Matrix;

![ml_4 10](https://github.com/jagadeeshreddy561/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120623104/2790c517-d220-4330-a9d8-edec24fd12dc)

Classification Report:

![ml_4 11](https://github.com/jagadeeshreddy561/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120623104/4b3b3103-9193-4644-89c2-bcb331db6254)

Prediction of LR:

![ml_4 12](https://github.com/jagadeeshreddy561/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120623104/d396b9b0-fc3c-43f6-aea9-bf562a9ee405)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
