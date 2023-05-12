# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries such as pandas module to read the corresponding csv file.
2. Upload the dataset values and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
3. Import LabelEncoder and encode the corresponding dataset values.
4. Import LogisticRegression from sklearn and apply the model on the dataset using train and test values of x and y and Predict the values of array using the variable y_pred.
5. Calculate the accuracy, confusion and the classification report by importing the required modules such as accuracy_score, confusion_matrix and the classification_report from sklearn.metrics module.
6. Apply new unknown values and print all the acqirred values for accuracy, confusion and the classification report.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Kavya k
RegisterNumber: 212222230065

import pandas as pd
data=pd.read_csv('Placement_Data.csv')
data.head()


data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis = 1)#removes the specified row or column
data1.head()


data1.isnull().sum()


data1.duplicated().sum()


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1


x=data1.iloc[:,:-1]
x


y=data1["status"]
y


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size = 0.2,random_state = 0)


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear")# a library for large
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
(accuracy)


from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
(confusion)


from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)


lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
*/
```

## Output:
1.Placement data
![image](https://github.com/kavyasenthamarai/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118668727/69b03db1-19e6-4d51-a9df-99b5b9542b23)

2.Salary data
![image](https://github.com/kavyasenthamarai/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118668727/08f2348a-5398-4ef0-ae49-4f64bba605be)

3.Checking the null() function
![image](https://github.com/kavyasenthamarai/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118668727/4756c165-4eda-461c-8ceb-a1743936df0b)

4.Data Duplicate
![image](https://github.com/kavyasenthamarai/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118668727/4c7fe663-0bee-4de4-8c46-4c6b2ac8075c)

5.Print data
![image](https://github.com/kavyasenthamarai/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118668727/1e61a092-de9d-4b2c-8fef-16b4f0667164)

6.Data-status
![image](https://github.com/kavyasenthamarai/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118668727/46bf6c47-4664-45c7-8f17-3dd410da507e)

7.y_prediction array
![image](https://github.com/kavyasenthamarai/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118668727/2c958851-e118-4c0a-860e-ef6ac8f546d3)

8.Accuracy value
![image](https://github.com/kavyasenthamarai/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118668727/b655d904-5736-45d7-9f56-fcf6d7dd624c)

9. Confusion array
![image](https://github.com/kavyasenthamarai/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118668727/cb00184e-fef0-4897-a7f6-d8125c0a4efc)

10. Classification report
![image](https://github.com/kavyasenthamarai/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118668727/cd1f57cc-50f5-44a9-b72d-ee7f4509c35f)

11.Prediction of LR
![image](https://github.com/kavyasenthamarai/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118668727/f4dad4dd-07e4-402a-96f3-0ed17d631943)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
