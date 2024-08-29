# EX No:02 Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: SHARUKESH R
RegisterNumber: 212223220106 
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)


```

## Output:
### Dataset
![Screenshot 2024-08-29 154100](https://github.com/user-attachments/assets/31d1c23a-c19a-42cf-9ee6-cd7973142167)

### Head values
![Screenshot 2024-08-29 154107](https://github.com/user-attachments/assets/5d3b6382-9379-4c40-8134-6a8cb24e1bd4)

### Tail values
![Screenshot 2024-08-29 154113](https://github.com/user-attachments/assets/a642ebb1-b514-4a06-b67a-6e856ddb71f5)

### X and Y values 
![Screenshot 2024-08-29 154134](https://github.com/user-attachments/assets/0bc91d8c-35a8-4e6f-9a03-fec61e850b63)

### Predication values of X and Y
![Screenshot 2024-08-29 154140](https://github.com/user-attachments/assets/db17caa7-3281-4fd2-b78a-ebe4672d80e4)

### MSE,MAE and RMSE
![Screenshot 2024-08-29 154147](https://github.com/user-attachments/assets/bea780e2-2533-495d-8d25-f45fd4c660c3)

### Training set
![Screenshot 2024-08-29 155420](https://github.com/user-attachments/assets/654ffdc0-e437-4dfc-85be-d06dc0513de6)

### Testing set
![Screenshot 2024-08-29 155501](https://github.com/user-attachments/assets/c83ab099-531e-42b2-bab4-eabbfce6ba4c)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
