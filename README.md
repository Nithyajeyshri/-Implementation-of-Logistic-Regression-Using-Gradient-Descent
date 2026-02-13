# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Start the program.
      
2.Data preprocessing: Cleanse data,handle missing values,encode categorical variables.

3.Model Training: Fit logistic regression model on preprocessed data.

4.Model Evaluation: Assess model performance using metrics like accuracyprecisioon,recall.

5.Prediction: Predict placement status for new student data using trained model.

6.End of program

## Program:
```
import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()
data1=data.copy()
data1.head()
data1=data1.drop(['sl_no','salary'],axis=1)
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
x=data1.iloc[:, : -1]
y=data1["status"]
theta=np.random.randn(x.shape[1])
def sigmoid(z):
    return 1/(1+np.exp(-z))
def loss(theta,x,y):
    h=sigmoid(x.dot(theta))
    return -np.sum(y*np.log(h)+(1-y)*np.log(1-h))
def gradient_descent(theta,x,y,alpha,num_iterations):
    m=len(y)
    for i in range (num_iterations):
        h=sigmoid(x.dot(theta))
        gradient=x.T.dot(h-y)/m
        theta-=alpha*gradient
    return theta
theta=gradient_descent(theta,x,y,alpha=0.01,num_iterations=1000)
def predict(theta,x):
    h=sigmoid(x.dot(theta))
    y_pred=np.where(h>=0.5,1,0)
    return y_pred
y_pred=predict(theta,x)
accuracy=np.mean(y_pred.flatten()==y)
print("Accuracy:",accuracy)
print("Predicted:\n",y_pred)
print("Actual:\n",y.values)
xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print("Prdicted Result:",y_prednew)
```

## Output:
<img width="1407" height="359" alt="Screenshot 2026-02-13 153725" src="https://github.com/user-attachments/assets/948e7e3d-7d6e-4eeb-a114-88af8a9ff7e6" />
<img width="856" height="489" alt="Screenshot 2026-02-13 153629" src="https://github.com/user-attachments/assets/d95e808a-15e4-45b9-809d-9f0aa4e88e00" />
<img width="608" height="138" alt="Screenshot 2026-02-13 153540" src="https://github.com/user-attachments/assets/b62e7ca5-9e63-4195-b059-8b06e9aac54e" />

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

