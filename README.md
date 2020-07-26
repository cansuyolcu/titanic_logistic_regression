# titanic_logistic_regression
In this project I worked with the [ Titanic Data Set from Kaggle](https://www.kaggle.com/c/titanic).
I tried to predict a classification- survival or deceased implementing Logistic Regression in Python for classification.
## Imports

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```

## The Data

```python
train = pd.read_csv('titanic_train.csv')
train.head()
```
<img src= "https://user-images.githubusercontent.com/66487971/88484963-3c6e2580-cf7b-11ea-89d8-36603e6cd48a.png" width = 1000>

## Missing Data
I use seaborn to create a simple heatmap to see where the missing data is.
```python
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
```
<img src= "https://user-images.githubusercontent.com/66487971/88485040-b8686d80-cf7b-11ea-902c-b0b164980c52.png" width = 400>
Roughly 20 percent of the Age data is missing. The proportion of Age missing is likely small enough for reasonable replacement with some form of imputation. Looking at the Cabin column, it looks like there are too much missing data to do something useful with at a basic level. I'll probably drop this later.

## Visualizing

```python
sns.set_style('whitegrid')
sns.countplot(x='Survived',data=train,palette='RdBu_r')
```
<img src= "https://user-images.githubusercontent.com/66487971/88485112-447a9500-cf7c-11ea-8a5b-d3e6a63b9445.png" width = 450>

```python
sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Sex',data=train,palette='RdBu_r')
```
<img src= "https://user-images.githubusercontent.com/66487971/88485156-902d3e80-cf7c-11ea-9a60-5b3d48da752b.png" width = 450>

```python
sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass',data=train,palette='rainbow')
```
<img src= "https://user-images.githubusercontent.com/66487971/88485191-b81ca200-cf7c-11ea-8b27-833b79393d3b.png" width = 450>

```python

sns.distplot(train['Age'].dropna(),kde=False,color='darkred',bins=30)
```
<img src= "https://user-images.githubusercontent.com/66487971/88485219-f3b76c00-cf7c-11ea-8f7c-719b241df07e.png" width = 450>

```python
sns.countplot(x='SibSp',data=train)
```
<img src= "https://user-images.githubusercontent.com/66487971/88485258-324d2680-cf7d-11ea-9652-9b81c7426cd4.png" width = 450>

```python
train['Fare'].hist(color='green',bins=40,figsize=(8,4))
```
<img src= "https://user-images.githubusercontent.com/66487971/88485271-53157c00-cf7d-11ea-86ff-e914dddc3a06.png" width = 450>

## Data Cleaning

I filled the missing ages with the mean age of all the passengers in that class .

```python
plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass',y='Age',data=train,palette='winter')
```
<img src= "(https://user-images.githubusercontent.com/66487971/88485392-0b432480-cf7e-11ea-9132-3e78a159f78f.png" width = 1000>

It can see the wealthier passengers in the higher classes tend to be older, which makes sense. I use these average age values to impute based on Pclass for Age.


```python 

def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age
        
```

Filling the gaps.

```python

train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)

```

Now I check the heatmap again.

```python

sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')

```

<img src= "https://user-images.githubusercontent.com/66487971/88485612-c0c2a780-cf7f-11ea-9033-f4b9ba418df1.png" width = 450>
  

Now I drop the Cabin Column.

```python

train.drop('Cabin',axis=1,inplace=True)
train.dropna(inplace=True)

```

## Converting Categorical Features 

I convert categorical features to dummy variables using pandas.

```python
train.info()
```

<img src= "https://user-images.githubusercontent.com/66487971/88485668-40507680-cf80-11ea-8b80-440d2a92addc.png" width = 350>

```python
sex = pd.get_dummies(train['Sex'],drop_first=True)
embark = pd.get_dummies(train['Embarked'],drop_first=True)

train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)

train = pd.concat([train,sex,embark],axis=1)

train.head()

```

<img src= "https://user-images.githubusercontent.com/66487971/88485694-7130ab80-cf80-11ea-8114-9d63fc173445.png" width = 500>

## Building a Logistic Regression model

I wanted to evaluate my classification so I split tge data into a training set and test set.

# Train Test Split


```python

from sklearn.model_selection import train_test_split

X_train=(train.drop('Survived',axis=1)
y_test=train['Survived']

```

# Training and Predicting

```python

from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)

```

# Evaluation

```python

from sklearn.metrics import classification_report

print(classification_report(y_test,predictions))

```

<img src= "https://user-images.githubusercontent.com/66487971/88485772-4004ab00-cf81-11ea-96ff-ad96e8df17fc.png" width = 450>

Not bad for a training data. Using the real test data on Kaggle gave 0.75 accuracy. After learning other methods I'll try to improve my score.

# This is the end of my project. Thanks for reading all the way through.
