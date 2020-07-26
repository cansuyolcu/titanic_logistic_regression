# titanic_logisticregression
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
