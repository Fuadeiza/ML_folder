import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns


train_df = pd.read_csv(r'C:\Users\DELL\Downloads\house-prices-advanced-regression-techniques\train.csv')
test_df = pd.read_csv(r'C:\Users\DELL\Downloads\house-prices-advanced-regression-techniques\test.csv')

train_id= train_df.pop('Id')

numerical_columns= train_df.select_dtypes(exclude=['object'])

# print(len(numerical_columns.columns[0]))

# print(numerical_columns.head())

# for i in numerical_columns.columns:
    # , len(train_df[i]))


x= numerical_columns.iloc[:, :-1]
y=numerical_columns.iloc[:, -1]

# print(x)
# print(y)

plt.figure(figsize=(15,5))
sns.heatmap(numerical_columns.isnull(),yticklabels=0,cbar=False,cmap='viridis')

plt.show()