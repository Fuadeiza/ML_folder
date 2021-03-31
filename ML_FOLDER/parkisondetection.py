import pandas as pd 
import numpy as np 
import xgboost as xgb 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier 
from sklearn.preprocessing import MinMaxScaler


parkison_df = pd.read_csv(r'C:\Users\DELL\Desktop\NEWS\parkinsons.data')

# print(parkison_df.head())

# x_val = parkison_df.iloc[:, parkison_df.columns != 'status']
x_val = parkison_df.drop(['status', 'name'], axis =1)
y_val = parkison_df['status']
# print(y_val.head())

scale = MinMaxScaler((-1,1))

x_value = scale.fit_transform(x_val)

X_train, X_test, y_train , y_test = train_test_split(x_value, y_val, test_size =0.2, random_state=7)

def clf(model):
    clf = model 
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {round(score*100,2)}%')


clf(XGBClassifier())

