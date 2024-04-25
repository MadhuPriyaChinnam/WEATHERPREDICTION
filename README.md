# WEATHERPREDICTION
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import VotingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.preprocessing import *

dataset = pd.read_csv("/content/weatherHistory.csv")
dataset.head()

dataset.isnull().sum() #there are 517 null values which should be replaced


dataset["Formatted Date"] = pd.to_datetime(dataset["Formatted Date"], format = "%Y-%m-%d %H:%M:%S.%f %z")

#"loud cover" has only one unique value
dataset = dataset.drop(["Loud Cover","Daily Summary"], axis=1)
# Drop non-numeric columns
numeric_columns = dataset.select_dtypes(include=[np.number]).columns
dataset_numeric = dataset[numeric_columns]


dataset_numeric.corr()

#apparent temperature is highly correlated to temperature and should be removed
dataset = dataset.drop(["Apparent Temperature (C)"], axis=1)
X = dataset

dataset["year"] = dataset["Formatted Date"].apply(lambda x: x.year)
dataset["month"] = dataset["Formatted Date"].apply(lambda x: x.month)
dataset["day"] = dataset["Formatted Date"].apply(lambda x: x.day)
dataset = dataset.drop(["Formatted Date"], axis=1)

#Label Encoding Text Values
le = LabelEncoder()
le.fit(dataset["Summary"])
dataset["Summary"] = le.transform(dataset["Summary"])
le.fit(dataset["Precip Type"])
dataset["Precip Type"] = le.transform(dataset["Precip Type"])
y = dataset["Temperature (C)"]
X = dataset.drop(["Temperature (C)"], axis = 1)
X

#test and train data set split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=40)

print(X_train)

print(y_train)

gbr=GradientBoostingRegressor()
rfr=RandomForestRegressor()
knr=KNeighborsRegressor()
dtr=DecisionTreeRegressor()
svr=SVR()
vr=VotingRegressor([('gbr',gbr),('rfr',rfr),('knr',knr),('dtr',dtr),('svr',svr)])

print(np.isnan(X_train).sum())
print(np.isnan(y_train).sum())

X_train[np.isnan(X_train)] = 0  # Replace NaN values with 0 (you can choose a different strategy if needed)
y_train[np.isnan(y_train)] = 0

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

vr.fit(X_train,y_train)

y_pred = vr.predict(X_test)

from sklearn.metrics import r2_score
r2_score(y_test, y_pred)
from sklearn.metrics import mean_squared_error
np.sqrt(mean_squared_error(y_test,y_pred))

gbr.fit(X_train,y_train)
rfr.fit(X_train,y_train)
knr.fit(X_train,y_train)
dtr.fit(X_train,y_train)
svr.fit(X_train,y_train)

from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(rfr, X_train, y_train, cv=5, scoring='r2')
print("Cross-Validation R-squared scores:", cv_scores)
print("Mean CV R-squared:", np.mean(cv_scores))

v1=r2_score(y_test, gbr.predict(X_test))
v2=r2_score(y_test, rfr.predict(X_test))
v3=r2_score(y_test, knr.predict(X_test))
v4=r2_score(y_test, dtr.predict(X_test))
v5=r2_score(y_test, svr.predict(X_test))
v6=r2_score(y_test, y_pred)
print(v1)
print(v2)
print(v3)
print(v4)
print(v5)
print(v6)


model_names = ['GradientBoosting', 'RandomForest', 'KNeighbors', 'DecisionTree', 'SVR', 'Voting']
r2_values = [v1, v2, v3, v4, v5, v6]
rmse_values = [sqrt(mean_squared_error(y_test, model.predict(X_test))) for model in [gbr, rfr, knr, dtr, svr, vr]]
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(14, 6))
ax[0].bar(model_names, r2_values)
ax[0].set_title('R-squared Comparison')
ax[0].set_ylabel('R-squared')
ax[1].bar(model_names, rmse_values)
ax[1].set_title('RMSE Comparison')
ax[1].set_ylabel('RMSE')
plt.tight_layout()
plt.show()

from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import VotingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Define base models
base_models = [
    ('gbr', GradientBoostingRegressor()),
    ('rfr', RandomForestRegressor()),
    ('knr', KNeighborsRegressor()),
    ('dtr', DecisionTreeRegressor()),
    ('svr', SVR())
]

# Initialize stacking regressor
stacking_reg = StackingRegressor(estimators=base_models, final_estimator=LinearRegression())

# Train stacking regressor
stacking_reg.fit(X_train, y_train)

# Make predictions
y_pred_stacking = stacking_reg.predict(X_test)
print(y_pred_stacking)

# Evaluate the performance
r2_stacking = r2_score(y_test, y_pred_stacking)
rmse_stacking = np.sqrt(mean_squared_error(y_test, y_pred_stacking))
print("Stacking R-squared:", r2_stacking)
print("Stacking RMSE:", rmse_stacking)
