# -*- coding: utf-8 -*-
"""
Created on Fri May 29 14:25:36 2020

@author: CONNIETANG
"""


# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from pandas.plotting import scatter_matrix
from sklearn import preprocessing
from sklearn import datasets
import numpy as np; np.random.seed(0)
import statsmodels.api as sm
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import make_classification
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import seaborn as sns; sns.set()
sns.set(style="ticks", color_codes=True)
from math import sqrt
import math
from sklearn.impute import SimpleImputer
from sklearn.datasets import make_regression


###################### Data Preparation ################################
## step 1  import CSV file
train = pd.read_csv(r"C:\Users\BIZtech\Desktop\train.csv")
train.head()
df = pd.read_csv(r"C:\Users\BIZtech\Desktop\train.csv")
train.dtypes

##  Step 2 Generate data overview for further analysis
print(train.head(20))
print(train.describe())
print("Rows     : ", train.shape[0])
print("Columns  : ", train.shape[1])
print(train.info())
print("\nFeatures : \n", train.columns.tolist())
print("\nMissing values :  ", train.isnull().sum().values.sum())
print("\nUnique values :  \n", train.nunique())
total_rows=len(df.axes[0])
total_cols=len(df.axes[1])
print("Number of Rows: "+str(total_rows))
print("Number of Columns: "+str(total_cols))
## Target variable Distribution, Skewness and Kurtosis
sns.distplot(train['SalePrice']);
#skewness and kurtosis
print("Skewness: %f" % train['SalePrice'].skew())
print("Kurtosis: %f" % train['SalePrice'].kurt())

## step 2b Visualization:Correlation heatmap & pairplot & Catplot
sns.pairplot(df)
plt.figure(figsize=(40,40))
sns.heatmap(df.corr(), annot = True,fmt='.1g',vmin=-1, vmax=1, center= 0,square=True)
cat_cols = ['MSZoning', 'Street',' Alley', 'LotShape', 'LandContour', 'Utilities',
            'LotConfig', 'LandSlope', 'Neighborhood',
            'Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofStyle','Exterior1st',
            'Exterior2nd','MasVnrType','ExterQual','ExterCond'
            'Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1',
            'BsmtFinType2', 'Heating','HeatingQC','CentralAir'
            'Electrical','KitchenQual','Functional','FireplaceQu','GarageType',
            'GarageFinish','GarageQual','GarageCond','PavedDrive'
            'PoolQC','Fence','MiscFeature','SaleType','SaleCondition']

sns.catplot(x="Street", y="SalePrice", data=train)
sns.catplot(x="Alley", y="SalePrice", data=train)
sns.catplot(x="LotShape", y="SalePrice", data=train)
sns.catplot(x="LandContour", y="SalePrice", data=train)
sns.catplot(x="Utilities", y="SalePrice", data=train)
sns.catplot(x="LotConfig", y="SalePrice", data=train)
sns.catplot(x="LandSlope", y="SalePrice", data=train)
sns.catplot(x="Neighborhood", y="SalePrice", data=train)
sns.catplot(x="Condition1", y="SalePrice", data=train)
sns.catplot(x="Condition2", y="SalePrice", data=train)
sns.catplot(x="BldgType", y="SalePrice", data=train)
sns.catplot(x="Exterior2nd", y="SalePrice", data=train)
sns.catplot(x="RoofStyle", y="SalePrice", data=train)
sns.catplot(x="RoofMatl", y="SalePrice", data=train)


############################## Data Preprocessing ##########################################
## Step 3 Handling missing values
Id_col = ['id']
target_col = ["SalePrice"]
total = train.isnull().sum().sort_values(ascending=False)
percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
f, ax = plt.subplots(figsize=(15, 6))
plt.xticks(rotation='90')
sns.barplot(x=missing_data.index, y=missing_data['Percent'])
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)
missing_data.head()
train.fillna(train.mean(), inplace=True)
train.isnull().sum().values.sum()
train.fillna("000",inplace=True)
train.isnull().sum().values.sum()
   
## Step 4 Label encoding categorical features
obj_df_train = df.select_dtypes(include=['object']).copy()
obj_df_train.head()
le=LabelEncoder()
for i in obj_df_train:
    train[i]=le.fit_transform(train[i])
data_standardized = preprocessing.scale(train)

## Step 5 Scaling Numerical columns
num_cols = [x for x in train.columns if x not in target_col + Id_col]
std = StandardScaler()
scaled = std.fit_transform(train[num_cols])
scaled = pd.DataFrame(scaled, columns=num_cols)
# dropping original values merging scaled values for numerical columns
df_train_og = train.copy()
train = train.drop(columns=num_cols, axis=1)
train = train.merge(scaled, left_index=True, right_index=True, how="left")
summary = (df_train_og[[i for i in df_train_og.columns if i not in Id_col]].
           describe().transpose().reset_index())
summary = summary.rename(columns={"index": "feature"})
summary = np.around(summary, 3)
val_lst = [summary['feature'], summary['count'],
           summary['mean'], summary['std'],
           summary['min'], summary['25%'],
           summary['50%'], summary['75%'], summary['max']]
print (summary)

################## Identify the important features for regression model ##############

## Step 6a Define dependent and independent variables
cols = [i for i in train.columns if i not in Id_col + target_col]
train_x = train[cols]
train_y = train[target_col]


## Step 6b Build a classification task using n informative features
X = train_x
y = train_y
X, y = make_classification(n_samples=10000, n_features=80, n_informative=40,
                           n_redundant=2, n_repeated=0, n_classes=8,
                           n_clusters_per_class=1, random_state=0)
## Step 7a Create the RFE object and compute a cross-validated score.
svc = SVC(kernel="linear")
rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(2),
              scoring='accuracy')
rfecv.fit(X, y)
print("Optimal number of features : %d" % rfecv.n_features_)

## Step 7b Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()

## Step 8a create the RFE model for the svm classifier and select attributes
svm = LinearSVC()
rfe = RFE(svm, 48)
rfe = rfe.fit(train_x, train_y)
## Step 8b print summaries for the selection of attributes
print(rfe.support_)
print(rfe.ranking_)
print(np.where(rfecv.support_ == False)[0])
## Step 8c Dropping the least important features
train_x.drop(train_x.columns[np.where(rfecv.support_ == False)[0]], axis=1, inplace=True)
print (train_x.columns.values)
################################ applying model ##############################

## Step 9 Split the data into training/testing sets
train_X,test_X,train_Y,test_Y=train_test_split(train_x, train_y,test_size=0.2)


## Step 10a Create linear regression model
regr = linear_model.LinearRegression()
## Step 10b Train the model using the training sets
regr.fit(train_X, train_Y)

## Step 11 Make predictions using the test set
train_Y_pred = regr.predict(test_X)
## Step 12a Print The coefficients
print('Coefficients: \n', regr.coef_)
## Step 12b Print The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(test_Y, train_Y_pred))
## Step 12c Print The root mean squared error
rms = sqrt(mean_squared_error(test_Y, train_Y_pred))
print(f'Root mean squared error: {rms}')
## Step 12d The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(test_Y, train_Y_pred))
