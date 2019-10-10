# Start Python Imports
import math, time, random, datetime

# Data Manipulation
import numpy as np
import pandas as pd
import re

# Visualization
import matplotlib.pyplot as plt
import missingno
import seaborn as sns
plt.style.use('seaborn-whitegrid')

# Preprocessing
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, label_binarize, StandardScaler

# Machine learning
import catboost
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import model_selection, tree, preprocessing, metrics, linear_model
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc, r2_score
from sklearn.metrics import accuracy_score, precision_score, f1_score, log_loss
from sklearn.model_selection import GridSearchCV



from catboost import CatBoostClassifier, Pool, cv

# ignoring warnings for the time being
import warnings
warnings.filterwarnings('ignore')
from Functions import *

from statistics import mode


########################################################################################################################
# EDA

df_train = pd.read_csv("titanic\\train.csv") # in current working dir
df_test = pd.read_csv("titanic\\test.csv")

# setting display options for functions like head
pd.set_option('display.width', 400)
pd.set_option('display.max_columns', len(df_train.columns))

# initial investigation
df_train.head()
df_train.describe(include='all')  # 1) missing ages need to be dealt with, 2) 75% under 38 years of age, 3) 75% paid under 31 pounds, whereas the max is 512 pounds

# Plot graphic of missing values
missingno.matrix(df_train, figsize=(10, 6))  # numbers on RHS indicate the lowest and highest amounts of data columns filled. cabin is sparcely populated so this may have to be removed. age is porbably an important factor, values may have to be imputed here
df_train.isnull().sum()

# heat map to see if any of the features are correlated
df_train_floats = df_train.loc[:, df_train.dtypes == 'float64']
df_train_ints = df_train.loc[:, df_train.dtypes == 'int64']
df_train_flints = pd.concat([df_train_floats, df_train_ints], axis=1)
label_encoder = LabelEncoder()
df_train_flints['Sex'] = label_encoder.fit_transform(df_train['Sex'])
sns.heatmap(df_train_flints.corr(), square=True, annot=True, linewidths=0.05, cmap='YlGnBu', vmin=-1)  # moderate correlation between survived and sex, small correlation between survived and pclass

# create two data frames: categorical,  continuous and/or binary
df_train_encode = pd.DataFrame()  # this data will be one hot encoded
df_train_keep = pd.DataFrame()  # this data frame will contain continuous data that may need to be standardised

# checking what data types we have
df_train.dtypes

########################################################
# exploring different features
df_train.head()  # to see features again

# Feature: Survived (1 = survived, 0 = died)
fig = plt.figure(figsize=(15, 2))
sns.countplot(y='Survived', data=df_train)
df_train.Survived.value_counts()
df_train_keep['Survived'] = df_train['Survived']  # adding Survived to the keep data frame

# Feature: Pclass
sns.countplot(df_train.Pclass)  # the ratio of classes is roughly 1:1:2 (first,second,third)
df_train_encode['Pclass'] = df_train['Pclass']  # adding Pclass to the encode data frame

# Feature: Name
len(df_train.Name.value_counts())== np.unique(df_train.Name).size  # check to see if there are any duplicates
titles_train = get_titles(df_train)
pd.value_counts(titles_train)  # frequencies of all the titles
df_train.iloc[titles_train.index('Don')]  # looking into weird looking titles
df_train_encode['Title'] = titles_train  # Will use this later to 1) impute ages, 2) possible predictor of survival by title

# Feature: Sex
plt.figure(figsize=(10, 3))
sns.countplot(y='Sex', data=df_train)
plt.figure(figsize=(10, 3))
g = sns.barplot(x="Sex", y="Survived", data=df_train, errwidth=0)
g = g.set_ylabel("Survival Probability")  # just under 20% of all males on board survived, while just under 75% of all females on-board survived.
df_train[["Sex", "Survived"]].groupby('Sex').mean()  # to get percentages that survived relative to their sex
df_train_encode['Sex'] = df_train['Sex']  # adding sex to be one hot encoded later.

# Feature: Age
df_train.Age.plot.hist()
df_train.Age.isnull().sum()
plt.figure(figsize=(8, 4))
g = sns.FacetGrid(df_train, col='Survived')
g = g.map(sns.distplot, "Age")  # notice how not a lot of people above ~60 didn't survive
# notice how a lot of babies survived, these were probably passes into lifeboats even if there wasn't space for the parent
# Dead and survived superimposed over each other. big difference noticed in the age group of baby and young children
df_given_ages = df_train[~np.isnan(df_train.Age)]
df_given_ages_survived = df_given_ages[df_given_ages.Survived == 1]
plt.figure(figsize=(8, 4))
g = sns.distplot(df_given_ages.Age, color="Red", kde=False)
g = sns.distplot(df_given_ages_survived.Age, color="Blue", kde=False)
g = g.legend(["All Passengers", "Not Survived"])
df_train_keep['Age'] = df_train['Age']  # copying our age data to our new data frame
mean_imputing_ages(df_train, df_train_encode, df_train_keep, titles_train)
df_train_encode = df_train_encode.drop(['Title'], axis=1)  # dropping title from encode

# Feature: SibSp
df_train_keep['SibSp'] = df_train['SibSp']  # adding the column SibSp to our subset data frame

# Feature: ParCh
df_train_keep['Parch'] = df_train['Parch']  # adding the column ParCh to our subset data frame

# Feature: Ticket
# leaving this feature out for the time being. combination of letters and number.

# Feature: Fare
df_train_keep['Fare'] = df_train['Fare']  # we will keep this column and add it to our subset data frame

# Feature: Cabin
# ignoring feature due to the amount of missing values

# Feature: Embarked
df_train_encode['Embarked'] = df_train['Embarked']  # adding this column to our subset data frame as it needs to be encoded
df_train.Embarked.isnull().sum()  # we have 2 null values in Embarked. From googling we found that there two passengers embarked in S
missing_embarked_index = df_train[df_train.Embarked.isnull()].index  # these are their two indices
df_train_encode.Embarked.iloc[missing_embarked_index[0]] = 'S'  # https://www.encyclopedia-titanica.org/titanic-survivor/amelia-icard.html
df_train_encode.Embarked.iloc[missing_embarked_index[1]] = 'S'  # https://www.encyclopedia-titanica.org/titanic-survivor/martha-evelyn-stone.html
df_train_encode.Embarked.isnull().sum()  # we have no more null values in our embarked column

# scaling our data in the keep data frame, except for survived
df_train_keep_stand = df_train_keep.drop(['Survived'], axis=1)
df_train_keep.drop(['Age', 'Fare', 'SibSp', 'Parch'], axis=1, inplace=True)
scalar = StandardScaler()  # standardisation
df_train_keep_stand[['Age', 'Fare', 'SibSp', 'Parch']] = scalar.fit_transform(df_train_keep_stand[['Age', 'Fare', 'SibSp', 'Parch']])

# one hot encoding
one_hot_cols = df_train_encode.columns.tolist()
df_train_encoded = pd.get_dummies(df_train_encode, columns=one_hot_cols, drop_first=True)  # one hot encodeing three features, dropping the first column of each of these features after this.
df_train_encoded.head()

# combining our 3 data frames
df_train_combined = pd.concat([df_train_keep, df_train_keep_stand, df_train_encoded], axis=1)

# creating a pseudo test set from our training set to do additional tests. (length 100)
pseudo_length = 100
df_train_combined_cropped = df_train_combined.iloc[0:df_train_combined.shape[0]-pseudo_length, :]  # X_train = df_train_combined_cropped + (X_pseudo_test + y_pseudo_test)

# the following two data frames are used in the grid search to update our algorithm
X_train_cropped = df_train_combined.drop('Survived', axis=1).iloc[0:df_train_combined.shape[0]-pseudo_length, :]  # X_train = X_train_cropped + X_pseudo_test
y_train_cropped = df_train_combined.Survived[0:df_train_combined.shape[0]-pseudo_length]  # y_train = y_train_cropped + y_pseudo_test

X_pseudo_test = df_train_combined.drop('Survived', axis=1).iloc[df_train_combined.shape[0]-pseudo_length::, :]
y_pseudo_test = df_train_combined.Survived[df_train_combined.shape[0]-pseudo_length::]
X_pseudo_test.reset_index(drop=True, inplace=True)
y_pseudo_test.reset_index(drop=True, inplace=True)

X_train = df_train_combined.drop('Survived', axis=1)
y_train = df_train_combined.Survived

#######################################################
# setting up the test data frame
df_test_encode = pd.DataFrame()  # creating two data frames
df_test_keep = pd.DataFrame()

# Feature: Pclass
df_test_encode['Pclass'] = df_test['Pclass']  # adding Pclass to encode
mean_first = df_train.Fare[df_train.Pclass == 1].mean()  # these will be used later to impute missing values in Pclass and in Fare
mean_second = df_train.Fare[df_train.Pclass == 2].mean()
mean_third = df_train.Fare[df_train.Pclass == 3].mean()
if df_test_encode.Pclass.isnull().sum() > 0:
    impute_Pclass(df_train, df_test, df_test_encode, mean_first, mean_second, mean_third)  # check to see if there is any missing values

# Feature: Name
titles_test = get_titles(df_test)  # need to get the titles of each passenger in order to impute the missing ages
df_test_encode['Title'] = titles_test  # adding to encode for now, may be removed later.

# Feature: Sex
# accounting for missing values
df_test_encode['Sex'] = df_test['Sex']  # adding Sex to encode
if df_test.Sex.isnull().sum() > 0:  # checking for null values
    impute_Sex(df_test, df_test_encode, df_train, titles_train, titles_test)

# Feature: Age
df_test_keep['Age'] = df_test['Age']  # adding Age to keep
if df_test_keep.Age.isnull().sum() > 0:  # check to see if we have null values (imputed from mean of the titles)
    mean_imputing_ages(df_test, df_test_encode, df_test_keep, titles_test)
df_test_encode = df_test_encode.drop(['Title'], axis=1)  # removing titles from encode as we do not need it any more to impute ages.

# Feature: SibSp
df_test_keep['SibSp'] = df_test['SibSp']  # adding SibSp to keep, this feature will be standardized later
if df_test_keep.SibSp.isnull().sum() > 0:  # check to see if there are any missing values, (missing values are imputed from the most common value of Sibsp)
    impute_SibSp(df_test, df_test_keep, df_train)

# Feature: Parch
df_test_keep['Parch'] = df_test['Parch']  # adding SibSp to keep, this feature will be standardized later
if df_test_keep.Parch.isnull().sum() > 0:  # check to see if there is any missing values, (missing values are imputed from the most common value of Parch)
    impute_Parch(df_test, df_test_keep, df_train)

# Feature: Fare
df_test_keep['Fare'] = df_test['Fare']  # adding Fare to keep, this feature will be standardized later
if df_test_keep.Fare.isnull().sum() > 0:  # check to see if there are any values missing from Fare (missing values will be imputed from the mean value of the class that the passenger is in)
    impute_Fare(df_test_keep, df_test_encode, mean_first, mean_second, mean_third)

# Feature: Embarked
df_test_encode['Embarked'] = df_test['Embarked']  # adding Embarked to encode, this will be one hot encoded later
if df_test_encode.Embarked.isnull().sum() > 0:  # check to see if there are any missing values from Embarked ( missing values will be imputed from the most common port of embarkation
    for row in df_test_encode.Embarked[df_test_encode.Embarked.isnull()].index:
        df_test_encode.Embarked[row] = mode(df_train.Embarked)

# standardizing the data that is in the keep data frame
df_test_keep_stand = df_test_keep
df_test_keep_stand[['Age', 'SibSp', 'Parch', 'Fare']] = scalar.fit_transform(df_test_keep[['Age', 'SibSp', 'Parch', 'Fare']])

# one hot encoding the data that is in encode
one_hot_cols = df_test_encode.columns.tolist()
df_test_encoded = pd.get_dummies(df_test_encode, columns=one_hot_cols, drop_first=True)  # also dropping the first column from each feature

# concatenating the two data frames together
df_test_combined = pd.concat([df_test_keep_stand, df_test_encoded], axis=1)

X_test = df_test_combined

nans_in_data_frame(X_test)  # check to see if there are any null values in the data frame
