from flask import Flask, render_template, request, flash, url_for, jsonify
import pandas as pd
import numpy as np
from flask import json
from sklearn.model_selection import cross_validate
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
import joblib
from sklearn.ensemble import RandomForestRegressor
from plotly.offline import init_notebook_mode, iplot
year = "2017"  # Year fetching From UI.
C_type ="RAPE"  # Crime type fetching from UI
state = "Andhra Pradesh" # State name fetching from UI
df = pd.read_csv("static/StateWiseCAWPred1990-2016.csv", header=None)
# Selecting State and its attributes.
data1 = df.loc[df[0] == state].values
for x in data1:
    if x[1] == C_type:
        test = x
        break

l = len(df.columns)
trendChangingYear = 2
xTrain = np.array([1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001,
                       2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016])
yTrain = test[2:29]

X = df.iloc[0, 2:l].values
y = test[2:]
regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)  # regression algorithm cealled.
    # Data set is fitted in regression and Reshaped it.
regressor.fit(X.reshape(-1, 1), y)
    # Finding Accuracy of Predictions.
accuracy = regressor.score(X.reshape(-1, 1), y)
print(accuracy)

