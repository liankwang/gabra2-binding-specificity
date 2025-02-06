'''
Utility functions for model evaluation and data processing
'''

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut, cross_val_score

def perform_loocv(X, y, model):
    loo = LeaveOneOut()
    scores = cross_val_score(model, X, y, cv=loo, scoring='neg_mean_squared_error')
    
    return -scores.mean()

def custom_loocv(X, y, model):
    y_true = []
    y_pred = []

    for i in range(len(X)):
        # Split the data into training and testing sets
        X_train = X.drop(index=i)
        y_train = y.drop(index=i)
        X_test = X.iloc[[i]]
        y_test = y.iloc[[i]]

        # Train the model
        model.fit(X_train, y_train)

        # Make predictions
        y_pred_i = model.predict(X_test)
        y_true.append(y_test.values[0])
        y_pred.append(y_pred_i[0])

    # Calculate performance metrics
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return [mse, r2]
