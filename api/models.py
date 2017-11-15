# -*- coding: utf-8 -*-
"""
@author: Robert Peetsalu
"""

from api.utils import encode
import numpy as np
import pandas as pd
import collections
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn.svm import LinearSVC

def train( data ):
    """ Trains a restaurant rating prediction model and stores it in model.sav.
    
    Parameters
    ----------
    data : Named tuple of Pandas DataFrames
        Output of the preprocess() method.
    """
    linear_clf = LinearSVC()
    linear_clf.fit( data.independent, data.dependent ) #TODO
    print( linear_clf.coef_ )
    print( linear_clf.intercept_ )
    joblib.dump( linear_clf, 'model.sav' )
    return True

def predict( restaurant ):
    """ Predicts restaurant rating.
    
    Parameters
    ----------
    restaurant : JSON list
        List of restaurant features.
    
    Returns
    -------
    rating : 0, 1 or 2
        Predicted rating of the restaurant.
    
    Example calls
    -------------
    predict( {'price':'low', 
              'dress_code':'formal', 
              'accessibility':'completely', 
              'parking_lot':'public', 
              'smoking_area':'not_permitted', 
              'other_services':'internet'} )
    """
    model = joblib.load( 'model.sav' )
    df_restaurant = restaurant.read_json() #TODO
    print( df_restaurant.info() )
    
    
    encoded_restaurant = encode( df_restaurant ) #TODO
    rating = model.predict( encoded_restaurant ) #TODO
    return rating

def preprocess( 
        dataframe, 
        test_size = 0.1, 
        random_state = np.random.randint( 2**31 - 1 )
    ):
    """ Splits a pandas DataFrame with no missing values into test and training set.
    
    Parameters
    ----------
    dataframe : Pandas DataFrame
        Dataframe with no missing values.
    test_size : float between 0.0 and 1.0 or positive integer less than the number of input rows. Default is 0.1
        Indicates the proportion or absolute number of input rows to be randomly separated for the test data set.
    random_state : integer like 0, 42 or numpy.random.randint( 2**31 - 1 ) (default)
        Pseudorandom number generator state for random sampling.
    
    Returns
    -------
    Named tuple 'data' consisting of Pandas DataFrames containing preprocessed data split in different ways
    
    x_train : Pandas DataFrame
        training set independent variable columns. 
    y_train : Pandas DataFrame
        training set dependent variable columns.
    x_test : Pandas DataFrame 
        testing set independent variable columns.
    y_test : Pandas DataFrame 
        testing set dependent variable columns.
    independent : Pandas DataFrame 
        independent variable columns before splitting the test set.
    dependent : Pandas DataFrame 
        dependent variable columns before splitting the test set.
    joined : Pandas DataFrame 
        independent and dependent variables in a single table.
    
    Example calls
    -------------
    minimal : preprocess( dataframe )
    
    detailed : preprocess( dataframe, 0.2, 0 )
    """
    dependent = encode( dataframe.loc[:, ['rating']] )
    independent = encode( dataframe.drop( 'rating', axis=1 ) )
    
    # Split the dataset into training and test sets
    x_train, x_test, y_train, y_test = train_test_split( 
            independent, 
            dependent, 
            test_size = test_size, 
            random_state = random_state 
        )
    
    # Prepare a named tuple for output
    data = collections.namedtuple( 
            'data', 
            ['x_train', 'y_train', 'x_test', 'y_test', 'independent', 'dependent', 'joined'] 
        )
    data.x_train, data.x_test, data.y_train, data.y_test = x_train, x_test, y_train, y_test
    data.independent, data.dependent = independent, dependent
    data.joined = pd.concat( ( data.independent, data.dependent ), axis=1 )
    
    return data