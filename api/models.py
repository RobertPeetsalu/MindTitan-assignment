# -*- coding: utf-8 -*-
"""
@author: Robert Peetsalu
"""

from api.utils import import_data, one_hot_encode
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib

def train():
    """ Trains a restaurant rating prediction model and stores it in model.sav.
    
    Returns
    -------
    linear_clf : LinearSVC object
        Trained restaurant rating prediction model.
    """
    data = import_data()
    dependent = data['rating']
    independent = one_hot_encode( data.drop( 'rating', axis=1 ) )
    print( independent.info() )
    linear_clf = LinearSVC( C=1.0, dual=False, fit_intercept=False )
    scores = cross_val_score( linear_clf, independent, dependent, cv=5, n_jobs=-1 )
    print( 'scores: ' + scores )
    linear_clf.fit( independent, dependent ) #TODO
    print( linear_clf.intercept_ )
    joblib.dump( linear_clf, 'model.sav' )
    return linear_clf

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
    
    
    encoded_restaurant = one_hot_encode( df_restaurant ) #TODO
    rating = model.predict( encoded_restaurant ) #TODO
    return rating