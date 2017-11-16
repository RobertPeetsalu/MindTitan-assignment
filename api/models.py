# -*- coding: utf-8 -*-
"""
@author: Robert Peetsalu
"""

from api.utils import import_data, one_hot_encode
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib

test_restaurant = '[{"price":"low", \
                     "dress_code":"formal", \
                     "accessibility":"completely", \
                     "parking_lot":"public", \
                     "smoking_area":"not_permitted", \
                     "other_services":"internet"}]'

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
    linear_clf = LinearSVC( C=1.0, dual=False, fit_intercept=False )
#    scores = cross_val_score( linear_clf, independent, dependent, cv=5 )
#    print( 'Cross-validation scores' )
#    print( scores )
    linear_clf.fit( independent, dependent )
#    print( linear_clf.predict( independent.loc[20:20] ) )
    joblib.dump( linear_clf, 'model.sav' )
    
#    print( data.loc[20:20].to_json( orient='records') )
    predict( test_restaurant ) #TODO remove
    
    return linear_clf

def predict( restaurant ):
    """ Predicts restaurant rating.
    
    Parameters
    ----------
    restaurant : JSON array
        List of restaurant features in square brackets.
    
    Returns
    -------
    rating : 0, 1 or 2
        Predicted rating of the restaurant.
    
    Example calls
    -------------
    predict( '[{"price":"low", 
                "dress_code":"formal", 
                "accessibility":"completely", 
                "parking_lot":"public", 
                "smoking_area":"not_permitted", 
                "other_services":"internet"}]' )
    """
    model = joblib.load( 'model.sav' )
    df_restaurant = pd.read_json( restaurant, 'records' ) #TODO test
    
    print( df_restaurant.info() )
    encoded_restaurant = one_hot_encode( df_restaurant ) #TODO test
    print( 'encoded' )
    print( encoded_restaurant.info() )
    rating = model.predict( encoded_restaurant ) #TODO test
    print( 'rating' )
    print( rating )
    return rating