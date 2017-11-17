# -*- coding: utf-8 -*-
"""
@author: Robert Peetsalu
"""
from api import log
from api.utils import import_data, one_hot_encode
from sys import exc_info
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from flask import abort
#from sklearn.model_selection import cross_val_score

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
#    print( '\nCross-validation scores' )
#    print( scores )
    linear_clf.fit( independent, dependent )
    joblib.dump( linear_clf, 'model.sav' )
    
    #Pickle one restaurant record with correct labels to avoid pickling LabelEncoder and OneHotEncoder
    record = independent.loc[0:0]
    joblib.dump( record, 'record.sav' )
    return linear_clf

def predict( restaurants ):
    """ Predicts ratings of a list of restaurants.
    
    Parameters
    ----------
    restaurants : JSON array
        List of restaurant features.
    
    Returns
    -------
    rating : list of integers 0, 1 or 2
        Predicted rating of the restaurant.
    
    Example calls
    -------------
    predict( '[{"price":"medium", 
                "dress_code":"formal", 
                "accessibility":"completely", 
                "parking_lot":"public", 
                "smoking_area":"not_permitted", 
                "other_services":"internet"},
               {"price":"low", 
                "dress_code":"formal", 
                "accessibility":"completely", 
                "parking_lot":"public", 
                "smoking_area":"not_permitted", 
                "other_services":"internet"}]' )
    """
    df_restaurants = pd.DataFrame.from_dict( restaurants )
    enc_df_restaurants = one_hot_encode( df_restaurants )
    
    #Expand dataframe with full list of acceptable labels by concatenating and then dropping a sample record
    sample_record = joblib.load( 'record.sav' )
    encoded_restaurants = pd.concat( [sample_record, enc_df_restaurants], ignore_index=True)
    encoded_restaurants = encoded_restaurants.fillna( False ).drop(0)
    log.info( '\nEncoded restaurant features:' )
    log.info( encoded_restaurants )
    model = joblib.load( 'model.sav' )
    try: 
        ratings = model.predict( encoded_restaurants )
        log.info( '\nPredicted restaurant ratings:' )
        log.info( ratings )
        return ratings
    except ValueError:
        log.info( "ValueError:", exc_info()[0] )
        abort( 416 )