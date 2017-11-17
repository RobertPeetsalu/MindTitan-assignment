# -*- coding: utf-8 -*-
"""
@author: Robert Peetsalu
"""
from api import log
from api.utils import import_data, one_hot_encode
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
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
    
    #Pickle an empty restaurant record with correct labels to avoid pickling LabelEncoder and OneHotEncoder
    empty_record = independent.loc[0:0].applymap(lambda x: False)
    joblib.dump( empty_record, 'empty_record.sav' )
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
    df_restaurant = pd.DataFrame.from_dict( restaurant )
    record = joblib.load( 'empty_record.sav' )
    record.update( one_hot_encode( df_restaurant ) )
#    log.info( '\nEncoded restaurant features:' )
#    log.info( record )
    model = joblib.load( 'model.sav' )
    rating = model.predict( record )
    log.info( '\nPredicted restaurant rating:' )
    log.info( rating )
    return rating