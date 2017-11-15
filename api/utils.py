# -*- coding: utf-8 -*-
"""
@author: Robert Peetsalu
"""

import os.path as path
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def import_data():
    """
    Imports and INNER JOINs restaurant rating tables, skips rows with missing values and duplicate client ratings.
    
    Returns
    -------
    data : Pandas DataFrame
        Dataframe only containing features required for restaurant rating classification.
    """
    PROJECT_DIR = path.dirname( path.dirname( path.dirname( __file__ ) ) )
    
    geoplaces = pd.read_csv( 
            filepath_or_buffer = path.join( PROJECT_DIR, 'data', 'geoplaces2.csv' ), 
            usecols = ['placeID','smoking_area', 'dress_code', 'accessibility', 'price', 'other_services'], 
            error_bad_lines = False 
        ).dropna()
    
    parking = pd.read_csv( 
            filepath_or_buffer = path.join( PROJECT_DIR, 'data', 'chefmozparking.csv' ), 
            usecols = ['placeID','parking_lot'], 
            error_bad_lines = False 
        ).dropna()
    
    rating = pd.read_csv( 
            filepath_or_buffer = path.join( PROJECT_DIR, 'data', 'rating_final.csv' ), 
            usecols = ['placeID', 'userID', 'rating'], 
            error_bad_lines = False, 
            dtype = {'rating': object}
        ).dropna()
    
    # Remove duplicate ratings from the same user about the same restaurant if any and drop userID
    rating = rating.drop_duplicates( ['placeID', 'userID'] ).drop( 'userID', axis=1 )
    
    # INNER JOIN tables on placeID to make a duplicate row for each client rating and parking type
    data = pd.merge( pd.merge( geoplaces, parking, on = 'placeID' ), rating, on = 'placeID' )
    
    return data.drop( 'placeID', axis=1 )

def encode( df ):
    """ One-hot-encode categorical features.
    
    Parameters
    ----------
    df : Pandas DataFrame 
        Dataframe with no missing values and containing only categorical features.
    
    Returns
    -------
    df : Pandas DataFrame
        Dataframe with encoded categorical features next to unencoded original features.
    """
    categories = df.apply( pd.Series.nunique )
    categorical = df.select_dtypes( include=['object'] ).axes[1]
    for column in categorical:
        encoder = LabelEncoder()
        encoded = encoder.fit_transform( df[column] )
        hot = OneHotEncoder( sparse = False, dtype = bool ).fit_transform( encoded.reshape( -1, 1 ) )
        for category in range( categories[column] ):
            category_name = encoder.classes_[ category ].replace( ' ', '_' )
            df[ column + '_' + category_name ] = hot[:,category]
    return df