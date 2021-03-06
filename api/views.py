# -*- coding: utf-8 -*-
"""
@author: Robert Peetsalu
"""

from api import app, log
from api.models import predict
from flask import jsonify, request, make_response, abort

labels = ['price', 'dress_code', 'accessibility', 'parking_lot', 'smoking_area', 'other_services']
example = ' Here is an example of a correct JSON string with required restaurant features: [{"price":"low", "dress_code":"formal", "accessibility":"completely", "parking_lot":"public", "smoking_area":"not_permitted", "other_services":"internet"}]'

@app.errorhandler( 400 )
def not_json( error ):
    not_json_hint = 'No JSON string was passed.' + example
    return make_response( jsonify( {'bad_request_error': not_json_hint} ), 400 )

@app.errorhandler( 416 )
def nonstandard_category( error ):
    nonstandard_category_hint = 'Some restaurant feature was given a nonstandard value. Please compare with past data.'
    return make_response( jsonify( {'bad_request_error': nonstandard_category_hint} ), 400 )

@app.errorhandler( 417 )
def missing_label( error ):
    missing_label_hint = 'Some required restaurant feature was missing from your JSON string.' + example
    return make_response( jsonify( {'bad_request_error': missing_label_hint} ), 400 )

@app.errorhandler( 418 )
def not_text( error ):
    not_text_hint = 'All restaurant feature values must be strings.' + example
    return make_response( jsonify( {'bad_request_error': not_text_hint} ), 400 )

@app.route( '/', methods=['GET', 'POST'] )
def index():
    log.info( 'request received' )
    if not request.json:
        abort( 400 )
    for restaurant in range( len( request.json ) ):
        for label in labels:
            if label not in request.json[restaurant].keys():
                abort( 417 )
            if not isinstance( request.json[restaurant][label] , str ):
                log.info( type( request.json[restaurant][label] ) )
                abort( 418 )
    log.info( 'nice JSON!')
    ratings = predict( request.get_json() )
    return jsonify( { 'ratings': ratings.tolist() } ), 201