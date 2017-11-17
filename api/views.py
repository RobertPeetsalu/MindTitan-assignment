# -*- coding: utf-8 -*-
"""
@author: Robert Peetsalu
"""

from api import app, log
from api.models import predict
from flask import jsonify, request, make_response, abort
import requests

labels = ['price', 'dress_code', 'accessibility', 'parking_lot', 'smoking_area', 'other_services']

@app.errorhandler( 400 )
def bad_request( error ):
    bad_request_hint = 'Correct request format is a JSON list containing following keys: ' + str( labels )
    return make_response( jsonify( {'bad_request_error': bad_request_hint} ), 400 )

@app.route( '/' )
def index():
    log.info( 'request received' )
    if not request.json:
        log.info( 'not JSON' )
        abort( 400 )
    for label in labels: #TODO test with bad requests
        if label not in request.json:
            log.info( 'missing label' )
            abort( 400 )
        if not isinstance( type( request.json[label] ), str ):
            log.info( 'non-text value' )
            abort( 400 )
    rating = predict( request.get_json() )
    return jsonify( { 'rating': rating } ), 201

@app.route( '/test' )
def test():
    test_restaurant = '[{"price":"low", \
                     "dress_code":"formal", \
                     "accessibility":"completely", \
                     "parking_lot":"public", \
                     "smoking_area":"not_permitted", \
                     "other_services":"internet"}]'
    response = requests.post( 'http://127.0.0.1:5000/', data=test_restaurant )
    log.info( response.text )
    return response