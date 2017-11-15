#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Robert Peetsalu
"""

from api import app
from api.utils import import_data
from api.models import preprocess, train
import seaborn as sns

raw_data = import_data()
data = preprocess( raw_data, 0.2, 42 )
#print( data.independent.info() )
#print( data.dependent.info() )
#sns.heatmap( data.joined.corr() )
#train( data )

app.debug = app.config['DEBUG']
app.run( host=app.config['HOST'], port=app.config['PORT'] )
