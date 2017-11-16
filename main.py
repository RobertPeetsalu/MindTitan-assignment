#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Robert Peetsalu
"""

from api import app
from api.models import train

train()

app.debug = app.config['DEBUG']
app.run( host=app.config['HOST'], port=app.config['PORT'] )
