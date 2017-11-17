
# Include private.py in this directory for passwords and other secret stuff.
try:
    from . import private
    BasePrivate = private.PrivateConfig
except ImportError as e:
    BasePrivate = object


class BaseConfig(BasePrivate):
    PORT = 5000


class DevelopmentConfig(BaseConfig):
    HOST = '127.0.0.1'
    DEBUG = True
    TESTING = True


class ProductionConfig(BaseConfig):
    HOST = '0.0.0.0'
    DEBUG = False
    TESTING = False

