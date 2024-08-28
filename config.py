# config.py
import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY')
    DEBUG = False

class ProductionConfig(Config):
    DEBUG = False

class StagingConfig(Config):
    DEBUG = True

class DevelopmentConfig(Config):
    DEBUG = True
