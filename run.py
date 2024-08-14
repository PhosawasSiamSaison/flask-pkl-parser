# run.py
import os
from app import app

if os.environ.get('FLASK_ENV') == 'production':
  app.config.from_object('config.ProductionConfig')
elif os.environ.get('FLASK_ENV') == 'staging':
  app.config.from_object('config.StagingConfig')
else:
  app.config.from_object('config.DevelopmentConfig')
  from dotenv import load_dotenv
  load_dotenv()

if __name__ == '__main__':
  app.run()
