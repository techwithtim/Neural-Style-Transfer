from flask import session
import time
from application import create_app
from application.database import DataBase
import config

# SETUP
app = create_app()

# MAINLINE
if __name__ == "__main__":  # start the web server 
    app.run(debug=True, threaded=True, host=str(config.Config.SERVER))