#!/usr/bin/env python

import json
import pickle
import logging
import numpy as np
from flask import Flask, request

# # Flask
# 
# 1. Setting up API for model using flask will trigger a warning - "This is a development server. Do not use it in a production deployment.". It is due to the setting of Flask supporting only a single thread which is not able to support high volume of requests. For more production deployment, gunicorn (in linux) or waitress module to support multiple threads.
#  
# 2. Deploy Your Machine Learning Model on Docker — Part 1, 2
#     * https://medium.com/analytics-vidhya/deploy-your-machine-learning-model-on-docker-ee2b931e133c
#     * https://medium.com/analytics-vidhya/deploy-your-machine-learning-model-on-docker-part-2-d9795fca6795
# 
# 3. Machine Learning Model Deployment using Flask - Flask, Gunicorn
#     * https://www.youtube.com/watch?v=hIq4bVT2ghk
#     
# 4. Decorator - wrapper to modify the function: https://realpython.com/primer-on-python-decorators/
# 5. Create log file for debugging.

# Load the trained model from the directory
with open('flask_pipeline.pkl', 'rb') as file:
    pipeline = pickle.load(file)
    
# Initialize a flask app
app = Flask(__name__)

# Create logging for debugging
logging.basicConfig(filename='flask.log', level=logging.DEBUG,
                   format='%(asctime)s %(name)s %(threadName)s: %(message)s')

logging.info('Pipeline loaded and flask app started.')

# Create an API GET endpoint: can be access from internet surfer
@app.route('/predict', methods=['GET']) # Name the api and declare the method
def predict():
    
    # Read the request data
    x_test = request.args.get('x_test')
    
    # Convert from json to original format (list) to pipeline required format (numpy array)
    x_test = np.array(json.loads(x_test))
    
    # Prediction
    y_pred = pipeline.predict(x_test)
    y_pred = 'No' if y_pred < 0.5 else 'Yes'
    
    logging.info('Prediction made.')
    
    return 'Breast Cancer Prediction: {}'.format(y_pred)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5000')

