

from __future__ import division, print_function
# coding=utf-8

import os

import numpy as np
from keras.preprocessing import image 



from keras.models import load_model



import tensorflow as tf

global graph
graph=tf.get_default_graph()

#global graph
#graph = tf.get_default_graph()



# Flask utils
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()

# Load your trained model
model = load_model('animal.h5')
       # Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#model.save('')
print('Model loaded. Check http://127.0.0.1:5000/')




@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('base.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['image']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        img = image.load_img(file_path, target_size=(64, 64))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        
        with graph.as_default():
            preds = model.predict_classes(x)
        index = ['Bears','Crows','Elephants','Racoons','Rats']
        text = "prediction : "+index[preds[0]]
        
               # ImageNet Decode
        

        
        return text
    


if __name__ == '__main__':
    app.run(debug=True,threaded = False)


