from flask import Flask, render_template, request
from werkzeug import secure_filename
import os
import numpy as np
from PIL import Image
import keras
from keras.models import load_model
import tensorflow as tf
import re
import base64

app = Flask(__name__)
UPLOAD_FOLDER = os.path.basename('test')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

global graph
graph = tf.get_default_graph()
model = load_model("model/seed33230.h5")

@app.route('/')
def welcome():
	return render_template('index.html')

@app.route('/upload')
def upload_file():
	return render_template('upload.html')
@app.route('/pic')
def take_photo():
	return render_template('tkpic.html')
	
@app.route('/uploader', methods = ['GET', 'POST'])
def upload():
   if request.method == 'POST':
      params = request.json
      print(params)
      f = request.files['file']
      f1 = np.array(f)
      print(f1.shape)
      f.save('test/'+secure_filename(f.filename))
      img =  Image.open('test/'+secure_filename(f.filename))
      img1 = img.resize((64, 64), Image.BICUBIC)
      img2 = np.array(img1)
      img2 = img2[:,:,:3]
      print(img2.shape)
      img2 = np.reshape(img2,(1,64,64,3))
      print(img2.shape)
      #img1.save('test/final.jpg')
      with graph.as_default():
           print(model.predict(img2)[0][0])
           if(round(model.predict(img2)[0][0]) == 1):
           	return render_template('result1.html')
           else:
           	return render_template('result0.html')
           	
def convertImage(imgData1):
	#print(type(imgData1))
	imgstr = re.search(b'base64,(.*)',imgData1).group(1)
	#print(imgstr)
	with open('test/out.png', 'wb') as output:
		output.write(base64.b64decode(imgstr))
	
	'''imgstr = re.search(b'base64,(.*)',imgData1).group(1) #print(imgstr)
	with open('output.png','wb') as output:
	   output.write(base64.b64decode(imgstr))'''


@app.route('/predict', methods =['GET', 'POST'])
def predict():
   imgData = request.get_data()
   #print(type(imgData))
   convertImage(imgData)
   img =  Image.open('test/out.png')
   img1 = img.resize((64, 64), Image.BICUBIC)
   img2 = np.array(img1)
   img2 = img2[:,:,:3]
   #print(img2.shape)
   img2 = np.reshape(img2,(1,64,64,3))
   #print(img2.shape)
   #img1.save('test/final.jpg')
   with graph.as_default():
        out = model.predict(img2)[0][0]
        print(out)
        #convert the response to a string
        response = str(out)
        return response	   	
           	

if __name__ == '__main__':
	app.run(host='0.0.0.0',port='5000',debug = True)
