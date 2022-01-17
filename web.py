from flask import Flask, render_template, request, send_from_directory
import cv2
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, BatchNormalization, Flatten
import numpy as np
from PIL import Image
import numpy as np
from matplotlib.pyplot import imshow

lo_model = load_model('static/model.h5')
lo_model.load_weights('static/model.h5')

COUNT = 0
app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 1

@app.route('/')
def man():
    return render_template('index.html')

@app.route('/home', methods=['POST'])
def home():
    # global COUNT
    img = request.files['image']

    img.save('static/0.jpg') 
    #def names(number):
     #   if number == 0:
      #     render_template('pred.html')  
       #    #return "Patient has tumor"
        #else:
         #   render_template('prediction.html')  
        # #return "COngratulations !!! Patient do not have tumor"

    img = Image.open(r"static/0.jpg")
    x = np.array(img.resize((128,128)))
    x = x.reshape((1,128,128,3))
    res = lo_model.predict_on_batch(x)
    classification = np.where(res == np.amax(res))[1][0]
    imshow(img)
    if classification == 0:
        return render_template('pred.html') 

    else:
        return render_template('prediction.html')    
    #('Prediction value =  ' + str(res[0][classification]*100))

   # img.save('static/{}.jpg'.format(COUNT))    
  #  img_arr = cv2.imread('static/{}.jpg'.format(COUNT))



   # img_arr = cv2.resize(img_arr, (128,128))
    # img_arr = img_arr / 255.0
   # img_arr = img_arr.reshape(1, 128,128,3)
   # prediction = lo_model.predict(img_arr)

   # x = round(prediction[0,0], 2)
   # y = round(prediction[0,1], 2)
   # preds = np.array([x,y])
  #  COUNT += 1
   
# return render_template('prediction.html')    

@app.route('/load_img')
def load_img():
    global COUNT
    return send_from_directory('static', "0.jpg")


if __name__ == '__main__':
          app.run(debug=True)      




