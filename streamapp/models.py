from django.db import models

import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.python.keras.models import load_model, model_from_json
from tensorflow.python.keras.initializers import glorot_uniform
from tensorflow.keras.utils import CustomObjectScope
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras import backend as K

import streamapp.preprocessing

letter_map = {1:"A",2:"B",3:"C",4:"D",5:"del",6:"E",7:"F",8:"G",9:"H",10:"I",11:"J",12:"K",13:"L",14:"M",15:"N",16:"nothing",17:"O",18:"P",
              19:"Q",20:"R",21:"S",22:"space",23:"T",24:"U",25:"V",26:"W",27:"X",28:"Y",29:"Z"}

# Create your models here.
class Classifier(models.Model):
    input = models.ImageField()
    output = models.CharField(max_length = 10) # should be at most 7 for "nothing"

    model_path = "models/v3-ft10-model.json"
    weights_path = "models/InceptionV3_weights_safe.01-0.17.h5"

    with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
        with open(model_path, 'r') as file:
            model = model_from_json(file.read())
            model.load_weights(weights_path)    

    def predict(self):
        x = image.img_to_array(self.input)
        x = preprocessing.edge_filter(x)
        y = self.model.predict([x/255.0])
        y = np.argmax(y[0])
        return letter_map[y+1]

    #def save(self, *args, **kwargs):
        #self.prediction = self.predict()
        #super().save(*args, **kwargs)