from django.shortcuts import render
from .forms import Imagee
from .models import UserUpload
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model
import base64
from io import BytesIO
import subprocess
import os
import matplotlib.pyplot as plt
import torch    

path_hubconfig = r'E:\company work\New folder (5)\New folder (5)\Deploy\yolov5-master'
path_weightfile = r'E:\company work\New folder (5)\New folder (5)\Deploy\best.pt'

model = torch.hub.load(path_hubconfig, 'custom', path=path_weightfile, force_reload=True, source='local')

# Create your views here.
def to_data_uri(pil_image):
        data = BytesIO()
        pil_image.save(data, "JPEG") # pick your format
        data64 = base64.b64encode(data.getvalue())
        return u'data:img/jpeg;base64,'+data64.decode('utf-8') 

def to_image(numpy_img):
    img = Image.fromarray((numpy_img*255).astype('uint8'), 'L')
    img.save("predicted_image.jpg", format="JPEG")
    return img

def read_image(path):
        x = cv2.imread(path, cv2.IMREAD_COLOR)
        x = cv2.resize(x, (128, 128))
        x = x/255.0
        x = x.astype(np.float32)
        x = np.expand_dims(x, axis=0)
        return x

def index(request):
    if request.method == "POST":
        form = Imagee(files=request.FILES)
        if form.is_valid():
            form.save()
            obj = form.instance
            result = UserUpload.objects.latest('id')
            models = load_model('E:/company work/New folder (5)/New folder (5)/Deploy/seg/model.h5')
            test_image = 'E:/company work/New folder (5)/New folder (5)/Deploy/media/' + str(result)
            y = read_image(test_image)
            y_pred = models.predict(y)[0] > 0.5
            y_pred = np.squeeze(y_pred, axis=-1)
            y_pred = y_pred.astype(np.int32)
            pil_image = to_image(y_pred)
            image_uri = to_data_uri(pil_image)

            inference_img = Image.open("predicted_image.jpg")

            results = model(inference_img, size=416)
            results.render()
            for img in results.ims:
                img_base64 = Image.fromarray(img)
                img_base64.save("segemented_image.jpg", format="JPEG")

            inf_img = Image.open("segemented_image.jpg")
            seg_uri = to_data_uri(inf_img)
            return render(request, 'index.html',{'form':form,'obj':obj,'image_uri': image_uri,'inference_img':seg_uri})
    else:
        form = Imagee()
    return render(request, 'index.html',{'form':form})