import os

from django.shortcuts import render
# Create your views here.

from django.core.files.storage import FileSystemStorage

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import json
from tensorflow import Graph
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.keras.applications.densenet import decode_predictions

img_height, img_width=32,32
os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

with open('./models/classes.json','r',encoding="utf-8") as f:
    labelInfo=f.read()

labelInfo=json.loads(labelInfo)


# model_graph = Graph()
# with model_graph.as_default():
#
#     model=load_model('./models/plant5000.h5')

def index(request):
    context={'a':1}
    return render(request,'index.html',context)

"""
def predictImage(request):
    print (request)
    print (request.POST.dict())
    fileObj = request.FILES['filePath']
    fs=FileSystemStorage()
    filePathName = fs.save(fileObj.name,fileObj)
    filePathName = fs.url(filePathName)
    testimage = '.'+filePathName
    img = image.load_img(testimage, targe_size=(img_height, img_width))
    img_arr = []
    x = image.img_to_array(img)
    x=cv2.resize(x, (32, 32))
    img_arr.append(x)
    img_arr = numpy.array(img_arr, dtype="float") / 255.0
    img_arr = img_arr.astype('float32') / 255

    with model_graph.as_default():
        predi = model.predict_classes(img_arr)
        print('測試資料的預測類別', predi)

    import numpy as np
    predictedLabel=labelInfo[str(np.argmax(predi[0]))]

    context={'filePathName':filePathName,'predictedLabel':predictedLabel[1]}
    return render(request,'index.html',context)

"""


def predictImage(request):
     print(request)
     print(request.POST.dict())
     fileObj = request.FILES['filePath']
     fs = FileSystemStorage()
     filePathName = fs.save(fileObj.name, fileObj)
     filePathName = fs.url(filePathName)
     testimage = '.' + filePathName
     img_or = cv2.imread(testimage)
     print(img_or.shape)
     img_test = cv2.resize(img_or, (img_height, img_width))
     print(img_test.shape)
     img_arr=[]
     img_arr.append(img_test)
     img_arr = np.array(img_arr, dtype="float") #/ 255.0
     img_arr = img_arr.astype('float32') / 255
     model_graph = Graph()
     with model_graph.as_default():
         model = load_model('./models/plant20201026_5000.h5')
         preds = model.predict_classes(img_arr)

         #print('測試資料的預測類別', preds)
     print(preds[0])
     #predictedLabel = labelInfo[str(np.argmax(preds[0]))]
     predictedLabel = labelInfo[str(preds[0])]
     print(predictedLabel)
     context = {'filePathName': filePathName, 'predictedLabel': predictedLabel[1]}
     return render(request, 'index.html', context)


"""
def predictImage(request):
    print (request)
    print (request.POST.dict())
    fileObj=request.FILES['filePath']
    fs=FileSystemStorage()
    filePathName=fs.save(fileObj.name,fileObj)
    filePathName=fs.url(filePathName)
    testimage='.'+filePathName
    img = image.load_img(testimage, target_size=(img_height, img_width))
    x = image.img_to_array(img)
    x=x/255
    print(x.shape)
    x=x.reshape(1,img_height, img_width,3)

    # with model_graph.as_default():
    model_graph = Graph()
    with model_graph.as_default():
        model = load_model('./models/plant5000.h5')
        predi = model.predict(x)
        #print(decode_predictions(predi, top=3)[0])
    predictedLabel=labelInfo[str(np.argmax(predi[0]))]

    context={'filePathName':filePathName,'predictedLabel':predictedLabel[1]}
    return render(request,'index.html',context)
"""

def viewDataBase(request):
    import os
    listOfImages=os.listdir('./media/')
    listOfImagesPath=['./media/'+i for i in listOfImages]
    context={'listOfImagesPath':listOfImagesPath}
    return render(request,'viewDB.html',context)