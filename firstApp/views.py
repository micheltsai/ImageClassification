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



# model_graph = Graph()
# with model_graph.as_default():
#
#     model=load_model('./models/plant5000.h5')

def index(request):
    context={'a':1}
    return render(request,'index.html',context)


def predictImage(request):
     print(request)
     print(request.POST.dict())
     fileObj = request.FILES['filePath']
     eid1= request.POST.get('eid') #select models
     print(eid1)
     modelPath=''
     if eid1=='Option1':
         modelPath='./models/plant20201026_5000.h5'
         clas_data = './models/classes.json'
     else:
        modelPath='./models/insect5000.h5'
        clas_data = './models/insectclass.json'
     print(modelPath)


     with open(clas_data, 'r', encoding="utf-8") as f:
         labelInfo = f.read()
     labelInfo = json.loads(labelInfo)

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
         print("plant")
         model = load_model(modelPath)
         preds = model.predict_classes(img_arr)
         #print('測試資料的預測類別', preds)
     print(preds[0])
     #predictedLabel = labelInfo[str(np.argmax(preds[0]))]
     predictedLabel = labelInfo[str(preds[0])]
     print(predictedLabel)
     context = {'filePathName': filePathName, 'predictedLabel': predictedLabel[1]}
     return render(request, 'index.html', context)



def viewDataBase(request):
    import os
    listOfImages=os.listdir('./media/')
    listOfImagesPath=['./media/'+i for i in listOfImages]
    context={'listOfImagesPath':listOfImagesPath}
    return render(request,'viewDB.html',context)