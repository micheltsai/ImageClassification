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

def insect(request2):
    context2={'a':1}
    print("insect")
    return render(request2,'insect.html',context2)


def predictImage2(request2):
     print('2')
     print(request2)
     print(request2.POST.dict())
     fileObj = request2.FILES['filePath']
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
         print("insect")
         model = load_model('./models/test20200517_128_32x32.h5')
         preds = model.predict_classes(img_arr)

         #print('測試資料的預測類別', preds)
     print(preds[0])
     #predictedLabel = labelInfo[str(np.argmax(preds[0]))]
     predictedLabel = labelInfo[str(preds[0])]
     print(predictedLabel)
     context = {'filePathName': filePathName, 'predictedLabel': predictedLabel[1]}
     return render(request2, 'insect.html', context)


def viewDataBase2(request):
    import os
    listOfImages=os.listdir('./media/')
    listOfImagesPath=['./media/'+i for i in listOfImages]
    context={'listOfImagesPath':listOfImagesPath}
    return render(request,'viewDB2.html',context)