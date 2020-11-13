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
#     model=load_model('./models/plant.h5')

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
     if eid1=='plant':
         modelPath='./models/plantv4_5000.h5'
         clas_data = './models/plantclassv4.json'
         classs=20
     else:
        modelPath='./models/insect12_8000.h5'
        clas_data = './models/insectclassv2.json'
        classs=12

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
     img_arr = np.array(img_arr, dtype="float") / 255.0
     img_arr = img_arr.astype('float32') / 255
     model_graph = Graph()

     with model_graph.as_default():
         print("plant")
         model = load_model(modelPath)
         predss = model.predict_classes(img_arr)
         #改成predict
         preds=model.predict(img_arr)

     print(preds)
     #最大
     print("最大的數:{}, 位於{}".format(np.max(preds), np.argmax(preds)))
     predmax = np.argmax(preds)

     preds_ = preds
     # 第二大
     preds_[0][np.argmax(preds,axis=1)] = np.min(preds)
     print(preds_)
     print("第二大的數:{}, 位於{}".format(np.max(preds_), np.argmax(preds_)))
     predtwo=np.argmax(preds_)

     # 第二大
     preds_[0][np.argmax(preds_, axis=1)] = np.min(preds)
     print(preds_)
     print("第三大的數:{}, 位於{}".format(np.max(preds_), np.argmax(preds_)))
     predthr = np.argmax(preds_)


     # 換算機率
     sum = preds.sum()
     print(sum)

     predictedLabel = []

     print("predmax")
     print(predmax)
     predictedLabel1 = labelInfo[str(predmax)]
     print("predict max")
     print(predictedLabel1)
     predictedLabel.append(predictedLabel1[1])
     #pp=preds[np.argmax(preds)]
     #print(pp)
     #print(pp[0]/sum)

     print("predtwo")
     print(predtwo)
     predictedLabel2 = labelInfo[str(predtwo)]
     print("predict two")
     print(predictedLabel2)
     #pp2 = preds_[np.argmax(preds_)]
     #print(pp2)
     #print(pp2[0] / sum)

     if predictedLabel1[1] not in predictedLabel:
         predictedLabel.append(predictedLabel2[1])


     print("predthr")
     print(predthr)
     predictedLabel3 = labelInfo[str(predthr)]
     print("predict 3")
     print(predictedLabel3)

     if predictedLabel2[1] not in predictedLabel:
         predictedLabel.append(predictedLabel3[1])

     #class
     print(predss[0])
     print("pred_class")
     print(labelInfo[str(predss[0])])

     #predictedLabel2 = labelInfo[str(np.argmax(preds[1]))]
     #predictedLabel = labelInfo[str(preds[0])]




     context = {'filePathName': filePathName, 'predictedLabel': predictedLabel}
     return render(request, 'index.html', context)



def viewDataBase(request):
    import os
    listOfImages=os.listdir('./media/')
    listOfImagesPath=['./media/'+i for i in listOfImages]
    context={'listOfImagesPath':listOfImagesPath}
    return render(request,'viewDB.html',context)

