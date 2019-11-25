#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
1.运行该程序前，请在人脸识别文件夹下创建face_trainer文件夹。
'''
# here put the import code
import numpy as np
from PIL import Image
import os
import cv2
import os
dir_path = os.path.abspath(os.path.dirname(__file__))

# 人脸数据路径
path = dir_path +'/facedata'
#导入训练模型
recognizer = cv2.face.LBPHFaceRecognizer_create()
#加载分类器
detector = cv2.CascadeClassifier(dir_path +"/model/haarcascade_frontalface_default.xml")

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]  # 
    faceSamples = []
    ids = [] 
    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L')   # 处理模型输入数据
        img_numpy = np.array(PIL_img, 'uint8')
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)
        for (x, y, w, h) in faces:
            faceSamples.append(img_numpy[y:y + h, x: x + w])
            ids.append(id)
    return faceSamples, ids


print('Training faces. It will take a few seconds. Wait ...')
print('正在训练模型，请等待...')
faces, ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))

recognizer.write(dir_path +'/face_trainer/trainer.yml')
print("{0} 人脸模型训练完毕，退出".format(len(np.unique(ids))))
print("模型保持在："dir_path +'/face_trainer/trainer.yml')