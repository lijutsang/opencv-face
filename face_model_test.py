#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
1. names[]中存储人的名字，若该人id为0则他的名字在第一位，id位1则排在第二位，以此类推。
2. 最终效果为一个绿框，框住人脸，左上角为红色的人名，左下角为黑色的概率。
'''
import cv2 #导入opencv库 计算机视觉库
import os 
from PIL import Image 

def paint_chinese_opencv(im,chinese,position,fontsize,color):#opencv输出中文
    img_PIL = Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))# 图像从OpenCV格式转换成PIL格式
    font = ImageFont.truetype('simhei.ttf',fontsize,encoding="utf-8")
    #color = (255,0,0) # 字体颜色
    #position = (100,100)# 文字输出位置
    draw = ImageDraw.Draw(img_PIL)
    draw.text(position,chinese,font=font,fill=color)# PIL图片上打印汉字 # 参数1：打印坐标，参数2：文本，参数3：字体颜色，参数4：字体
    img = cv2.cvtColor(np.asarray(img_PIL),cv2.COLOR_RGB2BGR)# PIL图片转cv2 图片
    return img
'''
使用方法：
img = paint_chinese_opencv(image,str,point,color)
//Mat 图片，string "中文"，Point (30,30),color (0,0,255)
cv2.imshow('putText显示添加文字操作的图像', img)  # 显示添加文字操作的图像
cv2.waitKey()
'''
dir_path = os.path.abspath(os.path.dirname(__file__))#当前绝对路径
#初始化模型
recognizer = cv2.face.LBPHFaceRecognizer_create()
#载入训练好的人脸模型
print('载入训练好的人脸模型:',dir_path +'/face_model/trainer.yml')
recognizer.read(dir_path +'/face_model/trainer.yml')
cascadePath = dir_path +"/model/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
#窗口设置字体
font = cv2.FONT_HERSHEY_SIMPLEX

idnum = 0
#设置已知姓名列表
names = ['liju', 'unknow']

cam = cv2.VideoCapture(0)
minW = 0.1*cam.get(3)#设置窗口宽度
minH = 0.1*cam.get(4)#设置窗口高度
ok = cam.isOpened()
if(ok):
    print('打开摄像头成功!')
    print('开始实时检测！')
else:
    print('打开摄像头失败，请检查设备!')

while ok:
    
    #捕获视频图片
    ret, img = cam.read()
    #彩色图转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #设置人脸检测输入参数设置
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH))
    )
    #在识别出的人脸位置用绿色框标出，右下角黑色字显示程序识别置信度，左上角红色显示名字
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        idnum, confidence = recognizer.predict(gray[y:y+h, x:x+w])
        # 显示置信度小于100的已知人脸
        if confidence < 100:
            idnum = names[idnum]
            confidence = "{0}%".format(round(100 - confidence))
        else:
            idnum = "unknown" #返回未知
            confidence = "{0}%".format(round(100 - confidence))
        #标记识别百分比和名字
        cv2.putText(img, str(idnum), (x+5, y-5), font, 1, (0, 0, 255), 1)
        cv2.putText(img, str(confidence), (x+5, y+h-5), font, 1, (0, 0, 0), 1)

    cv2.imshow('camera', img)
    k = cv2.waitKey(10)
    if k == 27: # 通过esc键退出采集人脸数据
        break

cam.release()
cv2.destroyAllWindows()