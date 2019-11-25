#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
第一步 人脸数据集收集
1.在运行该程序前，请先创建一个facedata文件夹并和你的程序放在一个文件夹下。
2.程序运行过程中，会提示你输入id，请从0开始输入，即第一个人的脸的数据id为0，第二个人的脸的数据id为1，运行一次可收集一张人脸的数据。
3.程序运行时间可能会比较长，可能会有几分钟，如果嫌长，可以将     
#得到1000个样本后退出摄像      这个注释前的1000，改为100。
如果实在等不及，可按esc退出，但可能会导致数据不够模型精度下降。
'''
# here put the import code
from cv2 import cv2
import os
from time import sleep

dir_path = os.path.abspath(os.path.dirname(__file__))
# 调用笔记本内置摄像头，所以参数为0，如果有其他的摄像头可以调整参数为1,2
cap = cv2.VideoCapture(0)
#导入人脸模型
face_detector = cv2.CascadeClassifier(dir_path +'/model/haarcascade_frontalface_default.xml')
#输入用户id:,一个人脸对应一个id [0,1,2,...]
face_id = input('\n 输入用户ID,enter user id:')
print('\n 开始捕获用户图片创建数据集，等待摄像头初始化 ...')

count = 0
while True:
    # 从摄像头读取图片
    sleep(0.1)
    sucess, img = cap.read()
    # 转为灰度图片
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 检测人脸
    faces = face_detector.detectMultiScale(gray, 1.3, 5) '''
    1.image表示的是要检测的输入图像2.objects表示检测到的人脸目标序列3.scaleFactor表示每次图像尺寸减小的比例
    4.minNeighbors表示每一个目标至少要被检测到3次才算是真的目标(因为周围的像素和不同的窗口大小都可以检测到人脸),
5.minSize为目标的最小尺寸
6.minSize为目标的最大尺寸
'''
    # 获得人脸在图像中的坐标值
    for (x, y, w, h) in faces:
        # 在人脸区域绘制矩形框
        cv2.rectangle(img, (x, y), (x+w, y+w), (255, 0, 0))
        count += 1
        print('\n 采集第{}张图片'.format(count)) 
        print('\n 通过esc键退出采集人脸数据')
        # 保存图像
        cv2.imwrite(dir_path+"/facedata/User." + str(face_id) + '.' + str(count) + '.jpg', gray[y: y + h, x: x + w])
        cv2.imshow('image', img)
        
    # 保持画面的持续。
    k = cv2.waitKey(1)
    if k == 27:   # 通过esc键退出采集人脸数据  cv2.waitKey(1) & 0xFF == ord('q'):
        print('key:',k)
        break
    if count >= 100:  # 得到n个样本后退出摄像
        break

# 关闭摄像头
cap.release()
cv2.destroyAllWindows()