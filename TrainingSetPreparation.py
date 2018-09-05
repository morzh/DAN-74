#coding=utf-8
from ImageServer import ImageServer
import numpy as np

'''
imageDirsPrefix = '/media/morzh/ext4_volume/data/Faces/data/images_noseShape/'
bboxesPrefix = '/home/morzh/work/Deep-Alignment-Network-tensorflow/data/'
datasetDir = "/media/morzh/ext4_volume/work/Deep-Alignment-Network-tensorflow/data/"
'''

imageDirsPrefix = '/data/images/'
bboxesPrefix = '/data/'
datasetDir = "/output/"

imageDirs = ["lfpw/trainset/", "helen/trainset/", "afw/", "300W/01_Indoor/", "300W/02_Outdoor/", "ibug/"]
boundingBoxFiles = ["py3boxesLFPWTrain.pkl", "py3boxesHelenTrain.pkl", "py3boxesAFW.pkl", "py3boxes300WIndoor.pkl", "py3boxes300WOutdoor.pkl", "py3boxesIBUG.pkl"]

imageDirs = [imageDirsPrefix+s for s in imageDirs]
boundingBoxFiles = [bboxesPrefix+s for s in boundingBoxFiles]



meanShape = np.load(bboxesPrefix+"meanFaceShape74.npz")["meanShape"]

trainSet = ImageServer(initialization='rect')#相当于没有用bbx做训练，直接用的特征点截取框
trainSet.PrepareData(imageDirs, None, meanShape, 100, 100000, True)#准备好图片名list，对应图片landmark的list，和对应图片的bbx的list，和meanshape。令我疑惑的是，startIdx=100，nImgs=100000，,300W数据集可没有那么多图片
trainSet.LoadImages()#读取图片，并对每张图调整好meanShape
trainSet.GeneratePerturbations(10, [0.2, 0.2, 20, 0.25])#位移0.2，旋转20度，放缩+-0.25
# import pdb; pdb.set_trace()
trainSet.NormalizeImages()#去均值，除以标准差
trainSet.Save(datasetDir)#保存成字典形式，key为'imgs'，'initlandmarks'，'gtlandmarks'


validationSet = ImageServer(initialization='box')
validationSet.PrepareData(imageDirs, boundingBoxFiles, meanShape, 0, 100, False)
validationSet.LoadImages()
validationSet.CropResizeRotateAll()
# import pdb
# pdb.set_trace()
validationSet.imgs = validationSet.imgs.astype(np.float32)
validationSet.NormalizeImages(trainSet)
validationSet.Save(datasetDir)
