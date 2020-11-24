import cv2
import matplotlib.pyplot as plt # plt 用於顯示圖片
import matplotlib.image as mpimg # mpimg 用於讀取圖片
import numpy as np
import os.path
import glob
def convertjpg(jpgfile,outdir,width=230,height=230):
    
    img = cv2.imread(jpgfile,cv2.COLOR_BGR2RGB)
    
    try:
        img = cv2.resize(img, (round(img.shape[1]/2),round(img.shape[0]/2)), interpolation=cv2.INTER_CUBIC) 
        mask = np.zeros(img.shape[:2], np.uint8)# 创建大小相同的掩模
        bgdModel = np.zeros((1,65), np.float64)# 创建背景图像
        fgdModel = np.zeros((1,65), np.float64)# 创建前景图像
        x0=(17*(img.shape[1]))/230
        x1_w=(207*(img.shape[1]))/230
        y0=(20*(img.shape[0]))/230
        y1_h=(206*(img.shape[0]))/230
        rect = (round(x0),round(y0),round(x1_w),round(y1_h))
        cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
        img = img*mask2[:,:,np.newaxis]
        cv2.imwrite(os.path.join(outdir,os.path.basename(jpgfile)), img)
    except Exception as e:
        print(e)
        
for jpgfile in glob.glob(r'C:\Users\user\Desktop\Aldea\secondrelease\C1-P1_Train\*.jpg'):
    convertjpg(jpgfile,r'C:\Users\user\Desktop\Aldea\secondrelease\opencvoutputtrain')