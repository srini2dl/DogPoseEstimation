import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd
from skimage import io

# Opens the Video file
cap = cv2.VideoCapture('videos/test.mp4')
i = 0
checkFrame = 60
while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    if i == checkFrame:
        imagePath = 'videos/sample_frame' + str(i) + '.jpg'
        cv2.imwrite(imagePath, frame)
        break
    i += 1

cap.release()
cv2.destroyAllWindows()
hdfPath = 'videos/testshuffle1_7500.h5'
predictedDataFrame = pd.read_hdf(hdfPath, 'df_with_missing')
plt.axis('off')
im=io.imread(imagePath)
h,w,numcolors=np.shape(im)
dotsize = 15
plt.figure(frameon=False,figsize=(w*2./100,h*1./100))
plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
plt.imshow(im,'gray')
bodyParts = ['L_F_Paw',
             'L_F_Knee',
             'L_F_Elbow',
             'L_B_Paw',
             'L_B_Knee',
              'L_B_Elbow',
              'R_F_Paw',
              'R_F_Knee',
              'R_F_Elbow',
              'R_B_Paw',
              'R_B_Knee',
              'R_B_Elbow',
              'TailBase',
              'L_EarBase',
              'R_EarBase',
              'Nose',
              'L_Eye',
              'R_Eye',
              'Withers',
              'Throat']
for bp in bodyParts:
    print(bp)
    if predictedDataFrame['shuffle1_7500'][bp]['likelihood'].values[checkFrame] > 0.5:
        x = predictedDataFrame['shuffle1_7500'][bp]['x'].values[checkFrame]
        y = predictedDataFrame['shuffle1_7500'][bp]['y'].values[checkFrame]
        plt.plot(x, y, '.', ms=dotsize, alpha=0.7)
plt.show()
