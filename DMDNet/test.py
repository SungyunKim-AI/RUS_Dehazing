import cv2
import numpy as np
from Module_Metrics import Entropy_Module

if __name__=='__main__':
    
    ent = Entropy_Module.Entropy_Module()
    f1 = cv2.imread('13_0.06499999999999999.jpg')
    f_ = cv2.imread('15_0.075.jpg')
    f2 = cv2.imread('17_0.06499999999999999.jpg')
    print(f1)
    cv2.imshow('f1',f1)
    cv2.imshow('f2',f2)
    cv2.imshow('diff_img',f2-f1)

    f1_ent = ent.get_cur(f1/255)
    f_ent = ent.get_cur(f_/255)
    f2_ent = ent.get_cur(f2/255)
    print(f1_ent, f_ent, f2_ent)
    print(np.sum(np.abs(f1-f2))/(256*256*3))
    cv2.waitKey(0)