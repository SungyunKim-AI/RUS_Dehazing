"""
Reference Paper
Image Haze Removal Using Airlight White Correction, Local Light Filter, and Aerial Perspective Prior
Yan-Tsung Peng et al.
"""
import cv2
import math
import numpy as np
from scipy.stats import rv_discrete

def RGB2HSV(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, S, V = cv2.split(hsv)        # Hue, Saturation, Value
    return hue
    
def PMF_H(hue):
    xk = np.arange(360)
    print(np.histogram(hue))
    pmf = rv_discrete.pmf(xk)
    
    k = math.floor(x)
    xlogy = lambda x, y : x*math.log(y)
    xlog1py = lambda x, y : x*math.log1p(y)
    logpmf = lambda n, p : math.lgamma(n+1)-math.lgamma(k+1)-math.lgamma(n-k+1)+xlogy(k,p)+xlog1py(n-k,-p)
    
    P_H = math.exp(logpmf(x,n,p))
    T_H = image.shape / 360
    
    if P_H > T_H:
        return 1
    else:
        return 0

if __name__ == "__main__":
    image = cv2.imread()