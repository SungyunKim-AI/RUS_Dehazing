import cv2
import matplotlib.pyplot as plt

def show_plt(x, y):
    plt.bar(x, y, align='center')
    plt.xlabel('deg.')
    plt.xlim([x[0], x[-1]])
    plt.ylabel('prob.')
    for i in range(len(y)):
        plt.vlines(x[i], 0, y[i])
    
    plt.show()
    
def show_img(imgName, img):
    cv2.imshow(imgName, img.astype('uint8'))
    cv2.waitKey(0)