import cv2
import glob
import math
import os
import numpy as np

class DCP:
    def __init__(self, image, patch_size):
        self.image = image
        self.patch_size = patch_size
        self.omega = 0.95
        self.A = self.get_A(image, self.getDarkChannel(image))

    def get_A(self, image, dcp):
        [h, w] = image.shape[:2]
        image_size = h * w
        numpx = int(max(math.floor(image_size / 1000), 1))
        darkvec = dcp.reshape(image_size)
        imvec = image.reshape(image_size, 3)

        indices = darkvec.argsort()
        indices = indices[image_size - numpx::]

        atmsum = np.zeros([1, 3])
        for ind in range(1, numpx):
            atmsum = atmsum + imvec[indices[ind]]

        A = atmsum / numpx
        return A

    def getDarkChannel(self, image):
        b, g, r = cv2.split(image)
        dc = cv2.min(cv2.min(r, g), b)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.patch_size, self.patch_size))
        dark = cv2.erode(dc, kernel)
        return dark

    def estimateTransmission(self, image, threshold):
        image2 = np.empty(image.shape, image.dtype)

        for ind in range(0, 3):
            image2[:, :, ind] = image[:, :, ind] / self.A[0, ind]

        transmission = 1 - self.omega * self.getDarkChannel(image2)
        
        if threshold is not None:
            transmission[transmission < threshold] = 1.
            
        return transmission

    def guidedFilter(self, image, p, r, eps):
        mean_I = cv2.boxFilter(image, cv2.CV_64F, (r, r))
        mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
        mean_Ip = cv2.boxFilter(image * p, cv2.CV_64F, (r, r))
        cov_Ip = mean_Ip - mean_I * mean_p

        mean_II = cv2.boxFilter(image * image, cv2.CV_64F, (r, r))
        var_I = mean_II - mean_I * mean_I

        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I

        mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
        mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))

        q = mean_a * image + mean_b

        return q

    def refineTransmission(self, image, et):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = np.float64(gray) / 255
        
        r = 60
        eps = 0.0001
        t = self.guidedFilter(gray, et, r, eps)

        return t

    def getRecoverImage(self, im, t, A, tx=0.1):
        res = np.empty(im.shape, im.dtype)
        t = cv2.max(t, tx)

        for ind in range(0, 3):
            res[:, :, ind] = (im[:, :, ind] - A[0, ind]) / t + A[0, ind]

        return res


if __name__ == "__main__":
    load_path = "/Users/sungyoon-kim/Downloads/BeDDE_part"
    save_path = "/Users/sungyoon-kim/Downloads/BeDDE_part/transmission_dcp/"

    for folder in glob.glob(load_path + '/*'):
        file_path = os.path.join(folder, 'fog')
        for file in glob.glob(file_path + '/*'):
            src = cv2.imread(file)
            image = src.astype('float64') / 255
            dcp = DCP(image, patch_size=15)

            transmission = dcp.estimateTransmission(image, threshold=0.2)
            # cv2.imshow('transmission', transmission)
            # cv2.waitKey(0)
            t = dcp.refineTransmission(src, transmission)
            # cv2.imshow('refined transmission', t)
            # cv2.waitKey(0)

            # Recover
            # J = Recover(img, t, A, 0.1)
            # J2 = Recover(img, t2, A, 0.1)
            # addhJ = cv2.hconcat([J, J2])
            # cv2.imshow('J', addhJ)
            # cv2.waitKey()

            # cv2.imshow('Dark Channel', addh)
            # cv2.waitKey(0)

            save_file = os.path.join(save_path, os.path.basename(file).split('.')[0])
            cv2.imwrite(save_file + '.jpeg', addh * 255)
