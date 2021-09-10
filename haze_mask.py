import os, glob, cv2, math
import numpy as np

def DarkChannel(image,sz):
    b,g,r = cv2.split(image)
    dc = cv2.min(cv2.min(r,g),b);
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(sz,sz))
    dark = cv2.erode(dc,kernel)
    return dark

def TransmissionEstimate(image,A,sz):
    omega = 0.95;
    im3 = np.empty(image.shape,image.dtype);

    for ind in range(0,3):
        im3[:,:,ind] = image[:,:,ind]/A[0,ind]

    transmission = 1 - omega*DarkChannel(im3,sz);
    return transmission

def Guidedfilter(im,p,r,eps):
    mean_I = cv2.boxFilter(im,cv2.CV_64F,(r,r));
    mean_p = cv2.boxFilter(p, cv2.CV_64F,(r,r));
    mean_Ip = cv2.boxFilter(im*p,cv2.CV_64F,(r,r));
    cov_Ip = mean_Ip - mean_I*mean_p;

    mean_II = cv2.boxFilter(im*im,cv2.CV_64F,(r,r));
    var_I   = mean_II - mean_I*mean_I;

    a = cov_Ip/(var_I + eps);
    b = mean_p - a*mean_I;

    mean_a = cv2.boxFilter(a,cv2.CV_64F,(r,r));
    mean_b = cv2.boxFilter(b,cv2.CV_64F,(r,r));

    q = mean_a*im + mean_b;
    q2 = mean_a*im + mean_b;
    
    q2[q2 < 0.2] = 1.
    
    return q, q2;

def TransmissionRefine(im,et):
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);
    gray = np.float64(gray)/255;
    r = 60;
    eps = 0.0001;
    t, t2 = Guidedfilter(gray,et,r,eps);

    return t, t2;

def AtmLight(im,dark):
    [h,w] = im.shape[:2]
    imsz = h*w
    numpx = int(max(math.floor(imsz/1000),1))
    darkvec = dark.reshape(imsz);
    imvec = im.reshape(imsz,3);

    indices = darkvec.argsort();
    indices = indices[imsz-numpx::]

    atmsum = np.zeros([1,3])
    for ind in range(1,numpx):
       atmsum = atmsum + imvec[indices[ind]]

    A = atmsum / numpx;
    return A

def Recover(im,t,A,tx = 0.1):
    res = np.empty(im.shape,im.dtype);
    t = cv2.max(t,tx);

    for ind in range(0,3):
        res[:,:,ind] = (im[:,:,ind]-A[0,ind])/t + A[0,ind]

    return res

if __name__=="__main__":
    load_path = "/Users/sungyoon-kim/Downloads/BeDDE_part"
    save_path = "/Users/sungyoon-kim/Downloads/BeDDE_part/transmission_dcp/"
    
    for folder in glob.glob(load_path + '/*'):
        file_path = os.path.join(folder, 'fog')
        for file in glob.glob(file_path + '/*'):
            src = cv2.imread(file)
            img = src.astype('float64')/255
            
            dark = DarkChannel(img, 15)
            A = AtmLight(img, dark);
            te = TransmissionEstimate(img, A, 15)
            t, t2 = TransmissionRefine(src, te)
            
            # Recover
            # J = Recover(img, t, A, 0.1)
            # J2 = Recover(img, t2, A, 0.1)
            # addhJ = cv2.hconcat([J, J2])
            # cv2.imshow('J', addhJ)
            # cv2.waitKey()
            
            addh = cv2.hconcat([t, t2])
            # cv2.imshow('Dark Channel', addh)
            # cv2.waitKey(0)
            
            save_file = os.path.join(save_path, os.path.basename(file).split('.')[0])
            cv2.imwrite(save_file +  '.jpeg', addh*255)
    
