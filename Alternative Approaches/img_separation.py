import cv2 as cv
import numpy as np

def preprocess_image(image):
    bilateral_filtered_image = cv.bilateralFilter(image, 7, 150, 150)
    gray_image = cv.cvtColor(bilateral_filtered_image, cv.COLOR_BGR2GRAY)
    
    return gray_image

def thresholding(bgd_path,img_path):
    bgd=preprocess_image(cv.imread(bgd_path))
    img=preprocess_image(cv.imread(img_path))
    img_sep=cv.absdiff(bgd,img)
    img_sep=cv.flip(img_sep,1)
    return img_sep

def post_process(image):
    kernel = np.ones((5,5),np.uint8)
    close_operated_image = cv.morphologyEx(image, cv.MORPH_CLOSE, kernel)
    _, thresholded = cv.threshold(close_operated_image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    thr = cv.medianBlur(thresholded, 5)
    
    return thr




img_thres=thresholding('test1.jpg','test2.jpg')
result=post_process(img_thres)
cv.imshow('i',result)
cv.waitKey(0)
cv.destroyAllWindows()
