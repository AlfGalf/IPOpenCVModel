import cv2 as cv
import numpy as np

def segment_hand(bgd_filename,frame,threshold=25):
    bgd=cv.imread(bgd_filename)
    bgd=cv.flip(bgd,1)
    #cv.accumulateWeighted(bgd,bgd,0.5)
    bgd= cv.cvtColor(bgd, cv.COLOR_BGR2GRAY)
    gray_bgd = cv.GaussianBlur(bgd, (9, 9), 0)
    
    #image to train for prediction thresholded
    diff = cv.absdiff(bgd.astype("uint8"), frame)

    #y is just unused variable neccesary due to threshold method
    y , thr_img = cv.threshold(diff, threshold, 255, cv.THRESH_BINARY)
    
    image=thr_img.copy()
    #Looking for the contours in the frame recorded
    contours, hierarchy = cv.findContours(thr_img.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    # To prevent errors if it did not detect contours exit the method
    if len(contours) == 0:
        return
    else:
        # The calculated contour is the hand
        hand_cont = max(contours, key=cv.contourArea)
        
        # Return the image in binary and contours
        return (thr_img, hand_cont)

def img_thresholding(bgd_filename,frame_filename):
    bgd_path=bgd_filename
    frame=cv.imread(frame_filename)
    
    # inverting the image due to mirror effect
    frame = cv.flip(frame, 1)
    frame_copy = frame.copy()    
    #converting image to grayscale for absdiff method
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    #smoothing the image
    gray_frame = cv.GaussianBlur(gray_frame, (9, 9), 0)

    hand = segment_hand(bgd_path,gray_frame)

    if hand is not None:            
        thr_img, hand_contoured = hand
    thr = cv.medianBlur(thr_img, 5)
    cv.imwrite('thresholded_image.jpg',thr)
    return thr_img

#Calls fuction on filenames
img=img_thresholding('bgd.jpg','hand.jpg')
#cv.imshow('image',img) Uncomment to display image
#cv.waitKey(0)
cv.destroyAllWindows()