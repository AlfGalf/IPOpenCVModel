'''
Python code to capture video and take 3 frames detecting the contour of an object (hand), 
which will be passed on to ML model for prediction
'''

import cv2 as cv
import numpy as np


bgd = None
frames_for_prediction=[]
input_weight = 0.5
#Area where hand has to be to be shown
box_top = 200
box_bottom = 600
box_right = 300
box_left = 700
#Amount of frames for ML model
start_captured_frame=200
end_of_captured_frame=start_captured_frame+3

#Method to show frames captured
def show_frames(frames_captured):
    if len(frames_captured)==0:
        return
    counter=0
    for i in frames_captured:
        counter=counter+1
        title="Prediction to model number "+str(counter)
        cv.imshow(title,i)
        #Close images by pressing 0
        cv.waitKey(0)
    cv.destroyAllWindows()

#Method to detect the background, this is needed to convert the difference when calculating contours of hand
def bgd_detection(frame, input_weight):

    global bgd
    
    if bgd is None:
        bgd = frame.copy().astype("float")
        return None

    cv.accumulateWeighted(frame, bgd, input_weight)

#Method to calculate the contours of hand
def segment_hand(frame, threshold=25):
    global bgd
    #image to train for prediction thresholded
    diff = cv.absdiff(bgd.astype("uint8"), frame)

    #y is just unused variable neccesary due to threshold method
    y , thresholded = cv.threshold(diff, threshold, 255, cv.THRESH_BINARY)
    
    
    image=thresholded.copy()
    #Looking for the contours in the frame recorded
    contours, hierarchy = cv.findContours(thresholded.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # To prevent errors if di nit detect contours exit the method
    if len(contours) == 0:
        return
    else:
        # The calculated contour is the hand
        hand_segment_max_cont = max(contours, key=cv.contourArea)
        
        # Return the image in binary and contours
        return (thresholded, hand_segment_max_cont)




cam = cv.VideoCapture(0)
num_frames =0
while True:
    ret, frame = cam.read()

    # inverting the image due to mirror effect
    frame = cv.flip(frame, 1)

    frame_copy = frame.copy()

    # box from the frame
    box_image = frame[box_top:box_bottom, box_right:box_left]
    #converting image to grayscale for absdiff method
    gray_frame = cv.cvtColor(box_image, cv.COLOR_BGR2GRAY)
    #smoothing the image
    gray_frame = cv.GaussianBlur(gray_frame, (9, 9), 0)

    #The first 70 frames will be used to detect the background
    if num_frames < 70:
        
        bgd_detection(gray_frame, input_weight)
        #Warn the user not to move anything inside the box area yet
        cv.putText(frame_copy, "Capturing background, please wait", (80, 100), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
    
    else: 
        # segmenting the hand region
        hand = segment_hand(gray_frame)
        

        # Add fps here to inform when we are recording input
        cv.putText(frame_copy, "Please do the ASL gesture you wish to know, will capture frames in x seconds", (80, 100), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
        # Checking if we are able to detect the hand
        if hand is not None:
           
                
            thresholded, hand_segment = hand

            # Drawing contours around hand segment
            cv.drawContours(frame_copy, [hand_segment + (box_right, box_top)], -1, (255, 0, 0),1)
            
            if num_frames>=start_captured_frame and num_frames<end_of_captured_frame:
                #Adding the frames we wish to predict
                frames_for_prediction.append(thresholded)
            '''
            Idea for model prediction (Resizing of images and passing it to model which would be loaded at the start as an .h5 file (TF Model))
            #thresholded = cv.resize(thresholded, (64, 64))
            #thresholded = cv.cvtColor(thresholded, cv.COLOR_GRAY2RGB)
            #thresholded = np.reshape(thresholded, (1,thresholded.shape[0],thresholded.shape[1],3))
            #array of frames we wish to train
            
                
            #64 x64 image, (1,64,64,3) RGB -->0-255
            #print(thresholded)
            #pred = model.predict(thresholded)

            '''
    # Draw box on window we are showing the user(frame_copy), parameters: start_point,end_point, colour and thickness

    cv.rectangle(frame_copy, (box_left, box_top), (box_right, box_bottom), (255,128,0), 3)

    # incrementing the number of frames for tracking
    num_frames += 1
    #Window we show to the user
    cv.imshow("Sign Detection", frame_copy)
    


    # Close windows with Esc if we wish to finish 
    k = cv.waitKey(1) & 0xFF

    if k == 27 or num_frames==end_of_captured_frame:
        break
show_frames(frames_for_prediction)



#Example

#new_recording=video_recording(3)
#show_frames(new_recording)