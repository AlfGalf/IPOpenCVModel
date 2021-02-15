import cv2 as cv
import numpy as np
'''
img=cv.imread('snake.jpeg')

cv.imshow('Snake',img)
'''



bgd = None
frames_for_prediction=[]
input_weight = 0.5
box_top = 200
box_bottom = 600
box_right = 300
box_left = 700
start_captured_frame=200
end_of_captured_frame=start_captured_frame+3

def show_frames(frames_captured):
    if len(frames_captured)==0:
        return
    counter=0
    for i in frames_captured:
        counter=counter+1
        title="Prediction to model number "+str(counter)
        #cv.putText(i, "Press 0 to close this will be one of the images passed to the training model", (40, 100), cv.FONT_HERSHEY_SIMPLEX, 0.25, (255,255,255), 2)
        cv.imshow(title,i)
        
        cv.waitKey(0)
    #capture.release()
    cv.destroyAllWindows()


def bgd_detection(frame, input_weight):

    global bgd
    
    if bgd is None:
        bgd = frame.copy().astype("float")
        return None

    cv.accumulateWeighted(frame, bgd, input_weight)

def segment_hand(frame, threshold=25):
    global bgd
    #image to train for prediction thresholded
    diff = cv.absdiff(bgd.astype("uint8"), frame)

    #y is just unused variable neccesary due to threshold method
    y , thresholded = cv.threshold(diff, threshold, 255, cv.THRESH_BINARY)
    
    #Fetching contours in the frame
    image=thresholded.copy()
    
    contours, hierarchy = cv.findContours(thresholded.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # If length of contours list = 0, means we didn't get any contours...
    if len(contours) == 0:
        return
    else:
        # The largest external contour should be the hand 
        hand_segment_max_cont = max(contours, key=cv.contourArea)
        
        # Returning the hand segment(max contour) and the thresholded image of hand...
        return (thresholded, hand_segment_max_cont)

cam = cv.VideoCapture(0)
num_frames =0
while True:
    ret, frame = cam.read()

    # filpping the frame to prevent inverted image of captured frame...
    frame = cv.flip(frame, 1)

    frame_copy = frame.copy()

    # box from the frame
    box_image = frame[box_top:box_bottom, box_right:box_left]
    #converting image to grayscale for absdiff method
    gray_frame = cv.cvtColor(box_image, cv.COLOR_BGR2GRAY)
    #smoothing the image
    gray_frame = cv.GaussianBlur(gray_frame, (9, 9), 0)


    if num_frames < 70:
        
        bgd_detection(gray_frame, input_weight)
        
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
            
            #cv.imshow("Thesholded Hand Image", thresholded)
            if num_frames>=start_captured_frame and num_frames<end_of_captured_frame:
                frames_for_prediction.append(thresholded)
            
            #thresholded = cv.resize(thresholded, (64, 64))
            #thresholded = cv.cvtColor(thresholded, cv.COLOR_GRAY2RGB)
            #thresholded = np.reshape(thresholded, (1,thresholded.shape[0],thresholded.shape[1],3))
            #array of frames we wish to train
            
                
            #64 x64 image, (1,64,64,3) RGB -->0-255
            #print(thresholded)
            #pred = model.predict(thresholded)
            #cv2.putText(frame_copy, word_dict[np.argmax(pred)], (170, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            
    # Draw ROI on frame_copy, start_point,end_point, colour and thickness

    cv.rectangle(frame_copy, (box_left, box_top), (box_right, box_bottom), (255,128,0), 3)

    # incrementing the number of frames for tracking
    num_frames += 1

    # Display the frame with segmented hand
    #cv.putText(frame_copy, "DataFlair hand sign recognition_ _ _", (10, 20), cv.FONT_ITALIC, 0.5, (51,255,51), 1)
    cv.imshow("Sign Detection", frame_copy)
    


    # Close windows with Esc
    k = cv.waitKey(1) & 0xFF

    if k == 27 or num_frames==end_of_captured_frame:
        break
show_frames(frames_for_prediction)
'''
cam.release()
cv.destroyAllWindows()
'''
'''
def video_recording(amount_of_frames):
    capture=cv.VideoCapture(0)
    counter=0
    frames=[]
    while True:
        counter=counter+1
        isTrue, frame=capture.read()
        print(counter)
        #cv.imshow('Video',frame)
        print(frame.shape)
        #if cv.waitKey(20) & 0xFF==ord('d'):
        frames.append(frame)
        if cv.waitKey(20) & counter>amount_of_frames-1:
            break
        
    return frames
'''


#Example

#new_recording=video_recording(3)
#show_frames(new_recording)