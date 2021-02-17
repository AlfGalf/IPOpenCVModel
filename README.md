# IPOpenCVModel
## **image_reading.py:**   
Python code to capture video and take 3 frames detecting the contour of an object (hand), which will be passed on to ML model for prediction.      
**Functionality** is as follows, open a new camera recording using the OpenCV2 library. Record the frames and abstract the frames corresponding 
to a box (this is the input zone for the user to do the gesture). With this frames we can know calculate the difference with the ones of  
the user doing the gesture to obtain a threshold image. Save 3 frames so we can pass them on to the ML model and make an accurate prediction  

**Testing** in order to test the code you can uncomment the last line and run it. It should open an instance of a video recording which  
will later on show how much time is left until we record the frames, as for testing it will show the 3 frames recorded.  

**Further Work** frame format is currently an array of with numbers varying from 0-255 for the 3 numbers in RGB




