# IPOpenCVModel

## Usage
Clone the repository: https://github.com/AlfGalf/IPOpenCVModel.git  
Run: python3 thresholding.py


## **image_reading.py:**   
Python code to read two image files (a background and the image) and threshold out the backgorund image with an object (hand), which will be passed on to ML model for prediction.      
**Functionality** is as follows, provide the filepath for the two images mentioned earlier on. Then we calculate the difference in pixels so we can recognise the background and therefore ignore it. Once this has been done we proceed to threshold the image with a result which is a new image in black and white.
**Testing** in order to test the code you can uncomment the last line and run it. You will have to provide two filepaths of images which are similar (one which is the background and the other one doing the gesture. 

**ML Work** ,the script currently returns the rsulting image and writes it into the directory where it is executed. Current idea is to provide 4 frames, (the background and 3 frames of the gesture). Call the function 3 times and will write the 3 images to folder and then ML script can read these images or directly call function in ML model and will return each image.

**Example**![bgd](https://user-images.githubusercontent.com/60605841/109489639-1499ef80-7a7f-11eb-8534-e4df2dc4140e.jpg)![hand](https://user-images.githubusercontent.com/60605841/109489746-3a26f900-7a7f-11eb-9ddd-bd63c071c3ed.jpg)
![thresholded_image](https://user-images.githubusercontent.com/60605841/109489809-4f038c80-7a7f-11eb-8ca7-29a858bbca82.jpg)
