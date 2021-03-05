'''
Import sys

'''
import os
import sys
sys.path.append(os.path.join(".."))
import cv2
import numpy as np
from PIL import Image



''' 
Reading file
'''
#fname = os.path.join("..", "data", "pic_assignment_5.jpg")
#image = cv2.imread(fname)

def main():
    
    
    fname = os.path.join("data", "pic_assignment_5.jpg")
    image = cv2.imread(fname)
    
    '''
    Drawing rectangular box
    '''
    #Creating the rectangle, by looking at the image and trying to find coordinates that match.
    image = cv2.rectangle(image, (1280,850), (2890,2790), (0, 255,0), 3)

    cv2.imwrite("data/image_with_ROI.jpg", image) #saving image to notebook.



    '''
    Creating mask
    '''
    image = cv2.imread(fname) #Rereading image.

    #creating a mask containing only zeros (all black mask.)
    mask = np.zeros(image.shape[:2], dtype="uint8")



    #Now creating the rectangle inside the all black mask.
    cv2.rectangle(mask, (1280,850), (2890,2790), 255, -1)




    #Saving the masked image. Using bitwise_and to return the masked image.
    masked = cv2.bitwise_and(image, image, mask=mask)



    #Saving the cropped image.
    cv2.imwrite("data/image_masked.jpg", image)




    '''
    Cropping image

    '''

    image = cv2.imread(fname)

    #Importing Image from PIL to crop the image an easier way.
    im = Image.open(r"data/pic_assignment_5.jpg") #Rereading the image again.

    #Checking the height and width of the image:
    height = image.shape[0]
    width = image.shape[1]

    # Setting the points for cropped image. Found again by trying different coordinates on the in the pictures.
    left = width//4
    top = height//4
    right = (width//4)*3
    bottom = (height//8)*7
  
    # Cropped image of above dimension 
    # (It will not change orginal image) 
    im1 = im.crop((left, top, right, bottom)) 

    #Saving the cropped image, as it is in the wrong format, but if we reload it with the with cv2, it can be worked with again.
    im1.save("data/image_cropped.jpg")



    '''
    Edge detection
    
    '''

    #Reading the cropped file.
    crop_file = os.path.join("data/image_cropped.jpg")
    crop_image = cv2.imread(crop_file)


    #making the image greyscale
    grey_crop = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)

    #burring the picture to find borders
    blurred = cv2.GaussianBlur(grey_crop, (5,5), 0)
    canny = cv2.Canny(blurred, 30, 150)





    (contours, _) = cv2.findContours(canny.copy(),
                 cv2.RETR_EXTERNAL,#Filtering out inner structures
                 cv2.CHAIN_APPROX_SIMPLE) #Finding contours

    #saving Letters edge detection.
    image_contours = (cv2.drawContours(crop_image.copy(), #draw contours
            contours,                      #our list of contours.
            -1,                            #which contours to draw
            (0,255,0),                     #contours color
            2))                             # contour pixel width
           

    #Saving image
    cv2.imwrite("data/image_letters.jpg", image_contours)
    
    
if __name__=="__main__":
    main()