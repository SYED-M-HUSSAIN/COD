# img3 = img2.astype(np.uint8)       
# # find contours in the thresholded image
# cnts = cv2.findContours(img3.copy(), cv2.RETR_EXTERNAL,
#                         cv2.CHAIN_APPROX_SIMPLE)
# cnts = imutils.grab_contours(cnts)
# print("[INFO] {} unique contours found".format(len(cnts)))

import cv2
import numpy as np
import os
import imutils

# folder path of the segmented images
folder_path = r'/home/hussain/Downloads/COD_data_sets/COD10K-v3/Test/GT_Object'
check_path=r'/home/hussain/Downloads/COD_data_sets/COD10K-v3/Test/Labels'
# image_path = r"/home/hussain/Downloads/COD_data_sets/COD10K-v3/Test/Image"
i = 0
for filename in os.listdir(folder_path):
    # file_name=os.path.splitext(filename)[0] + '.txt'
    # possible_path=os.path.join(check_path,file_name)
    # if os.path.exists(possible_path):
        #print('here')
        # if filename not in os.listdir(image_path):
        #     print('image not found')
        #     continue
        # Construct the full file path
    file_path = os.path.join(folder_path, filename)

    # Load the segmented image
    segmented_image = cv2.imread(file_path)

    # Convert the segmented image to grayscale
    gray_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)

    # Threshold the grayscale image to create a binary mask
    _, binary_mask = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)

    # find contours in the thresholded image
    cnts = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    print("[INFO] {} unique contours found".format(len(cnts)))
    # print(filename)

    # Labels folder 
    # newFileName = str(int(filename[:5]) + 141) + '.jpg'
    # output_folder_path=r'/home/hussain/Downloads/MoCA-Mask-Pseudo/MoCA-Video-Train/jerboa/Labels'
    output_file = os.path.join(check_path, filename)
    output_file_path = os.path.splitext(output_file)[0]+ '.txt'
    i += 1
    with open(output_file_path, 'w') as output_file:
        for j in cnts:
            rect = cv2.boundingRect(j)
            x,y,w,h = rect
            # cv2.rectangle(segmented_image, (x,y),(x+w,y+h),(255,0,0),2) # to draw the bbox

            # Normalizing for yolo
            x_center = x + (w / 2)
            y_center = y + (h / 2)
            normalized_x = x_center / segmented_image.shape[1]
            normalized_y = y_center / segmented_image.shape[0]
            normalized_width = w / segmented_image.shape[1]
            normalized_height = h / segmented_image.shape[0]
            print(x_center," ",y_center,"  ",w,"  ",h)
            

            # with open(output_file_path, 'w') as output_file:
            output_file.write(f'{0} {normalized_x:.6f} {normalized_y:.6f} {normalized_width:.6f} {normalized_height:.6f}\n')

            # For displaying the images
            # segmented_image = cv2.resize(segmented_image, (480, 480))
            # cv2.imshow(str(i), segmented_image)
            # cv2.waitKey(0)
            # i += 1
print(i, 'labels made')
