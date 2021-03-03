#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math 
from scipy.ndimage import interpolation as inter

def rotation_resize(img):
    if np.array(img).shape[0] < np.array(img).shape[1]:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    
    return cv2.resize(img,dsize=(400,600),interpolation=cv2.INTER_AREA)

def remove_shadow(img):
    img_resize = rotation_resize(img)
    rgb_planes = cv2.split(img_resize)

    result_planes = []
    result_norm_planes = []


    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)

        # plt.figure(figsize=[8,16])
        # plt.subplot(121)
        # plt.imshow(dilated_img)
        # plt.subplot(122)
        # plt.imshow(bg_img)

        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

        # plt.figure(figsize=[8,16])
        # plt.subplot(121)
        # plt.imshow(diff_img)
        # plt.subplot(122)
        # plt.imshow(norm_img)

        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)

        result = cv2.merge(result_planes)
        result_norm = cv2.merge(result_norm_planes)


    # plt.figure(figsize=[8,16])
    # plt.subplot(121)
    # plt.imshow(result)
    # plt.subplot(122)
    # plt.imshow(result_norm)

    # plt.figure(figsize=[10,10])
    # plt.imshow(result_norm)
    return result_norm


#%%

def compute_skew(file_name):
    
    #load in grayscale:
    src = cv2.imread(file_name,0)
    height, width = src.shape
    
    #invert the colors of our image:
    # cv2.bitwise_not(src, src)
    
    #Hough transform:
    minLineLength = width/2.0
    maxLineGap = 20
    lines = cv2.HoughLinesP(src,1,np.pi/180,200,minLineLength,maxLineGap)

    #calculate the angle between each line and the horizontal line:
    angle = 0.0
    nb_lines = len(lines)

    slope_sum = 0
    for line in lines:
        line = line[0]
        slope = (line[3]-line[1])/(line[2]-line[0])
        if slope == float('inf') or slope == float('-inf') or slope == float('nan'):
            continue
        else:
            slope_sum += slope
            
    return slope_sum/nb_lines
    

    
    # for line in lines:
    #     angle += math.atan2(line[0][3]*1.0 - line[0][1]*1.0,line[0][2]*1.0 - line[0][0]*1.0)
    
    # angle /= nb_lines*1.0
    
    return angle* 180.0 / np.pi
    # return angle


def deskew(file_name,angle):
    
    #load in grayscale:
    img = cv2.imread(file_name,0)
    wh = img.shape
    #invert the colors of our image:
    #cv2.bitwise_not(img, img)
    
    #compute the minimum bounding box:
    # non_zero_pixels = cv2.findNonZero(img)
    # print(len(non_zero_pixels))
    # center, wh, theta = cv2.minAreaRect(non_zero_pixels)
    
    center = (wh[1]//2,wh[0]//2)
    root_mat = cv2.getRotationMatrix2D(center, angle, 1)
    rows, cols = img.shape
    # rotated = cv2.warpAffine(img, root_mat, (cols, rows), flags=cv2.INTER_CUBIC)
    rotated = cv2.warpAffine(img, root_mat, (cols, rows), flags=cv2.INTER_CUBIC)
    return rotated
    #Border removing:
    sizex = np.int0(wh[0])
    sizey = np.int0(wh[1])
    # print(theta)
    # if theta > -45 :
    #     temp = sizex
    #     sizex= sizey
    #     sizey= temp
    # return cv2.getRectSubPix(rotated, (sizey,sizex), center)
#%%

img = cv2.imread('./img/test.jpg',0)
rotate_resized_img = rotation_resize(img)
cv2.imwrite('./result/rotate_resized.jpg',rotate_resized_img)

removed_shadow = remove_shadow(rotate_resized_img)
cv2.imwrite('./result/removed_shadow.jpg',removed_shadow)

file_path = './result/removed_shadow.jpg'
angle = compute_skew(file_path)
dst = deskew(file_path,angle)

cv2.imwrite('./result/skewed.jpg',dst)
#%%


file_path = 'removed_shadow.jpg'
angle = compute_skew(file_path)
dst = deskew(file_path,angle)


plt.figure(figsize=[15,30])
plt.subplot(121)
plt.imshow(dst)
plt.subplot(122)
plt.imshow(removed_shadow)

#%%
plt.figure(figsize=[15,15])
plt.imshow(removed_shadow-dst)

#%%
img = cv2.imread('test.jpg', 0)
removed_shadow = remove_shadow(img)

#%%
cv2.imwrite('removed_shadow.jpg',removed_shadow)


#%%




#%%
if __name__ == '__main__':
    img = cv2.imread('test.jpg', -1)
    removed_shadow = remove_shadow(img)


# https://www.sciencedirect.com/science/article/abs/pii/S0031320308002173