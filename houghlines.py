import cv2
import numpy as np
from itertools import combinations
from scipy.spatial.transform import Rotation


def unique(sorted_hough_space_unfiltered):
    unique=dict()
    rho=[]
    theta=[]
    val=0
    prev_rho=0
    prev_theta=0
    thresh_thet=10
    thresh_rho=50
    for key, value in sorted_hough_space_unfiltered.items():
        if val==0:
            prev_rho=key[0]
            prev_theta=key[1]
            rho.append(key[0])
            theta.append(key[1])
            val=value
        else:
            if prev_rho-thresh_rho <= key[0] <= prev_rho+thresh_rho and prev_theta-thresh_thet <= key[1] <= prev_theta+thresh_thet:
                val+=value
                prev_rho=key[0]
                prev_theta=key[1]
                rho.append(key[0])
                theta.append(key[1])
            else:
                unique[(int(np.mean(rho)),int(np.mean(theta)))]=val
                rho=[]
                theta=[]
                val=0
    if rho and val:
        unique[(int(np.mean(rho)),int(np.mean(theta)))]=val
    return unique
    
    
def hough(img):
#     Create a parameter space
#     Here we use a dictionary
    H=dict()
#     We check for pixels in image which have value more than 0(not black)
    co=np.where(img>0)
    co=np.array(co).T
    for point in co:
        for t in range(180):
#             Compute rho for theta 0-180
            d=point[0]*np.sin(np.deg2rad(t))+point[1]*np.cos(np.deg2rad(t))
            d=int(d)
#         Compare with the extreme cases for image
            if d<int(np.ceil(np.sqrt(np.square(img.shape[0]) + np.square(img.shape[1])))):
                if (d,t) in H:
#                 Upvote
                    H[(d,t)] += 1
                else:
#             Create a new vote
                    H[(d,t)] = 1
    return H

def intersection_point(rho1, theta1, rho2, theta2):
    A = np.array([[np.cos(theta1), np.sin(theta1)], [np.cos(theta2), np.sin(theta2)]])
    b = np.array([rho1, rho2])
    x, y = np.linalg.solve(A, b)
    return [int(x), int(y)]

def homography(camera_corners,world_corners):
    
    A=[]
    for i in range(camera_corners.shape[0]):
        x,y=camera_corners[i,0],camera_corners[i,1]
        xw,yw=world_corners[i,0],world_corners[i,1]
        A.append([x,y,1,0,0,0,-xw*x,-xw*y,-xw])
        A.append([0,0,0,x,y,1,-yw*x,-yw*y,-yw])
    A=np.array(A)
    eigenvalues, eigenvectors = np.linalg.eig(A.T@A)
    min_eig_idx=np.argmin(eigenvalues)
    smallest_eigen_vector=eigenvectors[:,min_eig_idx]
    H=np.reshape(smallest_eigen_vector,(3,3))
    H=H/H[2,2]
    return H


img = cv2.imread("book.jpeg")

scale_percent = 30 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
# resize image
img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Apply gaussian blur to he mask
blur = cv2.GaussianBlur(hsv, (9, 9), 3)

# Define the color range for the ball (in HSV format)
lower_color = np.array([0, 0, 24],np.uint8)
upper_color = np.array([179, 255, 73],np.uint8)
# Define the kernel size for the morphological operations
kernel_size = 7
# Create a mask for the ball color using cv2.inRange()
mask = cv2.inRange(blur, lower_color, upper_color)


# Apply morphological operations to the mask to fill in gaps
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
mask = cv2.dilate(mask, kernel,iterations=1)


#  Use canny edges to get the edges of the image mask
edges = cv2.Canny(mask,200, 240, apertureSize=3)

#  Get the hough space, sort and select to 10 values
hough_space = dict(sorted(hough(edges).items(), key=lambda item: item[1],reverse=True)[:500])

#  Get the hough space, sort and select to 10 values
hough_space = dict(sorted(hough(edges).items(), key=lambda item: item[1],reverse=True)[:20])

#     Sort the hough space w.r.t rho and theta
sorted_hough_space_unfiltered = dict(sorted(hough_space.items()))

#     Get the unique rhoand theta values
unique_=unique(sorted_hough_space_unfiltered)

#     Sort according to value and get the top 4 lines
unique_=dict(sorted(unique_.items(), key=lambda item: item[1],reverse=True)[:4])
for line in unique_:
    rho, theta = line
    theta=np.deg2rad(theta)
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)


#     Create combinations of lines
line_combinations = list(combinations(unique_.items(), 2))

intersection=[]
filter_int=[]
for (key1, value1), (key2, value2) in line_combinations:
    try:
#             Solve point of intersection of two lines
        intersection.append(intersection_point(key1[0],np.deg2rad(key1[1]), key2[0],np.deg2rad(key2[1]))) 
    except:
        print("Singular Matrix")

for x,y in intersection:
    if x>0 and y>0:
#             Get the valid cartesan co ordinates
        cv2.circle(img, (x, y), 5, (0, 0, 0), -1)
        cv2.putText(img, '{},{}'.format(x,y), (x-10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        filter_int.append([x,y])

cv2.imshow('Frame', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


