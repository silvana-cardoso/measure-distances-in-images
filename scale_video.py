#!/usr/bin/env python

# version: organized, all video close when x is pressed, circle by two and three points

# script to calculate the distance between n number of points in a image

from email.policy import default
import numpy as np
import cv2 as cv
import copy
import matplotlib.pyplot as plt
from skimage import draw
import argparse


xy_transformation = np.empty((0,6),int)

xy = np.empty((2,0),int)
x_d,y_d = 50,50 #coord para os valores de distancia na imagem

distance_i = 0
coordinates = []
dict_distances = {}

font = cv.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_thickness = 2
begEnd = True
plt.ion()


# funtion to calculate the circle that passes through three points
def findCircle(x1,y1,x2,y2,x3,y3):

    x1y1_2 = x1**2+y1**2
    x2y2_2 = x2**2+y2**2
    x3y3_2 = x3**2+y3**2

    # A = [[x^2+y^2 x y 1]
    #     [x1^2+y1^2 x1 y1 1]
    #     [x2^2+y2^2 x2 y2 1]
    #     [x3^2+y3^2 x3 y3 1]]

    
    m12 = np.linalg.det(np.array([[x1y1_2, y1, 1],  #m12(A)  is a minor of A
                    [x2y2_2, y2, 1],                #the determinant of A without row 1 or column 2.
                    [x3y3_2, y3, 1]]))
    m11 = np.linalg.det(np.array([
        [x1, y1,1],
        [x2, y2, 1],
        [x3, y3, 1]
    ]))

    m13 = np.linalg.det(np.array([
        [x1y1_2, x1, 1],
        [x2y2_2, x2, 1],
        [x3y3_2, x3, 1]
    ]))

    m14 = np.linalg.det(np.array([
        [x1y1_2, x1, y1],
        [x2y2_2, x2, y2],
        [x3y3_2, x3, y3]
    ]))


    x0 = .5 * m12/m11

    y0 = -.5 * m13/m11

    r = round(np.sqrt(x0**2+y0**2+m14/m11))

    # return the center of the circle and the radius
    return x0,y0,r

# function to display the coordinates of the point clicked on the image
# and select three points of the edge of the circle
def click_event_three_points(event, x, y, flags, paramns):

    global xy_transformation

    #checking for left mouse clicks
    if event == cv.EVENT_LBUTTONDBLCLK:

        #display the coordinates on the shell
        print(f'{x},{y}')

        xy_transformation = np.append(xy_transformation,np.array([x,y]))

        # display the coordinates on the image window
        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(cimg, str(x)+','+str(y),(x,y), font,1,(255,0,0),2)
        cv.circle(cimg,(x,y),1,(255,0,0),2)

        cv.imshow('Three points on the circle',cimg)


def transformation_matrix(dX):
    rX = dX/2
    features_um_to_pixels_dict = {(0,0):(cx,cy),
                                    (rX,0):(cx+r,cy),
                                    (0,rX):(cx,cy-r),
                                    (rX,rX): (cx+r,cy-r)}

    A = np.zeros((2 * len(features_um_to_pixels_dict), 6), dtype=float)
    b = np.zeros((2 * len(features_um_to_pixels_dict), 1), dtype=float)
    index = 0

    for XY,xy in features_um_to_pixels_dict.items():
        X = XY[0]
        Y = XY[1]
        x = xy[0]
        y = xy[1]
        A[2*index,0] = x
        A[2*index,1] = y
        A[2*index,2] = 1
        A[2*index+1,3] = x
        A[2*index+1,4] = y
        A[2*index+1,5] = 1

        b[2*index,0] = X
        b[2*index+1,0] = Y

        index+=1

    #scaled image
    # A = 2*A

    x, residuals, rank, singular_values = np.linalg.lstsq(A, b, rcond=None)

    pixels_to_um_transformation_mtx = np.array([[x[0,0],x[1,0],x[2,0]],[x[3,0],x[4,0],x[5,0]],[0,0,1]])
    um_to_pixels_transformation_mtx = np.linalg.inv(pixels_to_um_transformation_mtx)

    return pixels_to_um_transformation_mtx, um_to_pixels_transformation_mtx


def click_event_distances(event, x, y, flags, paramns):

    global xy, cache, img, distance_i, y_d,x_d,dict_distances,coordinates, font, font_scale
    global font_thickness, height, begEnd,r_start, c_start

    
    
    #checking for left mouse clicks
    if event == cv.EVENT_LBUTTONDBLCLK:
        plt.clf()
        plt.close()

        #display the coordinates on the shell
        print(f'{x},{y}')
        xy = np.append(xy,np.array([[x],[y]]),axis=1)

        cv.circle(img,(x,y),2,(0,0,255),2) #bgr
        cv.imshow('Distance between points',img)

        if xy.shape[1]%2==0 and xy.size> 2:
            cv.line(img,(xy[:,-2]),(xy[:,-1]),(237,242,244),2)

            font = cv.FONT_HERSHEY_SIMPLEX
            coordinates_d_i = np.rint((xy[:,-2] +xy[:,-1])/2).astype('int32')-10

            coordinates.append(coordinates_d_i)

            cv.putText(img,f'd{distance_i}',coordinates_d_i,font,0.7,(237,242,244),2)

            xy_h = np.append(xy[:,-2:], np.array([2*[1]]), axis=0)

            XY = np.empty((3,0))
            for i in range(2):
                XY = np.append(XY,pixels_to_um_transformation_mtx @ xy_h[:,[i]],axis=1)

            diff_points = np.diff(XY) #diferença entre o par de pontos

            distance = round(np.linalg.norm(diff_points),3)

            dict_distances[f'd{distance_i}'] =  distance

            text = f'd{distance_i} = {distance/1000} mm'

            cv.putText(img,text,(x_d,y_d),font,font_scale,((237,242,244)),font_thickness,cv.LINE_4)
            
            distance_i+=1

            (width,height), baseline = cv.getTextSize(text,font,font_scale,font_thickness)
            y_d += (height+10)

            cv.imshow('Distance between points',img)
        
        if (flags & cv.EVENT_FLAG_SHIFTKEY) and begEnd:

            r_start = x
            c_start = y

            # begEnd = False
            begEnd = not begEnd

        elif (flags & cv.EVENT_FLAG_SHIFTKEY) and (not begEnd):
            # begEnd = True
            begEnd = not begEnd
            r_end = x
            c_end = y

            gray = cv.cvtColor(cache.copy(), cv.COLOR_BGR2GRAY)
            line = np.transpose(np.array(draw.line(r_start,c_start,r_end,c_end))) # get the indices of pixels that belong to the line (array n x 2)
            data = gray.copy()[line[:,1],line[:,0]] #gray[y,x]

            x_line = copy.deepcopy(line[:,0])
            y_line = copy.deepcopy(line[:,1])
            
            fig, ax = plt.subplots(figsize=(8,6))

            ax.plot(x_line,data)
            ax.set_title(f'Intensity profile d{distance_i-1} = {dict_distances[f"d{distance_i-1}"] /1000} mm')
            ax.set_xlabel('x coordinate of the pixel')
            ax.set_ylabel('Intensity')


    #remove points
    
    if event == cv.EVENT_LBUTTONDOWN and (flags & cv.EVENT_FLAG_CTRLKEY):
        plt.clf()
        plt.close()
        # if begEnd = True, begEnd = False
        # if begEnd = False, begEnd = True
        begEnd = not begEnd

        img = cache.copy()
        j=0
        x_d,y_d = 50,50

        try:
            xy = np.delete(xy,-1,1)

            n_points = xy.shape[1]

            if n_points%2==1:
                try:
                    dict_distances.popitem()
                    coordinates.pop(-1)
                    distance_i-=1
                except:
                    dict_distances ={}

            if n_points>=1:
                
                for i in range(n_points):
                    cv.circle(img,(xy[0:2,i]),2,(0,0,255),2)
                    
                    if n_points>=2 and i%2==1:
                        cv.line(img,(xy[:,i-1]),(xy[:,i]),(237,242,244),2)

                        text = f'd{j} = {dict_distances[f"d{j}"]/1000} mm'

                        cv.putText(img,text,(x_d,y_d),font,font_scale,((237,242,244)),font_thickness,cv.LINE_4)
                        
                        cv.putText(img,f'd{j}',(coordinates[j][0],coordinates[j][1]),font,0.7,(237,242,244),2)

                        j+=1
                        (width,height), baseline = cv.getTextSize(text,font,font_scale,font_thickness)
                        y_d += (height+10)

                cv.imshow('Distance between points',img)
            else:
                cv.imshow('Distance between points',img)

        except:
            distance_i=0
            print('That is no point to remove')


if __name__ == '__main__': 

    global scale
    scale = 100

    parser = argparse.ArgumentParser(description = 'Calculate the distance between n number of points in a image.')
    parser.add_argument('--p', help='Redefine the three points used in the transformation. They form a circle.',
                        action = 'store_true')
    parser.add_argument('--i',help='Path to the new image used in the transformation.',
                        default = 'default_transformation_image.png')
    parser.add_argument('--n',help='Capture a new frame to be used in the transformation',
                        action='store_true')
    parser.add_argument('--d', help='Diameter of the circle passing through the three points defined in the image.',
                        type = float, default = None)
    args = parser.parse_args()

#########################################################################################
#####################PART 1: CONSTRUCT THE TRANSFORMATION MATRIX ########################
############################## PIXELS(x,y) -> REAL(X,Y) #################################
################### FIND THREE KNOWN POINTS ON THE EDGE OF A CIRCLE #####################
#########################################################################################
    if args.p:

        

        # if the three points will be define by a new image and not a frame
        if args.i == 'default_transformation_image.png' and args.d == None:
            args.d = 3134.5
        
        # verify if the new distance known was passed
        # if not, raise a error
        if (args.i != 'default_transformation_image.png' or args.n) and args.d == None:
            raise ValueError('The argument --d(diameter of the circle) must be passed if --i or --n is passed.')
        
        # if the three points wil be define by a frame insted of a image
        if args.n:

            cap_ft = cv.VideoCapture(0)

            while(cap_ft.isOpened()):
                _,frame = cap_ft.read()

                cv.namedWindow('Capture new img')
                dim = (int(frame.shape[1]*scale/100),int(frame.shape[0]*scale/100))
                frame = cv.resize(frame,dim)
                cv.imshow('Capture new img',frame)

                key_cap = cv.waitKey(1)

                # capture the frame if "enter" is pressed
                if key_cap == 13:
                    cimg = copy.deepcopy(frame)
                    print('New image captured')
                    cap_ft.release()
                    cv.destroyWindow('Capture new img')
                    break
                
                # close window and the loop when the "esc" key is pressed
                # close window and the loop when the "X" button is pressed

                elif key_cap == 27 or not cv.getWindowProperty('Capture new img', cv.WND_PROP_VISIBLE):
                    print("Operation Cancelled")
                    cimg = cv.imread(args.i)
                    dim = (int(cimg.shape[1]*scale/100),int(cimg.shape[0]*scale/100))
                    cimg = cv.resize(cimg,dim)
                    cv.destroyAllWindows()
                    break
                
           
        else:
            #reading the image
            cimg = cv.imread(args.i)
            dim = (int(cimg.shape[1]*scale/100),int(cimg.shape[0]*scale/100))
            cimg = cv.resize(cimg,dim)
        print(args.n)   
        # display the image
        cv.imshow('Three points on the circle',cimg)

        # setting mouse handler for the image and calling the click_event() function
        cv.setMouseCallback('Three points on the circle', click_event_three_points)

        while True:
            # wait for a key to be pressed
            k = cv.waitKey(1)

            # if enter was pressed
            if k==13:
                if xy_transformation.size<6:
                    midpoint = np.rint(np.mean(xy_transformation.reshape([2,2]),axis=0)).astype('int32')
                    cx, cy = midpoint
                    r = round(np.linalg.norm(xy_transformation[0:2]-midpoint))
                else:
                    cx, cy, r = findCircle(*xy_transformation[:6])

                pixels_to_um_transformation_mtx, um_to_pixels_transformation_mtx = transformation_matrix(args.d)

                # Changing the dtype  to int
                circle = np.uint16(np.around([cx,cy,r]))
                cimg = cimg.copy()

                # draw the outer circle
                cv.circle(cimg,(circle[0],circle[1]),circle[2],(0,255,0),2)
                # draw the center of the circle
                cv.circle(cimg,(circle[0],circle[1]),2,(255,0,0),2)

                cv.imshow('Three points on the circle',cimg)

            # if esc was pressed
            # or close window and the loop when the "X" button is pressed
            elif k == 27 or not cv.getWindowProperty('Three points on the circle', cv.WND_PROP_VISIBLE):
                print("Operation Cancelled")
                cv.destroyWindow('Three points on the circle')
                break

    else:
        cx = 674.4903354178515
        cy = 479.67364269471307
        r = 457

        pixels_to_um_transformation_mtx = np.array([[ 3.42943107e+00,  9.86640072e-12, -2.31311811e+03],
            [ 1.28866516e-12, -3.42943107e+00,  1.64500769e+03],
            [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
        um_to_pixels_transformation_mtx = np.array([[ 2.91593556e-01,  8.38908497e-13,  6.74490335e+02],
            [ 1.09571077e-13, -2.91593556e-01,  4.79673643e+02],
            [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])


    print(f'(pixels)cx = {round(cx)},cy = {round(cy)},r = {round(r)}')
    print(f'Transformation matrix (pixels to micrometers):\n{pixels_to_um_transformation_mtx}')


########################################################################################
############### PART 2: CALCULATE THE DISTANCE BETWEEN N POINTS ########################
############################# IN MICROMETERS ###########################################
########################################################################################
    cap = cv.VideoCapture(0)

    while (cap.isOpened()):
        _, frame = cap.read()

        cv.namedWindow('live')
        dim = (int(frame.shape[1]*scale/100),int(frame.shape[0]*scale/100))
        frame = cv.resize(frame,dim)
        cv.imshow('live',frame)

        key_cap = cv.waitKey(50)

        if key_cap == 13:
            img = copy.deepcopy(frame)
            print('Frame captured')
            cache = img.copy()
            # display the Distance between points
            cv.imshow('Distance between points',img)

            # setting mouse handler for the Distance between points and calling the click_event() function
            cv.setMouseCallback('Distance between points', click_event_distances)

            while True:
                k = cv.waitKey(1)
                # close window and the loop when the "X" button is pressed
                # or close window and the loop when the "esc" key is pressed
                if k ==27 or not cv.getWindowProperty('Distance between points', cv.WND_PROP_VISIBLE):
                    print("Operation Cancelled")
                    break

        # close window and the loop when the "X" button is pressed
        # close window and the loop when the "esc" key is pressed
        if key_cap ==27 or not cv.getWindowProperty('live', cv.WND_PROP_VISIBLE):
            print("End process")
            break

    cap.release()
    cv.destroyAllWindows()