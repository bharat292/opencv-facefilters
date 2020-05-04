from flask import Flask, render_template, Response,request
import cv2
import numpy as np
import dlib
import math
from math import hypot
import sys
from time import time
import random

app = Flask(__name__)

camera = cv2.VideoCapture(0)
def main_frames(name):
    if name == 'dog':
        nose_image = cv2.imread("dog_nose.png")
        ears_image = cv2.imread("dog_ears.png")
        _, frame1 = camera.read()
        rows, cols, _ = frame1.shape
        ears_mask = np.zeros((rows, cols), np.uint8)
        nose_mask = np.zeros((rows, cols), np.uint8)
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        while True:
            _,frame1 = camera.read()
            try:
                ears_mask.fill(0)
                nose_mask.fill(0)
                gray_frame1 = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
                faces= detector(gray_frame1)
                for face in faces:
                    landmarks = predictor(gray_frame1, face)
                    left_forehead = (landmarks.part(19).x,landmarks.part(19).y)
                    right_forehead = (landmarks.part(26).x,landmarks.part(26).y)
                    center_forehead = (landmarks.part(28).x, landmarks.part(28).y)
                    forehead_width = int(hypot(left_forehead[0]-right_forehead[0],left_forehead[1]-right_forehead[1])*2.6)
                    forehead_height = int(forehead_width*0.77)
                    top_left_ears = (int(center_forehead[0]-forehead_width/2),int(center_forehead[1]-forehead_height))
                    bottom_right_ears = (int(center_forehead[0]+forehead_width/2),int(center_forehead[1]+forehead_height))
                    dog_ears = cv2.resize(ears_image,(forehead_width,forehead_height))
                    dog_ears_gray = cv2.cvtColor(dog_ears,cv2.COLOR_BGR2GRAY)
                    _, ears_mask = cv2.threshold(dog_ears_gray,25,255,cv2.THRESH_BINARY_INV)
                    ears_area = frame1[top_left_ears[1]:top_left_ears[1]+forehead_height,
                                      top_left_ears[0]:top_left_ears[0]+forehead_width]
                    top_nose = (landmarks.part(29).x, landmarks.part(29).y)
                    center_nose = (landmarks.part(30).x, landmarks.part(30).y)
                    left_nose = (landmarks.part(31).x, landmarks.part(31).y)
                    right_nose = (landmarks.part(35).x, landmarks.part(35).y)
                    nose_width = int(hypot(left_nose[0] - right_nose[0],
                                       left_nose[1] - right_nose[1]) * 1.7)
                    nose_height = int(nose_width * 0.77)
                    top_left_nose = (int(center_nose[0] - nose_width / 2),
                                          int(center_nose[1] - nose_height / 2))
                    bottom_right_nose = (int(center_nose[0] + nose_width / 2),
                                   int(center_nose[1] + nose_height / 2))
                    nose_pig = cv2.resize(nose_image, (nose_width, nose_height))
                    nose_pig_gray = cv2.cvtColor(nose_pig, cv2.COLOR_BGR2GRAY)
                    _, nose_mask = cv2.threshold(nose_pig_gray, 25, 255, cv2.THRESH_BINARY_INV)
                    nose_area = frame1[top_left_nose[1]: top_left_nose[1] + nose_height,
                                top_left_nose[0]: top_left_nose[0] + nose_width]
                    ears_area_no_ears = cv2.bitwise_and(ears_area, ears_area, mask=ears_mask)
                    nose_area_no_nose = cv2.bitwise_and(nose_area, nose_area, mask=nose_mask)
                    frame1 = frame1  
                    final_ears = cv2.add(ears_area_no_ears,dog_ears)
                    frame1[top_left_ears[1]: top_left_ears[1]+forehead_height,
                         top_left_ears[0]:top_left_ears[0]+forehead_width] = final_ears
                    final_nose = cv2.add(nose_area_no_nose, nose_pig)
                    frame1[top_left_nose[1]: top_left_nose[1] + nose_height,
                        top_left_nose[0]: top_left_nose[0] + nose_width] = final_nose
            except: 
                _,frame2 = camera.read()
                ret, buffer = cv2.imencode('.jpg', frame2)
                frame2 = buffer.tobytes()
                yield (b'--frame2\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame2 + b'\r\n')
            else:    
                ret, buffer = cv2.imencode('.jpg', frame1)
                frame1 = buffer.tobytes()
                yield (b'--frame1\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame1 + b'\r\n')  
    if name=='pig':
        nose_image = cv2.imread("pig_nose.png")
        _,frame3 = camera.read()
        rows, cols, _ = frame3.shape
        nose_mask = np.zeros((rows, cols), np.uint8)
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        while True:
            success, frame3 = camera.read()
            try:
                nose_mask.fill(0)
                gray_frame3 = cv2.cvtColor(frame3, cv2.COLOR_BGR2GRAY)
                faces = detector(gray_frame3)
                for face in faces:
                    landmarks = predictor(gray_frame3, face)
                    top_nose = (landmarks.part(29).x, landmarks.part(29).y)
                    center_nose = (landmarks.part(30).x, landmarks.part(30).y)
                    left_nose = (landmarks.part(31).x, landmarks.part(31).y)
                    right_nose = (landmarks.part(35).x, landmarks.part(35).y)
                    nose_width = int(hypot(left_nose[0] - right_nose[0],
                                       left_nose[1] - right_nose[1]) * 1.7)
                    nose_height = int(nose_width * 0.77)
                    top_left = (int(center_nose[0] - nose_width / 2),
                                          int(center_nose[1] - nose_height / 2))
                    bottom_right = (int(center_nose[0] + nose_width / 2),
                                   int(center_nose[1] + nose_height / 2))
                    nose_pig = cv2.resize(nose_image, (nose_width, nose_height))
                    nose_pig_gray = cv2.cvtColor(nose_pig, cv2.COLOR_BGR2GRAY)
                    _, nose_mask = cv2.threshold(nose_pig_gray, 25, 255, cv2.THRESH_BINARY_INV)
                    nose_area = frame3[top_left[1]: int(top_left[1] + nose_height),
                                top_left[0]: int(top_left[0] + nose_width)]
                    nose_area_no_nose = cv2.bitwise_and(nose_area, nose_area, mask=nose_mask)
                    final_nose = cv2.add(nose_area_no_nose, nose_pig)
                    frame3[top_left[1]: top_left[1] + nose_height,
                    top_left[0]: top_left[0] + nose_width] = final_nose
            except: 
                _,frame4 = camera.read()
                ret, buffer = cv2.imencode('.jpg', frame4)
                frame4 = buffer.tobytes()
                yield (b'--frame4\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame4 + b'\r\n')
            else:    
                ret, buffer = cv2.imencode('.jpg', frame3)
                frame3 = buffer.tobytes()
                yield (b'--frame3\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame3 + b'\r\n')

    if name=='panda':
        panda_image = cv2.imread("panda_face.png")
        _, frame5 = camera.read()
        rows, cols, _ = frame5.shape
        panda_mask = np.zeros((rows, cols), np.uint8)
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        while True:
            _,frame5 = camera.read()
            try:
                panda_mask.fill(0)
                gray_frame5 = cv2.cvtColor(frame5,cv2.COLOR_BGR2GRAY)
                
                faces= detector(gray_frame5)
                for face in faces:
                    #logic behind the filter
                    landmarks = predictor(gray_frame5, face)
                    left_pandaface = (landmarks.part(1).x,landmarks.part(1).y)
                    right_pandaface = (landmarks.part(17).x,landmarks.part(17).y)
                    center_pandaface = (landmarks.part(28).x, landmarks.part(28).y)
                    pandaface_width = int(hypot(left_pandaface[0]-right_pandaface[0],left_pandaface[1]-right_pandaface[1])*7.0)
                    pandaface_height = int(pandaface_width*0.81)
                    top_left = (int(center_pandaface[0]-pandaface_width/2),int(center_pandaface[1]-pandaface_height/2))
                    bottom_right = (int(center_pandaface[0]+pandaface_width/2),int(center_pandaface[1]+pandaface_height/2))
                    panda_face = cv2.resize(panda_image,(pandaface_width,pandaface_height))
                    panda_face_gray = cv2.cvtColor(panda_face,cv2.COLOR_BGR2GRAY)
                    _, pandaface_mask = cv2.threshold(panda_face_gray,25,255,cv2.THRESH_BINARY_INV)
                    pandaface_area = frame5[top_left[1]:top_left[1]+pandaface_height,
                                      top_left[0]:top_left[0]+pandaface_width]
                    pandaface_area_no_face = cv2.bitwise_and(pandaface_area, pandaface_area, mask=pandaface_mask)
                    frame5 = frame5   
                    final_pandaface = cv2.add(pandaface_area_no_face,panda_face)
                    frame5[top_left[1]: top_left[1]+pandaface_height,
                         top_left[0]:top_left[0]+pandaface_width] = final_pandaface  
            except: 
                _,frame6 = camera.read()
                ret, buffer = cv2.imencode('.jpg', frame6)
                frame6 = buffer.tobytes()
                yield (b'--frame6\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame6 + b'\r\n')
            else:    
                ret, buffer = cv2.imencode('.jpg', frame5)
                frame5 = buffer.tobytes()
                yield (b'--frame5\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame5 + b'\r\n')

    if name=='snake': 
        #initializing font for puttext
        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        #loading apple image and making its mask to overlay on the camera feed
        apple = cv2.imread("mouse.png",-1)
        apple_mask = apple[:,:,3]
        apple_mask_inv = cv2.bitwise_not(apple_mask)
        apple = apple[:,:,0:3]
        # resizing apple images
        apple = cv2.resize(apple,(40,40),interpolation=cv2.INTER_AREA)
        apple_mask = cv2.resize(apple_mask,(40,40),interpolation=cv2.INTER_AREA)
        apple_mask_inv = cv2.resize(apple_mask_inv,(40,40),interpolation=cv2.INTER_AREA)
        #initilizing a black blank image
        blank_img = np.zeros((480,640,3),np.uint8)
        #kernels for morphological operations
        kernel_erode = np.ones((4,4),np.uint8)
        kernel_close = np.ones((15,15),np.uint8)
        #for blue [99,115,150] [110,255,255]
        #function for detecting red color
        def detect_red(hsv):
            #lower bound for red color hue saturation value
            lower = np.array([136, 87, 111])  # 136,87,111
            upper = np.array([179, 255, 255])  # 180,255,255
            mask1 = cv2.inRange(hsv, lower, upper)
            lower = np.array([0, 110, 100])
            upper = np.array([3, 255, 255])
            mask2 = cv2.inRange(hsv, lower, upper)
            maskred = mask1 + mask2
            maskred = cv2.erode(maskred, kernel_erode, iterations=1)
            maskred = cv2.morphologyEx(maskred,cv2.MORPH_CLOSE,kernel_close)
            return maskred

        #functions for detecting intersection of line segments.
        def orientation(p,q,r):
            val = int(((q[1] - p[1]) * (r[0] - q[0])) - ((q[0] - p[0]) * (r[1] - q[1])))
            if val == 0:
                #linear
                return 0
            elif (val>0):
                #clockwise
                return 1
            else:
                #anti-clockwise
                return 2

        def intersect(p,q,r,s):
            o1 = orientation(p, q, r)
            o2 = orientation(p, q, s)
            o3 = orientation(r, s, p)
            o4 = orientation(r, s, q)
            if(o1 != o2 and o3 != o4):
                return True

            return False

        #initilizing time (used for increasing the length of snake per second)
        start_time = int(time())
        # q used for intialization of points
        q,snake_len,score,temp=0,200,0,1
        # stores the center point of the red blob
        point_x,point_y = 0,0
        # stores the points which satisfy the condition, dist stores dist between 2 consecutive pts, length is len of snake
        last_point_x,last_point_y,dist,length = 0,0,0,0
        # stores all the points of the snake body
        points = []
        # stores the length between all the points
        list_len = []
        # generating random number for placement of apple image
        random_x = random.randint(10,550)
        random_y = random.randint(10,400)
        #used for checking intersections
        a,b,c,d = [],[],[],[]
        #main loop
        while 1:
            xr, yr, wr, hr = 0, 0, 0, 0
            _,frame7 = camera.read()
            #fliping the frame7 horizontally.
            #frame7 = cv2.flip(frame7,1)
            # initilizing the accepted points so that they are not at the top left corner
            if(q==0 and point_x!=0 and point_y!=0):
                last_point_x = point_x
                last_point_y = point_y
                q=1
            #converting to hsv
            hsv = cv2.cvtColor(frame7,cv2.COLOR_BGR2HSV)
            maskred = detect_red(hsv)
            #finding contours
            contour_red, _ = cv2.findContours(maskred,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            #drawing rectangle around the accepted blob
            try:
                for i in range (0,10):
                    xr, yr, wr, hr = cv2.boundingRect(contour_red[i])
                    if (wr*hr)>2000:
                        break
            except:
                pass
            cv2.rectangle(frame7, (xr, yr), (xr + wr, yr + hr), (0, 0, 255), 2)
            #making snake body
            point_x = int(xr+(wr/2))
            point_y = int(yr+(hr/2))
            # finding distance between the last point and the current point
            dist = int(math.sqrt(pow((last_point_x - point_x), 2) + pow((last_point_y - point_y), 2)))
            if (point_x!=0 and point_y!=0 and dist>5):
                #if the point is accepted it is added to points list and its length added to list_len
                list_len.append(dist)
                length += dist
                last_point_x = point_x
                last_point_y = point_y
                points.append([point_x, point_y])
            #if length becomes greater then the expected length, removing points from the back to decrease length
            if (length>=snake_len):
                for i in range(len(list_len)):
                    length -= list_len[0]
                    list_len.pop(0)
                    points.pop(0)
                    if(length<=snake_len):
                        break
            #initializing blank black image
            blank_img = np.zeros((480, 640, 3), np.uint8)
            #drawing the lines between all the points
            for i,j in enumerate(points):
                if (i==0):
                    continue
                cv2.line(blank_img, (points[i-1][0], points[i-1][1]), (j[0], j[1]), (0, 0, 255), 5)
            cv2.circle(blank_img, (last_point_x, last_point_y), 5 , (10, 200, 150), -1)
            #if snake eats apple increase score and find new position for apple
            if  (last_point_x>random_x and last_point_x<(random_x+40) and last_point_y>random_y and last_point_y<(random_y+40)):
                score +=1
                random_x = random.randint(10, 550)
                random_y = random.randint(10, 400)
            #adding blank image to captured frame7
            frame7 = cv2.add(frame7,blank_img)
            #adding apple image to frame7
            roi = frame7[random_y:random_y+40, random_x:random_x+40]
            img_bg = cv2.bitwise_and(roi, roi, mask=apple_mask_inv)
            img_fg = cv2.bitwise_and(apple, apple, mask=apple_mask)
            dst = cv2.add(img_bg, img_fg)
            frame7[random_y:random_y + 40, random_x:random_x + 40] = dst
            cv2.putText(frame7, str("Score - "+str(score)), (250, 450), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
            # checking for snake hitting itself
            if(len(points)>5):
                # a and b are the head points of snake and c,d are all other points
                b = points[len(points)-2]
                a = points[len(points)-1]
                for i in range(len(points)-3):
                    c = points[i]
                    d = points[i+1]
                    if(intersect(a,b,c,d) and len(c)!=0 and len(d)!=0):
                        temp = 0
                        break
                if temp==0:
           
                    start_time = int(time())
                    q,snake_len,score,temp=0,200,0,1
                    point_x,point_y = 0,0
                    last_point_x,last_point_y,dist,length = 0,0,0,0
                    points = []
                    list_len = []
                    temp=1

            ret, buffer = cv2.imencode('.jpg', frame7)
            frame7 = buffer.tobytes()
            yield (b'--frame7\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame7 + b'\r\n')

            # increasing the length of snake 40px per second
            if((int(time())-start_time)>1):
                snake_len += 40
                start_time = int(time())

    if name=='fish':
        #initializing font for puttext
        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        #loading apple image and making its mask to overlay on the camera feed
        apple = cv2.imread("prawn.png",-1)
        apple_mask = apple[:,:,3]
        apple_mask_inv = cv2.bitwise_not(apple_mask)
        apple = apple[:,:,0:3]
        # resizing apple images
        apple = cv2.resize(apple,(40,40),interpolation=cv2.INTER_AREA)
        apple_mask = cv2.resize(apple_mask,(40,40),interpolation=cv2.INTER_AREA)
        apple_mask_inv = cv2.resize(apple_mask_inv,(40,40),interpolation=cv2.INTER_AREA)

        fish = cv2.imread('fish.png')
        fish = cv2.resize(fish,(100,100))
        fish_gray = cv2.cvtColor(fish,cv2.COLOR_BGR2GRAY)
        _, fish_mask = cv2.threshold(fish_gray,25,255,cv2.THRESH_BINARY_INV)
        #initilizing a black blank image
        blank_img = np.zeros((480,640,3),np.uint8)
        #kernels for morphological operations
        kernel_erode = np.ones((4,4),np.uint8)
        kernel_close = np.ones((15,15),np.uint8)
        #for blue [99,115,150] [110,255,255]
        #function for detecting red color
        def detect_red(hsv):
            #lower bound for red color hue saturation value
            lower = np.array([136, 87, 111])  # 136,87,111
            upper = np.array([179, 255, 255])  # 180,255,255
            mask1 = cv2.inRange(hsv, lower, upper)
            lower = np.array([0, 110, 100])
            upper = np.array([3, 255, 255])
            mask2 = cv2.inRange(hsv, lower, upper)
            maskred = mask1 + mask2
            maskred = cv2.erode(maskred, kernel_erode, iterations=1)
            maskred = cv2.morphologyEx(maskred,cv2.MORPH_CLOSE,kernel_close)
            return maskred

        #functions for detecting intersection of line segments.
        def orientation(p,q,r):
            val = int(((q[1] - p[1]) * (r[0] - q[0])) - ((q[0] - p[0]) * (r[1] - q[1])))
            if val == 0:
                #linear
                return 0
            elif (val>0):
                #clockwise
                return 1
            else:
                #anti-clockwise
                return 2

        def intersect(p,q,r,s):
            o1 = orientation(p, q, r)
            o2 = orientation(p, q, s)
            o3 = orientation(r, s, p)
            o4 = orientation(r, s, q)
            if(o1 != o2 and o3 != o4):
                return True

            return False

        #initilizing time (used for increasing the length of snake per second)
        start_time = int(time())
        # q used for intialization of points
        q,snake_len,score,temp=0,200,0,1
        # stores the center point of the red blob
        point_x,point_y = 0,0
        # stores the points which satisfy the condition, dist stores dist between 2 consecutive pts, length is len of snake
        last_point_x,last_point_y,dist,length = 0,0,0,0
        # stores all the points of the snake body
        points = []
        # stores the length between all the points
        list_len = []
        # generating random number for placement of apple image
        random_x = random.randint(10,550)
        random_y = random.randint(10,400)
        #used for checking intersections
        a,b,c,d = [],[],[],[]
        #main loop
        while 1:
            xr, yr, wr, hr = 0, 0, 0, 0
            _,frame8 = camera.read()
            #fliping the frame8 horizontally.
            #frame8 = cv2.flip(frame8,1)
            # initilizing the accepted points so that they are not at the top left corner
            if(q==0 and point_x!=0 and point_y!=0):
                last_point_x = point_x
                last_point_y = point_y
                q=1
            #converting to hsv
            hsv = cv2.cvtColor(frame8,cv2.COLOR_BGR2HSV)
            maskred = detect_red(hsv)
            #finding contours
            contour_red, _ = cv2.findContours(maskred,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            #drawing rectangle around the accepted blob
            try:
                for i in range (0,10):
                    xr, yr, wr, hr = cv2.boundingRect(contour_red[i])
                    if (wr*hr)>2000:
                        break
            except:
                pass
            #cv2.rectangle(frame8, (xr, yr), (xr + wr, yr + hr), (0, 0, 255), 2)
            #making snake body
            point_x = int(xr+(wr/2))
            point_y = int(yr+(hr/2))
            # finding distance between the last point and the current point
            dist = int(math.sqrt(pow((last_point_x - point_x), 2) + pow((last_point_y - point_y), 2)))
            if (point_x!=0 and point_y!=0 and dist>5):
                #if the point is accepted it is added to points list and its length added to list_len
                list_len.append(dist)
                length += dist
                last_point_x = point_x
                last_point_y = point_y
                points.append([point_x, point_y])
            #if length becomes greater then the expected length, removing points from the back to decrease length
            if (length>=snake_len):
                for i in range(len(list_len)):
                    length -= list_len[0]
                    list_len.pop(0)
                    points.pop(0)
                    if(length<=snake_len):
                        break
            #initializing blank black image
            blank_img = np.zeros((480, 640, 3), np.uint8)
            #drawing the lines between all the points
            for i,j in enumerate(points):
                if (i==0):
                    continue
                cv2.line(frame8, (points[i-1][0], points[i-1][1]), (j[0], j[1]), (255, 0,0), 5)
                try:
                    fish_area = blank_img[yr-50:yr+50,xr-50:xr+50]
                    fish_area_no_fish = cv2.bitwise_and(fish_area, fish_area, mask=fish_mask)   
                    final_fish = cv2.add(fish_area_no_fish,fish)
                    blank_img[yr-50:yr+50,xr-50:xr+50] = final_fish 
                except:
                    blank_img = blank_img
                else: 
                    fish_area = blank_img[yr-50:yr+50,xr-50:xr+50]
                    fish_area_no_fish = cv2.bitwise_and(fish_area, fish_area, mask=fish_mask)   
                    final_fish = cv2.add(fish_area_no_fish,fish)
                    blank_img[yr-50:yr+50,xr-50:xr+50] = final_fish
            #if snake eats apple increase score and find new position for apple
            if  (last_point_x>random_x and last_point_x<(random_x+40) and last_point_y>random_y and last_point_y<(random_y+40)):
                score +=1
                random_x = random.randint(10, 550)
                random_y = random.randint(10, 400)
            #adding blank image to captured frame8
            frame8 = cv2.add(frame8,blank_img)
            #adding apple image to frame8
            roi = frame8[random_y:random_y+40, random_x:random_x+40]
            img_bg = cv2.bitwise_and(roi, roi, mask=apple_mask_inv)
            img_fg = cv2.bitwise_and(apple, apple, mask=apple_mask)
            dst = cv2.add(img_bg, img_fg)
            frame8[random_y:random_y + 40, random_x:random_x + 40] = dst
            cv2.putText(frame8, str("Score - "+str(score)), (250, 450), font, 1, (255, 255, 0), 2, cv2.LINE_AA)
            # checking for snake hitting itself
            if(len(points)>5):
                # a and b are the head points of snake and c,d are all other points
                b = points[len(points)-2]
                a = points[len(points)-1]
                for i in range(len(points)-3):
                    c = points[i]
                    d = points[i+1]
                    if(intersect(a,b,c,d) and len(c)!=0 and len(d)!=0):
                        temp = 0
                        break
                if temp==0:
                    start_time = int(time())
                    q,snake_len,score,temp=0,200,0,1
                    point_x,point_y = 0,0
                    last_point_x,last_point_y,dist,length = 0,0,0,0
                    points = []
                    list_len = []
                    temp=1

            def verify_alpha_channel(frame8):
                try:
                    frame8.shape[3] # looking for the alpha channel
                except IndexError:
                    frame8 = cv2.cvtColor(frame8, cv2.COLOR_BGR2BGRA)
                return frame8
            def apply_color_overlay(frame8, intensity=1, blue=200, green=0, red=0):
                frame8 = verify_alpha_channel(frame8)
                frame_h, frame_w, frame_c = frame8.shape
                sepia_bgra = (blue, green, red, 1)
                overlay = np.full((frame_h, frame_w, 4), sepia_bgra, dtype='uint8')
                cv2.addWeighted(overlay, intensity, frame8, 1.0, 0, frame8)
                return frame8
            frame8 = apply_color_overlay(frame8)
            ret, buffer = cv2.imencode('.jpg', frame8)
            frame8 = buffer.tobytes()
            yield (b'--frame8\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame8 + b'\r\n')

            # increasing the length of snake 40px per second
            if((int(time())-start_time)>1):
                snake_len += 40
                start_time = int(time())
            # increasing the length of snake 40px per second
            if((int(time())-start_time)>1):
                snake_len += 40
                start_time = int(time())
            
                                                           
@app.route('/video_feed/1')
def video_feed_dog():
    print(request.base_url, file=sys.stdout)
    return Response(main_frames('dog'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/video_feed/2')
def video_feed_pig():
    print(request.base_url, file=sys.stdout)
    return Response(main_frames('pig'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed/3')
def video_feed_panda():
    print(request.base_url, file=sys.stdout)
    return Response(main_frames('panda'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')   

@app.route('/video_feed/4')
def video_feed_snake():
    print(request.base_url, file=sys.stdout)
    return Response(main_frames('snake'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')                        

@app.route('/video_feed/5')
def video_feed_fish():
    print(request.base_url, file=sys.stdout)
    return Response(main_frames('fish'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')                        


@app.route('/pig')
def pig():
    return render_template('pig.html')

@app.route('/dog')
def dog():
    return render_template('dog.html')    

@app.route('/panda')
def panda():
    return render_template('panda.html')    

@app.route('/snake')
def snake():
    return render_template('snake.html') 

@app.route('/fish')
def fish():
    return render_template('fish.html')     

@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(host='127.0.0.1')
