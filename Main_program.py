# Import libraries
import cv2   # Access camera
import mediapipe as mp     # Object detection,hand recognision,face recognision
# mediapipe -- google -- inbulit methods, developers.google.com/mediapipe
# mediapipe -- neural networks analysis on image -> Hand (Identified) - Tracking


import pickle
model=pickle.load(open('model.pkl','rb')) # Loading the ML model


# Select some attributes --- Instructing mediapipe 
mp_hands=mp.solutions.hands  # hands
mp_drawing=mp.solutions.drawing_utils # Drawing utility to draw circles at landmarks
mp_drawing_styles=mp.solutions.drawing_styles # It ia drawing a line in multi colors


# select the camera
cam=cv2.VideoCapture(0)  # 0 means Laptop camera
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,  # 50%
    min_tracking_confidence=0.5
) as hands:     # Min confidence (deticetion.tracking - 50%)
    while cam.isOpened():   # whether camera is open or not
        success,image=cam.read()  # camera will read the frame of the image
        imageWidth,imageHeight=image.shape[:2]   # Width, Height, Depth(Channels)
        if not success:  # camera is not working
            continue
        # if success: convert image from BGR to RGB 
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        results=hands.process(image) # Pre-trained Deeplearning model from mediapipe --> return the coordinates of hands and landmarks
        image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)  # RGB to BGR

        # Checking there are landmarks or not  --- landmarks are the circle points in hand viewed in camera
        if results.multi_hand_landmarks: # hands are identified # if hands are there in image or not
            # More than one hands
            for hand_landmarks in results.multi_hand_landmarks:  # left,right hands

                mp_drawing.draw_landmarks(image,hand_landmarks,
                                          mp_hands.HAND_CONNECTIONS,
                                          mp_drawing_styles.get_default_hand_landmarks_style(),
                                          mp_drawing_styles.get_default_hand_connections_style())  # drawing lines to the landmarks(circles)
                
                # ********************** $$$$$$#
                data=[]  # Empty list to store the landmarks data
                for point in mp_hands.HandLandmark:   # Traverse through 21 landmarks
                    normalizedLandmark=hand_landmarks.landmark[point]   # extract the landmark of respective point
                    data.append(normalizedLandmark.x)   # X-coordinate
                    data.append(normalizedLandmark.y)  # y-coordinate
                    data.append(normalizedLandmark.z)   # z-coordinate
                print(len(data))  # 63 = 21+21+21
                # print(data)  

                result=model.predict([data]) # data = [[ ]]
                print(result)
                font=cv2.FONT_HERSHEY_SIMPLEX  # font style

                org=(50,50)  # position on the camera
                fontScale=1  # Bold
                color=(255,0,255) # Blue,Green,Red
                thickness=2 # Line Thickness
                image=cv2.putText(image,result[0],org,font,fontScale,color,thickness,cv2.LINE_AA)


        cv2.imshow('Hand Tracking',image) # Displaying the frame after the landmarks
        if cv2.waitKey(5) & 0xFF==27:   # means pressing esc key or q will break the infinite loop
            break
cam.release()   # Stopping the camera










