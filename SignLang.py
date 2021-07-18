import cv2 #opencv library
import mediapipe as mp #https://google.github.io/mediapipe/solutions/hands #sexy link...sab deep me bataya he

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0) #to capture a video...0 means capture from laptop camera

finger_tips = [8, 12 ,16, 20]
thumb_tip = 4

like_img = cv2.imread("images/Like.jpg") #Read an image from file
like_img = cv2.resize(like_img, (200,180)) #resize image

dislike_img = cv2.imread("images/Dislike.jpg") #Read an image from file
dislike_img = cv2.resize(dislike_img, (200,180)) #resize image

while True:
    ret, img = cap.read() # Capture frame-by-frame
    #cap.read() returns a bool (True/False)

    img = cv2.flip(img,1) #flips img by 180 degree
    h, w, c = img.shape  #rows, columns and channels

    results = hands.process(img)
    #print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            lm_list=[]
            for id_,lm in enumerate(hand_landmark.landmark):
                lm_list.append(lm)

            finger_fold_status=[]
            for tip in finger_tips:    
                x, y= int(lm_list[tip].x*w), int(lm_list[tip].y*h)
                #print(id_, ":" , x, y) 
                cv2.circle(img, (x,y), 15, (255,0,0), cv2.FILLED)

                if lm_list[tip].x<lm_list[tip-3].x:
                    cv2.circle(img, (x,y), 15, (0,255,0), cv2.FILLED)
                    finger_fold_status.append(True)
                else:
                    finger_fold_status.append(False)

            print(finger_fold_status)      

            if all(finger_fold_status):
                if lm_list[thumb_tip].y < lm_list[thumb_tip-1].y < lm_list[thumb_tip-2].y:                
                        print("LIKE") 
                        cv2.putText(img, "LIKE", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)
                        h, w, c = like_img.shape
                        img[35:h+35, 0:w] = like_img

                if lm_list[thumb_tip].y > lm_list[thumb_tip-1].y > lm_list[thumb_tip-2].y:
                        print("DISLIKE")  
                        cv2.putText(img, "DISLIKE", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3) 
                        h, w, c = dislike_img.shape
                        img[35:h+35, 0:w] = dislike_img          
            
            mp_draw.draw_landmarks(img, hand_landmark, mp_hands.HAND_CONNECTIONS)
            # format he mp_draw.DrawingSpec(tuple(rgb color), thickness, circle radius)

    cv2.imshow("Hand Tracking", img) #Display a video in an OpenCV window
    cv2.waitKey(1)