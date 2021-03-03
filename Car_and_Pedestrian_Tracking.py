import cv2

# Our video
web = cv2.VideoCapture('Tesla Autopilot Dashcam Compilation 2018 Version.mp4')

# Our pre-trained car and pedestrian classifier
car_tracker_file = 'cars.xml' 
pedestrian_tracker_file = 'haarcascade_fullbody.xml'

car_tracker = cv2.CascadeClassifier(car_tracker_file)
pedestrian_tracker = cv2.CascadeClassifier(pedestrian_tracker_file)


while True:
    # Read the current frame
    (succesful_frame_read, frame) = web.read() 

    # Safe coding
    if succesful_frame_read:
        # Must convert to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break  

    cars = car_tracker.detectMultiScale(gray_frame)
    pedestrians = pedestrian_tracker.detectMultiScale(gray_frame)

    for (x,w,h,k) in cars:
        cv2.rectangle(frame, (x, w), (x+h, w + k), (0, 0, 255), 2)

    for (x,w,h,k) in pedestrians:
        cv2.rectangle(frame, (x, w), (x+h, w + k), (0, 255, 255), 2)
    
    cv2.imshow('Clever Programmer Car and Pedestrian Detector', frame)


    key = cv2.waitKey(1)

    if key == 81 or key == 113:
        break

print('Code completed')