import cv2

# Initialize video capture
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Load class names
classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
print(f"Loaded {len(classNames)} class names.")

# Paths to the configuration and weights files
configPath = r'C:\Users\fiona\Desktop\objectdetextion\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = r'C:\Users\fiona\Desktop\objectdetextion\frozen_inference_graph.pb'

# Load the model
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Video stream processing
while True:
    success, img = cap.read()
    if not success:
        break

    # Object detection
    classIds, confs, bbox = net.detect(img, confThreshold=0.5)
    
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            if 0 < classId <= len(classNames):
                label = f'{classNames[classId - 1].upper()} {confidence:.2f}'
                cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                cv2.putText(img, label, (box[0] + 10, box[1] + 30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            else:
                print(f"Invalid classId: {classId}")
    else:
        print("No objects detected.")

    # Display the output
    cv2.imshow("Output", img)
    
    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



