import numpy as np
import cv2
import os

# 절대 경로로 설정 (적절히 수정)
config_path = "/Users/leegeunbum/TeamProject/TeamZZang/yolov3.cfg"
weights_path = "/Users/leegeunbum/TeamProject/TeamZZang/yolov3.weights"
names_path = "/Users/leegeunbum/TeamProject/TeamZZang/coco.names"
image_path = "/Users/leegeunbum/TeamProject/TeamZZang/Image1.jpg"

# 파일 존재 여부 확인
if not os.path.exists(config_path):
    print(f"Error: {config_path} 파일을 찾을 수 없습니다.")
if not os.path.exists(weights_path):
    print(f"Error: {weights_path} 파일을 찾을 수 없습니다.")
if not os.path.exists(names_path):
    print(f"Error: {names_path} 파일을 찾을 수 없습니다.")
if not os.path.exists(image_path):
    print(f"Error: {image_path} 파일을 찾을 수 없습니다.")

# Net 로드
net = cv2.dnn.readNet(weights_path, config_path)
classes = []
with open(names_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

colors = np.random.uniform(0, 255, size=(len(classes), 3))

img = cv2.imread(image_path)
img = cv2.resize(img, None, fx=0.4, fy=0.4)
height, width, channels = img.shape

# 새로운 이미지 크기 계산
scale_percent = 20
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)

# 크기 조정
resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

height, width, channels = resized_img.shape

blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

class_ids = []
confidences = []
boxes = []


for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

font = cv2.FONT_HERSHEY_PLAIN
for i in indexes.flatten():
    x, y, w, h = boxes[i]
    label = str(classes[class_ids[i]])
    color = colors[i]
    cv2.rectangle(resized_img, (x, y), (x + w, y + h), color, 2)
    cv2.putText(resized_img, label, (x, y + 30), font, 3, color, 3)

cv2.imshow("Image", resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
