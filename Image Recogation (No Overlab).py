import numpy as np, cv2
import os

net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_name = net.getLayerNames()
output_layers = [layer_name[i - 1] for i in net.getUnconnectedOutLayers()]

colors= np.random.uniform(0, 255, size=(len(classes), 3))

image_filename = "Image1.jpg"       # 불러올 이미지를 'image_filename'에 저장
img = cv2.imread(image_filename) 

# 새로운 이미지 크기 계산
scale_percent = 20
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)

# 크기 조정
resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

height, width, channels = resized_img.shape

blob = cv2.dnn.blobFromImage(resized_img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
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
        if confidence > 0.05:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            x = int(center_x - w/2)
            y = int(center_y - h/2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

font = cv2.FONT_HERSHEY_PLAIN
for i in indexes:
    x, y, w, h = boxes[i]
    label = str(classes[class_ids[i]])
    color = colors[i]
    cv2.rectangle(resized_img, (x, y), (x + w, y + h), color, 2)
    cv2.putText(resized_img, label, (x, y + 30), font, 2.5, color, 3)

## 인식된 이미지 저장
# 현재 스크립트가 위치한 디렉토리 파일을 불러옴
script_dir = os.path.dirname(os.path.abspath(__file__))

# "Output_Image"라는 폴더가 현재 스크립트 디렉토리 파일 안에 위치하도록 함
output_folder = os.path.join(script_dir, "Output_Images(No overlap)")
if not os.path.exists(output_folder):   # "Output_Images(No overlap)"라는 폴더가 존재하는지 확인
    os.makedirs(output_folder)          # 존재 하지 않으면 폴더를 생성

# 로드 한 이미지의 파일 확장자의 이름만 추출해 저장
base_name = os.path.splitext(os.path.basename(image_filename))[0]

# output_folder 안에 있는 모든 파일과 폴더 이름을 리스트로 반환하고 저장
existing_files = os.listdir(output_folder)

# 이미지 이름과 베이스 이름이 일치하는 파일의 목록을 저장
matching_files = [filename for filename in existing_files if filename.startswith(f"Result {base_name}")]

# 이미지가 이미 저장되지 않은 경우에만 저장
if not matching_files:
    # 이미지의 결과를 저장할 파일 이름 생성
    filename = os.path.join(output_folder, f"Result {base_name}.jpg")

    # 결과 이미지 저장
    cv2.imwrite(filename, resized_img)  # 이미지 저장

# 이미지 창으로 표시
cv2.imshow("Image", resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()