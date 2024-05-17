import numpy as np
import cv2

# YOLO 네트워크 가중치와 구성 파일 로드
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# 클래스 이름 로드
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# 네트워크의 모든 레이어 이름 가져오기
layer_name = net.getLayerNames()

# 출력 레이어 이름 가져오기
output_layers = [layer_name[i - 1] for i in net.getUnconnectedOutLayers()]

# 각 클래스에 대한 랜덤 색상 설정
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# 이미지 로드
img = cv2.imread("Image2.jpg")

# 새로운 이미지 크기 계산
scale_percent = 20
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)

# 크기 조정
resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
height, width, channels = resized_img.shape

# 이미지를 블롭으로 변환 (YOLO 네트워크에 입력하기 위한 전처리)
blob = cv2.dnn.blobFromImage(resized_img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)  # 네트워크에 블롭 입력

# 네트워크 순방향 실행 (출력 레이어를 통한 예측)
outs = net.forward(output_layers)

# 예측된 바운딩 박스와 관련 정보 초기화
class_ids = []
confidences = []
boxes = []

# 네트워크의 모든 출력에 대해 반복
for out in outs:
    for detection in out:
        scores = detection[5:]  # 클래스 확률 가져오기
        class_id = np.argmax(scores)  # 가장 높은 확률의 클래스 ID 가져오기
        confidence = scores[class_id]  # 그 클래스의 확률 값
        if confidence > 0.5:  # 신뢰도가 50% 이상인 경우에만 고려
            # 박스 중심 좌표, 너비, 높이 가져오기
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            x = int(center_x - w / 2)  # 좌측 상단 x 좌표 계산
            y = int(center_y - h / 2)  # 좌측 상단 y 좌표 계산
            # 박스, 신뢰도, 클래스 ID 저장
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# 비최대 억제를 사용하여 중복된 박스 제거
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# 결과 이미지에 바운딩 박스와 라벨 그리기
font = cv2.FONT_HERSHEY_PLAIN
for i in indexes:
    x, y, w, h = boxes[i]
    label = str(classes[class_ids[i]])
    color = colors[class_ids[i]]
    cv2.rectangle(resized_img, (x, y), (x + w, y + h), color, 2)  # 박스 그리기
    cv2.putText(resized_img, label, (x, y + 30), font, 3, color, 3)  # 라벨 그리기

# 결과 이미지 표시
cv2.imshow("Image", resized_img)
cv2.waitKey(0)  # 키 입력 대기
cv2.destroyAllWindows()  # 모든 창 닫기