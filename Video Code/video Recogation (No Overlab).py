import numpy as np
import cv2, os
import imutils
from imutils.video import FPS

# YOLO 모델 및 클래스 로드
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

# 비디오 파일 경로
video_path = "video1.mp4"

# 비디오 파일 열기
vs = cv2.VideoCapture(video_path)

# 비디오가 제대로 열렸는지 확인
if not vs.isOpened():
    print("Error: Could not open video file.")
    exit()

(W, H) = (None, None)
writer = None

# 현재 스크립트가 위치한 디렉토리 파일을 불러옴
script_dir = os.path.dirname(os.path.abspath(__file__))

# "Output_Videos"라는 폴더가 현재 스크립트 디렉토리 파일 안에 위치하도록 함
output_folder = os.path.join(script_dir, "Output_Videos")
if not os.path.exists(output_folder):   # "Output_Videos"라는 폴더가 존재하는지 확인
    os.makedirs(output_folder)          # 존재하지 않으면 폴더를 생성

# 로드 한 비디오의 파일 확장자의 이름만 추출해 저장
base_name = os.path.splitext(os.path.basename(video_path))[0]

# output_folder 안에 있는 모든 파일과 폴더 이름을 리스트로 반환하고 저장
existing_files = os.listdir(output_folder)

# existing_files 리스트에서 "Result Video"로 시작하는 파일들을 찾기
if existing_files:
    # 비디오 이름과 베이스 이름이 일치하는 파일의 목록을 저장
    matching_files = [filename for filename in existing_files if filename.startswith(f"Result {base_name}")]
    if matching_files:  # 숫자를 추출하여 가장 큰 값을 last_filename에 저장
        last_filename = max([int(filename.split(f"Result {base_name}")[1].split(".")[0]) for filename in matching_files])
    else:               # matching_files가 비어있으면 0으로 설정
        last_filename = 0
else:                   # existing_files가 비어있으면 0으로 설정
    last_filename = 0

# "Result Video"라는 중복되지 않는 비디오 저장
output_path = os.path.join(output_folder, f"Result {base_name} {last_filename + 1}.mp4")

# 비디오 스트림 프레임 반복
while True:
    # 프레임 읽기
    ret, frame = vs.read()

    # 비디오가 끝나면 종료
    if not ret:
        break
    
    if frame is None:
        break  # None일 경우 종료

    frame = imutils.resize(frame, width=1000)
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(net.getUnconnectedOutLayersNames())

    class_ids = []
    confidences = []
    boxes = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > 0.5:  # 객체 탐지 신뢰도 임계값
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)  # NMS 임계값

    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            color = [int(c) for c in COLORS[class_ids[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = "{} : {:.2f}%".format(classes[class_ids[i]], confidences[i] * 100)
            y = y - 15 if y - 15 > 15 else y + 15
            cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.imshow("Real-Time Object Detection", frame)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord("q"):
        break
    
    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MP4V")
        writer = cv2.VideoWriter(output_path, fourcc, 25, (frame.shape[1], frame.shape[0]), True)
    
    writer.write(frame)

# 비디오 파일 닫기
vs.release()
writer.release()
cv2.destroyAllWindows()