## 이미지 & 영상 객체 인식 및 검출
> Yolo 모델을 사용하여 OpenCV로 인식 및 검출

YOLO란?
> YOLO(You Only Look Once)는 물체 검출(Object Detection)에서 대표적인 딥러닝 모델이라고 할 수 있습니다. 물체 검출(Object Detection)은 이미지 내에 존재하는 물체를 찾고, 이들을 구분하는 기술입니다. 영상처리나 CV분야에서 기본적이고 또 많이 쓰는 기법입니다. YOLO(You Only Look Once)모델은 말 그대로 이미지를 한번만 보고 바로 물체를 검출하는 딥러닝 기술을 이용한 물체 검출 모델입니다.
> YOLO의 특징은 3가지 정도 있습니다.
  1. 이미지 전체를 한 번만 본다.
  2. 통합된 모델을 사용해 간단합니다.
  3. 기존의 모델보다 빠른 성능으로 실시간 객체 검출이 가능합니다.
  빠르고 간단한 장점이 있지만, 작은 객체의 인식률이 떨어진다는 단점도 있습니다.

Yolo 설치 가이드
> yolov3.weight, yolov3.cfg, coco.name 파일
  > 드라이브 주소
  * #### https://drive.google.com/file/d/1gglYx1BXLOFQ0ettKLMpozoc6In8G2xq/view
  > 참고 GitHub 주소
  * #### https://github.com/patrick013/Object-Detection---Yolov3/blob/master/model/yolov3.weights

객체 인식이란?
> 객체 인식은 이미지 또는 비디오 상의 객체를 식별하는 컴퓨터 비전 기술입니다.
  객체 인식은 딥러닝과 머신 러닝 알고리즘을 통해 산출되는 핵심 기술입니다.
  사람은 사진 또는 비디오를 볼 때 인물, 물체, 장면 및 시각적 세부 사항을 쉽게 알아챌 수 있습니다.

## 예시 이미지
### 원본 이미지 / 출력시 이미지
<figure class="half">  
  <a href="link"><img src="Image%20File/Image1.jpg" alt="원본사진" width="300" height="200"> 
  <a href="link"><img src="Saved_Images/Save%20Image1.jpg" alt="검출사진" width="300" height="200">
  <figcaption></figcaption>
</figure>


#### > 참고 자료
  * #### https://velog.io/@hhhong/Object-Detection-with-YOLO
  * #### https://brunch.co.kr/@aischool/11
  * #### https://kr.mathworks.com/solutions/image-video-processing/object-recognition.html

## 내 컴퓨터에 OpenCV 설치하자

* [Window에 OpenCV 설치](https://docs.opencv.org/3.4.3/d5/de5/tutorial_py_setup_in_windows.html)
* [Linux나 Mac에 OpenCV 설치](https://docs.opencv.org/4.0.0-beta/d2/de6/tutorial_py_setup_in_ubuntu.html)

### 설치확인
```bash
$ python
>>> import cv2
```
간단한 코드 설명
### Yolo 파일 로드
```bash
import numpy as np
import cv2
import imutils
import os

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
```
### 이미지 로드 및 크기 조절
```bash
def load_image():
    image_filename = "Image1.jpg"  # 불러올 이미지를 'image_filename'에 저장
    img = cv2.imread(image_filename)  # 이미지 읽기
    if img is None:  # 이미지 존재 여부 확인
        print("Error: Could not open image file.")
        return

    # 새로운 이미지 크기 계산
    scale_percent = 15
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    # 크기 조정
    resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
```
### 객체 인식 부분
```bash
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
            scores = detection[5:]           # 클래스 확률 가져오기
            class_id = np.argmax(scores)     # 가장 높은 확률의 클래스 ID 가져오기
            confidence = scores[class_id]    # 그 클래스의 확률 값

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
        label = f"{classes[class_ids[i]]}: {confidences[i] * 100:.2f}%"     # 감지 신뢰도 확률
        color = np.random.randint(0, 255, size=(3,)).tolist()               # 색상 랜덤 지정
        cv2.rectangle(resized_img, (x, y), (x + w, y + h), color, 2)        # 박스 그리기
        text_size = cv2.getTextSize(label, font, 2.5, 3)[0]                 # 텍스트 사이즈
        text_y = y - 10 if y - 10 > 10 else y + 30                          # 텍스트 상자 위에 표시
        cv2.putText(resized_img, label, (x, text_y), font, 1.5, color, 3)   # 라벨 그리기
```
#### 원본 이미지
  <img src="Image%20File/Image2.jpg" alt="원본사진" width="300" height="200">
  
#### 출력시 이미지
  <img src="Saved_Images/Save%20Image2.jpg" alt="검출사진" width="300" height="200">

#### 비디오 로드
```bash
# 비디오 파일 열기
vs = cv2.VideoCapture(video_path)

# 비디오가 제대로 열렸는지 확인
if not vs.isOpened():
    print("Error: Could not open video file.")
    exit()

fps = vs.get(cv2.CAP_PROP_FPS)
total_frames = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))

current_frame = 0
paused = False

W, H = None, None
writer = None

# 원본영상의 프레임
original_fps = fps
frame_buffer = []
```

#### 객체 인식 부분
```bash
# 프레임 생성
def update_frame():
    global current_frame, paused, vs, W, H, writer

    # 영상이 멈춰 있지 않을때
    if not paused:
        ret, frame = vs.read()
        if not ret:
            return

        if frame is None:   # None일 경우 종료
            return 

        # 이전 프레임과 현재 프레임의 중간값 구하기
        if len(frame_buffer) > 0:
            frame = cv2.addWeighted(frame, 0.5, frame_buffer[-1], 0.5, 0)

        # 프레임 버퍼
        frame_buffer.append(frame)
        if len(frame_buffer) > 2:
            frame_buffer.pop(0)

        # 프레임 사이즈 재설정
        frame = imutils.resize(frame, width=800)
        if W is None or H is None:
            (H, W) = frame.shape[:2]

        # 이미지를 블롭으로 변환 (YOLO 네트워크에 입력하기 위한 전처리)
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        layerOutputs = net.forward(net.getUnconnectedOutLayersNames())

        # 예측된 바운딩 박스와 관련 정보 초기화
        class_ids = []
        confidences = []
        boxes = []

        # 네트워크의 모든 출력에 대해 반복
        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]          # 클래스 확률 가져오기
                class_id = np.argmax(scores)    # 가장 높은 확률의 클래스 ID 가져오기
                confidence = scores[class_id]   # 그 클래스의 확률 값

                if confidence > 0.5:  # 신뢰도가 50% 이상인 경우에만 고려
                    # 박스 중심 좌표, 너비, 높이 가져오기
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))  # 좌측 상단 x 좌표 계산
                    y = int(centerY - (height / 2)) # 좌측 상단 y 좌표 계산
                    # 박스, 신뢰도, 클래스 ID 저장
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # 비최대 억제를 사용하여 중복된 박스 제거
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)  # NMS 임계값

        # 결과 이미지에 바운딩 박스와 라벨 그리기
        if len(idxs) > 0:
            for i in idxs.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                color = [int(c) for c in COLORS[class_ids[i]]]                              # 색상 랜덤 지정
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)                      # 박스 그리기
                text = "{} : {:.2f}%".format(classes[class_ids[i]], confidences[i] * 100)   # 텍스트
                y = y - 15 if y - 15 > 15 else y + 15                                       # 텍스트 상자 위에 표시
                cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)   # 라벨 그리기
```
#### 원본 영상

#### 출력시 영상

#### tkinter로 버튼 설정(이미지)
```bash

# refresh 버튼 설정
refresh_button = tk.Button(root, text="Refresh", command=load_image, image=refresh_photo, compound=tk.BOTTOM, width=35, height=40, bg="white")
refresh_button.pack(side=tk.LEFT, padx=20, pady=30)
    
# save 버튼 설정
save_button = tk.Button(root, text="Save", command=saved_image, image=save_photo, compound=tk.BOTTOM, width=35, height=40, bg="white")
save_button.pack(side=tk.RIGHT, padx=20, pady=30)
```
#### 이미지 버튼 설정

#### tkinter로 버튼 설정(영상)
```bash
# tkinter 설정
root = tk.Tk()
root.title("Real-Time Object Detection")

# 현재 시간을 나타낼 레이블
time_label = tk.Label(root, text="00:00", font=("Helvetica", 12))
time_label.pack()

# backward 실행 함수 생성
def backward():
    global current_frame, backward_photo, backward_button
    current_time = int(current_frame / fps)     # 시간 계산
    current_time -= 3                           # 이동할 시간 (초) 조정
    if current_time < 0:
        current_time = 0
    current_frame = int(current_time * fps)
    vs.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

    # 이미지를 버튼에 설정
    backward_button.config(image=backward_photo)

# backward 버튼 나타내기
backward_button = tk.Button(root, text=" -3 ", command=backward, image=backward_photo, compound=tk.BOTTOM, width=50, height=40, bg="white")
backward_button.pack(side='left', padx=10, pady=10)

# forward 실행 함수 생성
def forward():
    global current_frame
    current_time = int(current_frame / fps)       # 시간 계산
    current_time += 3                             # 이동할 시간 (초) 조정
    if current_time > total_frames / fps:
        current_time = total_frames / fps
    current_frame = int(current_time * fps)
    vs.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

# forward 버튼 나타내기
forward_button = tk.Button(root, text=" +3 ", command=forward, image=forward_photo, compound=tk.BOTTOM, width=50, height=40, bg="white")
forward_button.pack(side='right', padx=(10, 10))

## 가운데 위치 시킬 버튼을 위해 프레임 생성
center_frame = tk.Frame(root)
center_frame.pack(expand=True)

# play/stop 실행 함수 생성
def toggle_pause():
    global paused
    paused = not paused
    if paused:
        # 버튼의 이미지를 stop 이미지로 변경
        pause_button.config(image=play_photo, text="Play", compound=tk.BOTTOM)
    else:
        # 버튼의 이미지를 start 이미지로 변경
        pause_button.config(image=stop_photo, text="Stop", compound=tk.BOTTOM)

# play/stop 버튼 나타내기
pause_button = tk.Button(center_frame, text="Stop", command=toggle_pause, image=stop_photo, compound=tk.BOTTOM, width=35, height=40, bg="white")
pause_button.pack(side='left', expand=True)
    
# 이미지를 PhotoImage 객체로 변환
restart_photo = ImageTk.PhotoImage(restart_image)

# restart 실행 함수 생성
def restart():
    global current_frame, paused
    current_frame = 0
    vs.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
    paused = False
    toggle_pause()
    slider.set(current_frame)

# restart 버튼 나타내기
restart_button = tk.Button(center_frame, text="Restart", command=restart, image=restart_photo, compound=tk.BOTTOM, width=35, height=40, bg="white")
restart_button.pack(side='left', padx=(10, 10))
```
#### 영상 버튼 생성

#### 이미지 출처
 ![Image1] <a href="https://www.pexels.com/ko-kr/photo/1108099/"> 출처 Pexels/Chevanon Photography </a>
 ![Image2] <a href="https://www.pexels.com/ko-kr/photo/8916937/"> 출처 Pexels/Lu Li </a>
 
### - 관련주소
#### https://kr.mathworks.com/solutions/image-video-processing/object-recognition.html



