import numpy as np, cv2, imutils
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# YOLO 모델 및 클래스 로드
net = cv2.dnn.readNet("yolov3.weights", "Yolo/yolov3.cfg")
classes = []
with open("Yolo/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

# 비디오 파일 경로
video_path = "Video File/video2.mp4"

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

# tkinter 설정
root = tk.Tk()
root.title("Real-Time Object Detection")

# Canvas 생성
canvas = tk.Canvas(root, width=800, height=350, bg='lightgray')
canvas.pack()

# 현재 시간을 나타낼 레이블
time_label = tk.Label(root, text="00:00", font=("Helvetica", 12))
time_label.pack()

# 슬라이더 콜백 함수
def on_trackbar(val):
    global current_frame, vs
    current_frame = int(val)
    vs.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
    ret, frame = vs.read()
    if ret:
        display_frame(frame)

# 슬라이더 생성
slider = ttk.Scale(root, from_=0, to=total_frames - 1, orient='horizontal', command=on_trackbar, length=300)
slider.pack(fill='x', padx=10, pady=10)

def display_frame(frame):
    frame = imutils.resize(frame, width=800)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = ImageTk.PhotoImage(image)
    canvas.create_image(0, 0, anchor=tk.NW, image=image)
    canvas.image = image

# 원본영상의 프레임
original_fps = fps
frame_buffer = []

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

        # display 실행
        display_frame(frame)

        # 현재 프레임에 맞게 슬라이더 위치 업데이트
        current_frame = int(vs.get(cv2.CAP_PROP_POS_FRAMES))
        slider.set(current_frame)

    # 현재 재생 시간 업데이트
    current_time = int(current_frame / original_fps)
    time_str = "{:02d}:{:02d}".format(current_time // 60, current_time % 60)
    time_label.config(text=time_str)

    # 다음 프레임 재생 간격 설정 (원본 비디오의 FPS에 따라)
    next_frame_delay = int(50 / original_fps)
    root.after(next_frame_delay, update_frame)

## back 버튼 생성
# UI에서 이미지 불러오기
back_img = "UI/back.jpg"

# 이미지 불러오기 및 크기 조정
backward_image = Image.open(back_img)             # 이미지 열기
backward_image = backward_image.resize((30, 28))  # 이미지 크기 조정
    
# 이미지를 PhotoImage 객체로 변환
backward_photo = ImageTk.PhotoImage(backward_image)

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

## front 버튼 생성
# UI에서 이미지 불러오기
front_img = "UI/front.jpg"

# 이미지 불러오기 및 크기 조정
forward_image = Image.open(front_img)           # 이미지 열기
forward_image = forward_image.resize((30, 28))  # 이미지 크기 조정
    
# 이미지를 PhotoImage 객체로 변환
forward_photo = ImageTk.PhotoImage(forward_image)

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

## stop/play 버튼 생성
# UI에서 이미지 불러오기
stop_img = "UI/stop.jpg"
play_img = "UI/play.jpg"

# 이미지 불러오기 및 크기 조정
stop_image = Image.open(stop_img)           # 이미지 열기
stop_image = stop_image.resize((28, 28))    # 이미지 크기 조정
play_image = Image.open(play_img)           # 이미지 열기
play_image = play_image.resize((25, 25))    # 이미지 크기 조정

# 이미지를 PhotoImage 객체로 변환
stop_photo = ImageTk.PhotoImage(stop_image)
play_photo = ImageTk.PhotoImage(play_image)

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

## restart 버튼 생성
# UI에서 이미지 불러오기
restart_img = "UI/restart.jpg"

# 이미지 불러오기 및 크기 조정  
restart_image = Image.open(restart_img)         # 이미지 열기
restart_image = restart_image.resize((28, 28))  # 이미지 크기 조정
    
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

# 'q' 키로 종료
def quit_app(event):
    root.quit()
root.bind("<q>", quit_app)

# 프레임 업데이트 시작
update_frame()

# tkinter 메인 루프 시작
root.mainloop()

# 비디오 파일 닫기
vs.release()
if writer is not None:
    writer.release()
cv2.destroyAllWindows()