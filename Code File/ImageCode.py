import numpy as np, cv2, imutils, os
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# YOLO 네트워크 가중치와 구성 파일 로드
net = cv2.dnn.readNet("yolov3.weights", "Yolo/yolov3.cfg")

# 클래스 이름 로드
classes = []
with open("Yolo/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# 네트워크의 모든 레이어 이름 가져오기
layer_name = net.getLayerNames()

# 출력 레이어 이름 가져오기
output_layers = [layer_name[i - 1] for i in net.getUnconnectedOutLayers()]

# 이미지 로드
def load_image():
    image_filename = "Image File/Image1.jpg"  # 불러올 이미지를 'image_filename'에 저장
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

    # 이미지 위치 조정
    root.update()
    img_x = (canvas.winfo_width() - width) // 2

    # OpenCV 이미지를 Tkinter 이미지로 변환하여 캔버스에 출력
    image = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    photo = ImageTk.PhotoImage(image)
    canvas.delete("all")  # 기존 이미지를 지움
    canvas.config(width=width, height=height)
    canvas.create_image(img_x, 0, anchor=tk.NW, image=photo)
    canvas.image = photo  # Garbage collection 방지를 위해 이미지를 캔버스 객체에 연결    
    canvas.detected_image = image

# 이미지 저장을 위해 함수 생성
def saved_image():
    if not hasattr(canvas, 'detected_image'):
        print("Error: No image to save.")
        return

    # 폴더 생성
    save_folder = "Saved_Images"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # 파일 이름에 번호 붙이기
    base_filename = "Save Image"
    i = 1
    while True:
        save_path = os.path.join(save_folder, f"{base_filename}{i}.jpg")
        if not os.path.exists(save_path):
            break
        i += 1

    # 이미지 저장
    canvas.detected_image.save(save_path)
    
# 루트 윈도우 생성
root = tk.Tk()
root.title("Object Detection")      # 제목을 'Object Detection'으로 설정

# 캔버스 사이즈 조정(이미지 크기에 맞게)
canvas = tk.Canvas(root, width=800, height=1000)
canvas.pack()

load_image()

## 새로고침 버튼 생성
# UI에서 이미지 불러오기
refresh_img = "UI/refresh.jpg"

# 이미지 불러오기 및 크기 조정  
refresh_image = Image.open(refresh_img)
refresh_image = refresh_image.resize((25, 25))  # 이미지 크기 조정
    
# 이미지를 PhotoImage 객체로 변환
refresh_photo = ImageTk.PhotoImage(refresh_image)

# refresh 버튼 설정
refresh_button = tk.Button(root, text="Refresh", command=load_image, image=refresh_photo, compound=tk.BOTTOM, width=35, height=40, bg="white")
refresh_button.pack(side=tk.LEFT, padx=20, pady=30)


## 이미지 저장 버튼 생성
# UI에서 이미지 불러오기
save_img = "UI/save.jpg" 

# 이미지 불러오기 및 크기 조정  
save_image = Image.open(save_img)
save_image = save_image.resize((25, 25))  # 이미지 크기 조정
    
# 이미지를 PhotoImage 객체로 변환
save_photo = ImageTk.PhotoImage(save_image)
    
# save 버튼 설정
save_button = tk.Button(root, text="Save", command=saved_image, image=save_photo, compound=tk.BOTTOM, width=35, height=40, bg="white")
save_button.pack(side=tk.RIGHT, padx=20, pady=30)


# 'q' 키로 종료
def quit_app(event):
    root.quit()
root.bind("<q>", quit_app)

# tkinter 메인 루프 시작
root.mainloop()

cv2.destroyAllWindows()