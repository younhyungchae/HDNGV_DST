# HDNGV_DST
Multimodal Dialogue State Tracking with HDNGV

## 데이터셋 준비
### Image Classifer Training 데이터셋
이미지를 학습셋과 테스트셋으로 분할
폴더명은 Classification을 위한 class 이름

![image](https://github.com/younhyungchae/HDNGV_DST/assets/104618372/e56bb888-0e95-4d0c-a376-8dfb29537e42)

## 패키지 설치
```
pip install -r requirements.txt
```

## YOLOv8 (Image Classifier 학습)
```
python train_yolo.py
```

## Gradio 실행
```
python gradio.py
```
