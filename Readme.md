# [P stage3] CV 18조 언제오르조

## Project Overview
스마트폰으로 카드를 결제하거나, 카메라로 카드를 인식할 경우 자동으로 카드 번호가 입력되는 경우가 있습니다. 또 주차장에 들어가면 차량 번호가 자동으로 인식되는 경우도 흔히 있습니다. 이처럼 OCR (Optimal Character Recognition) 기술은 사람이 직접 쓰거나 이미지 속에 있는 문자를 얻은 다음 이를 컴퓨터가 인식할 수 있도록 하는 기술로, 컴퓨터 비전 분야에서 현재 널리 쓰이는 대표적인 기술 중 하나입니다.

![image](https://user-images.githubusercontent.com/59071505/164969135-b192c281-3036-4d29-a4cf-a91a452256c1.png)

OCR task는 글자 검출 (text detection), 글자 인식 (text recognition), 정렬기 (Serializer) 등의 모듈로 이루어져 있습니다. 본 대회는 아래와 같은 특징이 있습니다.

- 본 대회에서는 '글자 검출' task 만을 해결
- 예측 csv 파일 제출 (Evaluation) 방식이 아닌 **model checkpoint 와 inference.py 를 제출하여 채점**하는 방식
- **Input** : 글자가 포함된 전체 이미지
- **Output** : bbox 좌표가 포함된 UFO Format



## Dataset
- Aihub Dataset의 경우 Validation의 책표지 데이터만을 사용

|Dataset|Train|Valid|Total|
|---|---|---|---|
|ICDAR17|7200|1800|9000|
|ICDAR19|10000| x |10000|
|Aihub 책표지| x | 5063 |5063|

├── code  
│   ├── model.py  
│   ├── loss.py  
│   ├── east_dataset.py  
│   ├── dataset.py  
│   ├── train.py  
│   ├── inference.py  
│   ├── detect.py  
│   ├── deteval.py  
│   ├── convert_mlt.py  
│   ├── Aihub.py  
│   ├── augmentation.py  
│   ├── split_data.py  
│   ├── OCR_EDA.ipynb  
│   ├── merge_json.ipynb  
│   └── requirements.txt

## 1. Install Requirements
```
pip install -r requirements.txt
```

## 2. Create Dataset
- 각 링크에서 데이터 다운로드 후 ufo 변환
1. [ICDAR17](https://rrc.cvc.uab.es/?ch=8&com=downloads), [ICDAR19](https://rrc.cvc.uab.es/?ch=15&com=downloads)
    ```
    python convert_mlt.py
    ```
    
2. [Aihub](https://aihub.or.kr/aidata/33985/download)
    ```
    python Aihub.py
    ```

## 3. Merge Dataset
1. image
   - 하나의 디렉토리로 이동하여 합치기
  
2. annotation
    > merge_json.ipynb

## 4. Train
```
Model : EAST
optimizer : Adam
scheduler : MultiStepLR
epoch : 200
loss : EAST_Loss(sm_Loss(Dice coefficient) + geo_Loss(IOU,Cosine))
```

```
python train.py
```
