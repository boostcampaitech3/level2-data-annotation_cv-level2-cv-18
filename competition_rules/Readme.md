## 베이스라인 모델인 EAST 모델이 정의되어 있는 아래 파일들은 변경사항 없이 그대로 이용해야 합니다.

- model.py
- loss.py
- east_dataset.py
- detect.py

#### 이외의 다른 파일을 변경하거나 새로운 파일을 작성하는 것은 자유롭게 진행하셔도 됩니다.
#### [예시] dataset.py에서 pre-processing, data augmentation 부분을 변경
#### [예시] train.py에서 learning rate scheduling 부분을 변경

### code/model.py
설명 : EAST 모델이 정의된 파일입니다.
수정 불가

### code/loss.py

설명 : 학습을 위한 loss fucntion이 정의되어 있는 파일입니다.
수정 불가

### code/east_dataset.py

설명 : EAST 학습에 필요한 형식의 데이터셋이 정의되어 있는 파일입니다.
수정 불가

### code/dataset.py

설명 : 이미지와 글자 영역의 정보 등을 제공하는 데이터셋이 정의되어 있는 파일입니다.

### code/train.py

설명 : 모델의 학습 절차가 정의되어 있는 파일입니다.

### code/inference.py

설명 : 모델의 추론 절차가 정의되어 있는 파일입니다.

### code/detect.py

설명 : 모델의 추론에 필요한 기타 함수들이 정의되어 있는 파일입니다.
수정 불가

### code/deteval.py

설명 : DetEval 평가를 위한 함수들이 정의되어 있는 파일입니다.

### code/convert_mlt.py

설명 : ICDAR17 데이터셋에 언어 등의 필터를 적용해 새로운 부분 데이터셋을 생성하는 스크립트입니다.

### code/requirements.txt

설명 : 패키지 설치를 위한 파일입니다.

### code/pth

설명 : 이미지넷 기학습 가중치가 들어있는 폴더입니다.

## 평가방법은 7강에서 소개되는 DetEval 방식으로 계산되어 진행됩니다.
DetEval은, 이미지 레벨에서 정답 박스가 여러개 존재하고, 예측한 박스가 여러개가 있을 경우, 박스끼리의 다중 매칭을 허용하여 점수를 주는 평가방법 중 하나 입니다.

평가가 이루어지는 방법은 다음과 같습니다.

1) 모든 정답/예측박스들에 대해서 Area Recall, Area Precision을 미리 계산해냅니다.
여기서 Area Recall, Area Precision은 다음과 같습니다.
Area Recall = 정답과 예측박스가 겹치는 영역 / 정답 박스의 영역
Area Precision = 정답과 예측박스가 겹치는 영역 / 예측 박스의 영역

2) 모든 정답 박스와 예측 박스를 순회하면서, 매칭이 되었는지 판단하여 박스 레벨로 정답 여부를 측정합니다.
박스들이 매칭이 되는 조건은 박스들을 순회하면서,
위에서 계산한 Area Recall, Area Precision이 0 이상일 경우 매칭 여부를 판단하게 되며,
박스의 정답 여부는 Area Recall 0.8 이상, Area Precision 0.4 이상을 기준으로 하고 있습니다.

3) 모든 이미지에 대하여 Recall, Precision을 구한 이후, 최종 F1-Score은 모든 이미지 레벨에서 측정 값의 평균으로 측정됩니다.
테스트 셋은 여러장의 이미지로 구성되어있는데요,
위의 예시에서처럼 모든 이미지들에 대해서 Recall, Precision, 점수를 구한 이후,
모든 이미지들에 대해서 해당 값을 평균내어 최종 점수를 구하게 됩니다.