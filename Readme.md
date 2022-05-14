# 🚀[LEVEL2 P_stage 글자 검출 대회] 언제오르조
![image](https://user-images.githubusercontent.com/59071505/168423013-c7314b24-8fe1-45cb-a5a3-61d05873f43f.png)

&nbsp; 
## 🔥 Member 🔥
<table>
  <tr height="125px">
    <td align="center" width="120px">
      <a href="https://github.com/kimkihoon0515"><img src="https://avatars.githubusercontent.com/kimkihoon0515"/></a>
    </td>
    <td align="center" width="120px">
      <a href="https://github.com/ed-kyu"><img src="https://avatars.githubusercontent.com/ed-kyu"/></a>
    </td>
    <td align="center" width="120px">
      <a href="https://github.com/GwonPyo"><img src="https://avatars.githubusercontent.com/GwonPyo"/></a>
    </td>
    <td align="center" width="120px">
      <a href="https://github.com/ysw2946"><img src="https://avatars.githubusercontent.com/ysw2946"/></a>
    </td>
    <td align="center" width="120px">
      <a href="https://github.com/jsh0551"><img src="https://avatars.githubusercontent.com/jsh0551"/></a>
    </td>
    <td align="center" width="120px">
      <a href="https://github.com/YJ0522771"><img src="https://avatars.githubusercontent.com/YJ0522771"/></a>
    </td>

  </tr>
  <tr height="70px">
    <td align="center" width="120px">
      <a href="https://github.com/kimkihoon0515">김기훈_T3019</a>
    </td>
    <td align="center" width="120px">
      <a href="https://github.com/ed-kyu">김승규_T3037</a>
    </td>
    <td align="center" width="120px">
      <a href="https://github.com/GwonPyo">남권표_T3072</a>
    </td>
    <td align="center" width="120px">
      <a href="https://github.com/ysw2946">유승우_T3130</a>
    </td>
    <td align="center" width="120px">
      <a href="https://github.com/jsh0551">장수호_T3185</a>
    </td>
    <td align="center" width="120px">
      <a href="https://github.com/YJ0522771">조유진_T3208</a>
    </td>
  </tr>
</table>

&nbsp; 
## 🔍Project Overview

스마트폰으로 카드를 결제하거나, 카메라로 카드를 인식할 경우 자동으로 카드 번호가 입력되는 경우가 있습니다. 또 주차장에 들어가면 차량 번호가 자동으로 인식되는 경우도 흔히 있습니다. 이처럼 OCR (Optimal Character Recognition) 기술은 사람이 직접 쓰거나 이미지 속에 있는 문자를 얻은 다음 이를 컴퓨터가 인식할 수 있도록 하는 기술로, 컴퓨터 비전 분야에서 현재 널리 쓰이는 대표적인 기술 중 하나입니다.

OCR task는 글자 검출 (text detection), 글자 인식 (text recognition), 정렬기 (Serializer) 등의 모듈로 이루어져 있습니다. 본 대회는 아래와 같은 특징이 있습니다.

&nbsp;

## ❗Competitioin_rules
- 본 대회에서는 '글자 검출' task 만을 해결
  
- **Input** : 글자가 포함된 전체 이미지
- **Output** : bbox 좌표가 포함된 UFO Format
- 베이스라인 모델인 EAST 모델이 정의되어 있는 아래 파일들은 변경사항 없이 그대로 이용해야 합니다.
    - model.py
    - loss.py
    - east_dataset.py
    - detect.py

- 평가방법 : Deteval
  1) 모든 정답/예측박스들에 대해서 Area Recall, Area Precision을 미리 계산해냅니다.
  2) 모든 정답 박스와 예측 박스를 순회하면서, 매칭이 되었는지 판단하여 박스 레벨로 정답 여부를 측정합니다.
  3) 모든 이미지에 대하여 Recall, Precision을 구한 이후, 최종 F1-Score은 모든 이미지 레벨에서 측정 값의 평균으로 측정됩니다.

&nbsp;

## 🗂️Dataset
- Upstage_data : 1288 images
  
- ICDAR17_MLT : 9000 images
- ICDAR19_MLT : 10000 images
- Aihub OCR : more or less 5000 images

&nbsp;

## 🧱Structure
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
└── data

&nbsp;

## 🧪Experiments
1. Upstage annotation 수정
   - 캠퍼들이 작업한 Upstage image들의 annotation이 제대로 되어있지 않은 것을 확인
   - 1288장의 이미지들을 전수조사하여 Labelme를 통해 수정

    |  | Data | F1 score | recall | precision |
    | --- | --- | --- | --- | --- |
    | 수정 전 | ICDAR17_MLT(ko) + upstage data | 0.4517 | 0.3470 | 0.6468 |
    | 수정 후 | ICDAR17_MLT(ko) + upstage data | 0.5078 | 0.3938 | 0.7150 |

&nbsp;

2. ICDAR 데이터의 제외영역 포함 여부
   - ICDAR 데이터의 경우, annotation의 기준이 기존의 annotation guide와 상당히 다른 것을 확인
   - test data의 경우 기존의 annotation guide를 따를거라 생각하여, 제외영역을 학습에 포함

    |  | Data | F1 score | recall | precision |
    | --- | --- | --- | --- | --- |
    | 제외영역 불포함 | ICDAR17_MLT(ko) + ICDAR19_MLT(ko) | 0.5477 | 0.4279 | 0.7608 |
    | 제외영역 포함 | ICDAR17_MLT(ko) + ICDAR19_MLT(ko) | 0.5856 | 0.4712 | 0.7733 |

&nbsp;

3. Augmentation
   - augmentation이 적용된 이미지를 시각화해보면서 적절한 augmentation 기법을 선정
   - 대회 Base augmentation(flip,rotate,crop)
   - 이외 Augmentation(RandomBrightnessContrast, GaussNoise, CLAHE)

    |  | Data | F1 score | recall | precision |
    | --- | --- | --- | --- | --- |
    | Base | ICDAR17_MLT(ko) | 0.4586 | 0.3495 | 0.6664 |
    | Add augmentation | ICDAR17_MLT(ko) | 0.6270 | 0.5241 | 0.7801 |

&nbsp;

4. 다양한 데이터셋 비교
   - Korean, English 뿐만 아니라 다양한 언어를 학습에 포함
   - 수정한 Upstage data, ICDAR19_MLT, Aihub OCR 데이터셋 사용
  
    |  | Data | F1 score | recall | precision |
    | --- | --- | --- | --- | --- |
    | 1 | ICDAR17_MLT(ko) + upstage data | 0.5078 | 0.3938 | 0.7150 |
    | 2 | ICDAR17_MLT(ko) + upstage data + Aihub(실제 야외 촬영 한글 이미지)| 0.3942 | 0.2896 | 0.6168 |
    | 3 | ICDAR17_MLT(en,ko) + upstage data | 0.5620 | 0.4553 | 0.7338 |
    | 4 | ICDAR17_MLT(all) + upstage data | 0.6717 | 0.5772 | 0.8031 |
    | 5 | ICDAR17_MLT(ko) + ICDAR19_MLT(ko) | 0.5477 | 0.4279 | 0.7608 |
    | 6 | ICDAR19_MLT(Except French) | 0.6719 | 0.5842 | 0.7906 |
    | 7 | ICDAR17_MLT(all) + ICDAR19_MLT(all) + upstage data | 0.6443 | 0.5549 | 0.7682 |


&nbsp;

## 🏆Result
- 총 19 팀 참여
- Public : 13등 -> Private : 5등

![image](https://user-images.githubusercontent.com/59071505/168441937-2fdb6476-7554-4208-8abc-386c8d71fafc.png)

&nbsp;
|  | Data | F1 score | recall | precision |
| --- | --- | --- | --- | --- |
| 1 | ICDAR17_MLT + upstage data | 0.6717 -> 0.6596 | 0.5772 -> 0.5778 | 0.8031 -> 0.7685 |
| 2 | ICDAR19_MLT(Except French) | 0.6719 -> 0.6843 | 0.5842 -> 0.6039 | 0.7906 -> 0.7893 |


&nbsp;

## 💡Usage
1. Install Requirements
    ```
    pip install -r requirements.txt
    ```

2. Create Dataset
- 각 링크에서 데이터 다운로드 후 ufo 변환

  - [ICDAR17](https://rrc.cvc.uab.es/?ch=8&com=downloads), [ICDAR19](https://rrc.cvc.uab.es/?ch=15&com=downloads)
      ```
      python code/convert_mlt.py
      ```
      
  - [Aihub](https://aihub.or.kr/aidata/33985/download)
      ```
      python code/Aihub.py
      ```

3. Merge Dataset
   - image
      - 복사 붙여넣기를 통해 한 디렉토리로 합치기
     
   - annotation
       > merge_json.ipynb

4. Train
    ```
    Model : EAST
    optimizer : Adam
    scheduler : MultiStepLR
    epoch : 200
    loss : EAST_Loss(sm_Loss(Dice coefficient) + geo_Loss(IOU,Cosine))
    ```

    ```
    python code/train.py
    ```
