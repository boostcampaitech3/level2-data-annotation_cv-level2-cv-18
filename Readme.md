# ๐[LEVEL2 P_stage ๊ธ์ ๊ฒ์ถ ๋ํ] ์ธ์ ์ค๋ฅด์กฐ
![image](https://user-images.githubusercontent.com/59071505/168423013-c7314b24-8fe1-45cb-a5a3-61d05873f43f.png)

&nbsp; 
## ๐ฅ Member ๐ฅ
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
      <a href="https://github.com/kimkihoon0515">๊น๊ธฐํ_T3019</a>
    </td>
    <td align="center" width="120px">
      <a href="https://github.com/ed-kyu">๊น์น๊ท_T3037</a>
    </td>
    <td align="center" width="120px">
      <a href="https://github.com/GwonPyo">๋จ๊ถํ_T3072</a>
    </td>
    <td align="center" width="120px">
      <a href="https://github.com/ysw2946">์ ์น์ฐ_T3130</a>
    </td>
    <td align="center" width="120px">
      <a href="https://github.com/jsh0551">์ฅ์ํธ_T3185</a>
    </td>
    <td align="center" width="120px">
      <a href="https://github.com/YJ0522771">์กฐ์ ์ง_T3208</a>
    </td>
  </tr>
</table>

&nbsp; 
## ๐Project Overview

์ค๋งํธํฐ์ผ๋ก ์นด๋๋ฅผ ๊ฒฐ์ ํ๊ฑฐ๋, ์นด๋ฉ๋ผ๋ก ์นด๋๋ฅผ ์ธ์ํ  ๊ฒฝ์ฐ ์๋์ผ๋ก ์นด๋ ๋ฒํธ๊ฐ ์๋ ฅ๋๋ ๊ฒฝ์ฐ๊ฐ ์์ต๋๋ค. ๋ ์ฃผ์ฐจ์ฅ์ ๋ค์ด๊ฐ๋ฉด ์ฐจ๋ ๋ฒํธ๊ฐ ์๋์ผ๋ก ์ธ์๋๋ ๊ฒฝ์ฐ๋ ํํ ์์ต๋๋ค. ์ด์ฒ๋ผ OCR (Optimal Character Recognition) ๊ธฐ์ ์ ์ฌ๋์ด ์ง์  ์ฐ๊ฑฐ๋ ์ด๋ฏธ์ง ์์ ์๋ ๋ฌธ์๋ฅผ ์ป์ ๋ค์ ์ด๋ฅผ ์ปดํจํฐ๊ฐ ์ธ์ํ  ์ ์๋๋ก ํ๋ ๊ธฐ์ ๋ก, ์ปดํจํฐ ๋น์  ๋ถ์ผ์์ ํ์ฌ ๋๋ฆฌ ์ฐ์ด๋ ๋ํ์ ์ธ ๊ธฐ์  ์ค ํ๋์๋๋ค.

OCR task๋ ๊ธ์ ๊ฒ์ถ (text detection), ๊ธ์ ์ธ์ (text recognition), ์ ๋ ฌ๊ธฐ (Serializer) ๋ฑ์ ๋ชจ๋๋ก ์ด๋ฃจ์ด์ ธ ์์ต๋๋ค. ๋ณธ ๋ํ๋ ์๋์ ๊ฐ์ ํน์ง์ด ์์ต๋๋ค.

&nbsp;

## โCompetitioin_rules
- ๋ณธ ๋ํ์์๋ '๊ธ์ ๊ฒ์ถ' task ๋ง์ ํด๊ฒฐ
  
- **Input** : ๊ธ์๊ฐ ํฌํจ๋ ์ ์ฒด ์ด๋ฏธ์ง
- **Output** : bbox ์ขํ๊ฐ ํฌํจ๋ UFO Format
- ๋ฒ ์ด์ค๋ผ์ธ ๋ชจ๋ธ์ธ EAST ๋ชจ๋ธ์ด ์ ์๋์ด ์๋ ์๋ ํ์ผ๋ค์ ๋ณ๊ฒฝ์ฌํญ ์์ด ๊ทธ๋๋ก ์ด์ฉํด์ผ ํฉ๋๋ค.
    - model.py
    - loss.py
    - east_dataset.py
    - detect.py

- ํ๊ฐ๋ฐฉ๋ฒ : Deteval
  1) ๋ชจ๋  ์ ๋ต/์์ธก๋ฐ์ค๋ค์ ๋ํด์ Area Recall, Area Precision์ ๋ฏธ๋ฆฌ ๊ณ์ฐํด๋๋๋ค.
  2) ๋ชจ๋  ์ ๋ต ๋ฐ์ค์ ์์ธก ๋ฐ์ค๋ฅผ ์ํํ๋ฉด์, ๋งค์นญ์ด ๋์๋์ง ํ๋จํ์ฌ ๋ฐ์ค ๋ ๋ฒจ๋ก ์ ๋ต ์ฌ๋ถ๋ฅผ ์ธก์ ํฉ๋๋ค.
  3) ๋ชจ๋  ์ด๋ฏธ์ง์ ๋ํ์ฌ Recall, Precision์ ๊ตฌํ ์ดํ, ์ต์ข F1-Score์ ๋ชจ๋  ์ด๋ฏธ์ง ๋ ๋ฒจ์์ ์ธก์  ๊ฐ์ ํ๊ท ์ผ๋ก ์ธก์ ๋ฉ๋๋ค.

&nbsp;

## ๐๏ธDataset
- Upstage_data : 1288 images
  
- ICDAR17_MLT : 9000 images
- ICDAR19_MLT : 10000 images
- Aihub OCR : more or less 5000 images

&nbsp;

## ๐งฑStructure
โโโ code  
โย ย  โโโ model.py  
โย ย  โโโ loss.py  
โย ย  โโโ east_dataset.py  
โย ย  โโโ dataset.py  
โย ย  โโโ train.py  
โย ย  โโโ inference.py  
โย ย  โโโ detect.py  
โย ย  โโโ deteval.py  
โย ย  โโโ convert_mlt.py  
โย ย  โโโ Aihub.py  
โย ย  โโโ augmentation.py  
โย ย  โโโ split_data.py  
โย ย  โโโ OCR_EDA.ipynb  
โย ย  โโโ merge_json.ipynb  
โย ย  โโโ requirements.txt  
โโโ data

&nbsp;

## ๐งชExperiments
1. Upstage annotation ์์ 
   - ์บ ํผ๋ค์ด ์์ํ Upstage image๋ค์ annotation์ด ์ ๋๋ก ๋์ด์์ง ์์ ๊ฒ์ ํ์ธ
   - 1288์ฅ์ ์ด๋ฏธ์ง๋ค์ ์ ์์กฐ์ฌํ์ฌ Labelme๋ฅผ ํตํด ์์ 

    |  | Data | F1 score | recall | precision |
    | --- | --- | --- | --- | --- |
    | ์์  ์  | ICDAR17_MLT(ko) + upstage data | 0.4517 | 0.3470 | 0.6468 |
    | ์์  ํ | ICDAR17_MLT(ko) + upstage data | 0.5078 | 0.3938 | 0.7150 |

&nbsp;

2. ICDAR ๋ฐ์ดํฐ์ ์ ์ธ์์ญ ํฌํจ ์ฌ๋ถ
   - ICDAR ๋ฐ์ดํฐ์ ๊ฒฝ์ฐ, annotation์ ๊ธฐ์ค์ด ๊ธฐ์กด์ annotation guide์ ์๋นํ ๋ค๋ฅธ ๊ฒ์ ํ์ธ
   - test data์ ๊ฒฝ์ฐ ๊ธฐ์กด์ annotation guide๋ฅผ ๋ฐ๋ฅผ๊ฑฐ๋ผ ์๊ฐํ์ฌ, ์ ์ธ์์ญ์ ํ์ต์ ํฌํจ

    |  | Data | F1 score | recall | precision |
    | --- | --- | --- | --- | --- |
    | ์ ์ธ์์ญ ๋ถํฌํจ | ICDAR17_MLT(ko) + ICDAR19_MLT(ko) | 0.5477 | 0.4279 | 0.7608 |
    | ์ ์ธ์์ญ ํฌํจ | ICDAR17_MLT(ko) + ICDAR19_MLT(ko) | 0.5856 | 0.4712 | 0.7733 |

&nbsp;

3. Augmentation
   - augmentation์ด ์ ์ฉ๋ ์ด๋ฏธ์ง๋ฅผ ์๊ฐํํด๋ณด๋ฉด์ ์ ์ ํ augmentation ๊ธฐ๋ฒ์ ์ ์ 
   - ๋ํ Base augmentation(flip,rotate,crop)
   - ์ด์ธ Augmentation(RandomBrightnessContrast, GaussNoise, CLAHE)

    |  | Data | F1 score | recall | precision |
    | --- | --- | --- | --- | --- |
    | Base | ICDAR17_MLT(ko) | 0.4586 | 0.3495 | 0.6664 |
    | Add augmentation | ICDAR17_MLT(ko) | 0.6270 | 0.5241 | 0.7801 |

&nbsp;

4. ๋ค์ํ ๋ฐ์ดํฐ์ ๋น๊ต
   - Korean, English ๋ฟ๋ง ์๋๋ผ ๋ค์ํ ์ธ์ด๋ฅผ ํ์ต์ ํฌํจ
   - ์์ ํ Upstage data, ICDAR19_MLT, Aihub OCR ๋ฐ์ดํฐ์ ์ฌ์ฉ
  
    |  | Data | F1 score | recall | precision |
    | --- | --- | --- | --- | --- |
    | 1 | ICDAR17_MLT(ko) + upstage data | 0.5078 | 0.3938 | 0.7150 |
    | 2 | ICDAR17_MLT(ko) + upstage data + Aihub(์ค์  ์ผ์ธ ์ดฌ์ ํ๊ธ ์ด๋ฏธ์ง)| 0.3942 | 0.2896 | 0.6168 |
    | 3 | ICDAR17_MLT(en,ko) + upstage data | 0.5620 | 0.4553 | 0.7338 |
    | 4 | ICDAR17_MLT(all) + upstage data | 0.6717 | 0.5772 | 0.8031 |
    | 5 | ICDAR17_MLT(ko) + ICDAR19_MLT(ko) | 0.5477 | 0.4279 | 0.7608 |
    | 6 | ICDAR19_MLT(Except French) | 0.6719 | 0.5842 | 0.7906 |
    | 7 | ICDAR17_MLT(all) + ICDAR19_MLT(all) + upstage data | 0.6443 | 0.5549 | 0.7682 |


&nbsp;

## ๐Result
- ์ด 19 ํ ์ฐธ์ฌ
- Public : 13๋ฑ -> Private : 5๋ฑ

![image](https://user-images.githubusercontent.com/59071505/168441937-2fdb6476-7554-4208-8abc-386c8d71fafc.png)

&nbsp;
|  | Data | F1 score | recall | precision |
| --- | --- | --- | --- | --- |
| 1 | ICDAR17_MLT + upstage data | 0.6717 -> 0.6596 | 0.5772 -> 0.5778 | 0.8031 -> 0.7685 |
| 2 | ICDAR19_MLT(Except French) | 0.6719 -> 0.6843 | 0.5842 -> 0.6039 | 0.7906 -> 0.7893 |


&nbsp;

## ๐กUsage
1. Install Requirements
    ```
    pip install -r requirements.txt
    ```

2. Create Dataset
- ๊ฐ ๋งํฌ์์ ๋ฐ์ดํฐ ๋ค์ด๋ก๋ ํ ufo ๋ณํ

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
      - ๋ณต์ฌ ๋ถ์ฌ๋ฃ๊ธฐ๋ฅผ ํตํด ํ ๋๋ ํ ๋ฆฌ๋ก ํฉ์น๊ธฐ
     
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
