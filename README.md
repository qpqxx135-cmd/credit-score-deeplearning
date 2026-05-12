# Credit Score Deep Learning

## 프로젝트 개요
고객의 금융 정보를 바탕으로 신용점수 등급을 예측하는 딥러닝 기반 다중분류 프로젝트입니다.  
예측 대상은 `Credit_Score`이며, 클래스는 `Good`, `Standard`, `Poor`입니다.

## 데이터 출처
Kaggle Credit Score Classification  
https://www.kaggle.com/datasets/parisrohan/credit-score-classification

## EDA

### 결측치 확인
전체 결측치 개수를 확인한 결과 0개로 나타났습니다.

<img width="414" height="462" alt="null_check" src="https://github.com/user-attachments/assets/e8706813-eba5-4695-9a5d-2056951a0655" />

### Credit_Score 클래스 분포
`Credit_Score`는 `Standard`, `Poor`, `Good` 세 클래스로 구성되어 있으며, `Standard` 클래스의 비율이 가장 높았습니다.

### 주요 수치형 변수 분포
소득, 이자율, 미지급 잔액 등 주요 수치형 변수들의 분포를 확인했습니다.

### Credit_Score별 주요 변수 차이
`Outstanding_Debt`와 `Interest_Rate`는 신용점수 등급별로 뚜렷한 차이를 보였습니다.

<img width="704" height="470" alt="outstanding_debt" src="https://github.com/user-attachments/assets/ef4af4c0-d2db-437d-a8b9-520245c2de87" />

<img width="686" height="470" alt="interest_rate" src="https://github.com/user-attachments/assets/71093827-968b-4405-abf3-c3e6484268d4" />

### 상관관계 분석
상관관계 히트맵을 통해 수치형 변수들 간의 관계를 확인했습니다.

<img width="1383" height="1190" alt="correlation_heatmap" src="https://github.com/user-attachments/assets/5884f4a5-0e37-46ee-a92a-a65b46635d50" />

`Annual_Income`과 `Monthly_Inhand_Salary`는 강한 양의 상관관계를 보였습니다.  
또한 `Interest_Rate`, `Delay_from_due_date`, `Num_of_Delayed_Payment`, `Outstanding_Debt`는 서로 양의 상관관계를 보여 연체, 이자율, 미지급 잔액 관련 변수들이 함께 증가하는 경향을 확인했습니다.  
반면 `Credit_History_Age`는 위험 관련 변수들과 음의 상관관계를 보여 신용 이력이 길수록 연체 및 미지급 잔액 관련 위험이 낮아지는 경향을 확인했습니다.

## 전처리
- `ID`, `Customer_ID`, `Name`, `SSN` 제거
- 범주형 변수 Label Encoding
- train/test split 후 StandardScaler 적용
- 데이터 누수 방지를 위해 train 데이터에만 scaler fit

## 모델링
기본 모델은 64-32 구조의 MLP로 구성했습니다.  
이후 성능 개선을 위해 256-128-64 구조의 ImprovedMLP를 추가로 실험했습니다.

개선 모델에는 다음을 적용했습니다.

- BatchNorm1d
- Dropout
- weight_decay
- batch_size 조정
- learning rate를 낮춘 추가 학습

## 결과

| 모델 | 주요 설정 | Validation Accuracy |
|---|---|---:|
| 기본 MLP | 64-32 hidden layer, batch size 32, epoch 20 | 71.70% |
| 개선 MLP | 256-128-64 hidden layer, BatchNorm, Dropout, batch size 128, epoch 50 | 74.65% |
| 개선 MLP + 추가 학습 | learning rate 0.0005, 추가 5 epoch | 75.42% |

최종적으로 개선 모델에 learning rate를 낮춘 추가 학습을 적용하여 `75.42%`의 Best Validation Accuracy를 달성했습니다.
