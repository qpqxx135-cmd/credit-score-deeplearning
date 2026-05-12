# Credit Score Deep Learning

## 프로젝트 개요
고객의 금융 정보를 바탕으로 신용점수 등급을 예측하는 딥러닝 기반 다중분류 프로젝트입니다.  
예측 대상은 `Credit_Score`이며, 클래스는 `Good`, `Standard`, `Poor`입니다.

## 데이터 출처
Kaggle Credit Score Classification  
https://www.kaggle.com/datasets/parisrohan/credit-score-classification

## EDA
- 결측치 0개 확인
- <img width="414" height="462" alt="스크린샷 2026-05-12 오전 11 27 41" src="https://github.com/user-attachments/assets/e8706813-eba5-4695-9a5d-2056951a0655" />

- `Credit_Score` 클래스 분포 확인
- 주요 수치형 변수 분포 확인
- `Outstanding_Debt`, `Interest_Rate`와 신용점수 관계 확인
  <img width="704" height="470" alt="image" src="https://github.com/user-attachments/assets/ef4af4c0-d2db-437d-a8b9-520245c2de87" />
  <img width="686" height="470" alt="image" src="https://github.com/user-attachments/assets/71093827-968b-4405-abf3-c3e6484268d4" />

- 상관관계 히트맵을 통해 수치형 변수들 간의 관계를 확인
  <img width="1383" height="1190" alt="image" src="https://github.com/user-attachments/assets/5884f4a5-0e37-46ee-a92a-a65b46635d50" />


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
기본 모델 Final Validation Accuracy: 71.70%  
개선 모델 50 epoch Best Validation Accuracy: 74.65%  
추가 학습 후 Best Validation Accuracy: 75.42%
