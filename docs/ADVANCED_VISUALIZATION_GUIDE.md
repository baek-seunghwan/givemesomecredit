# 📊 고급 시각화 및 그래프 가이드

## 🎨 직접 만들 수 있는 고퀄리티 그래프

### 1️⃣ **Confusion Matrix (혼동 행렬) - 히트맵**
```
실시간 시각적 표현
┌─────────────────────────────────┐
│      Predicted (예측)            │
│    Negative    │    Positive     │
├────────────┼─────────────┤
│ Negative │   2,365 (TN) │ 3,482 (FP) │
│ Actual   │              │            │
│ Positive │     358 (FN) │16,600 (TP) │
└─────────────────────────────────┘
```

**시각화 요소**:
- 색상 그래디언트 (연한색 → 진한색)
- 각 셀에 숫자 및 백분율 표시
- 대각선 강조 (정확한 예측)

**코드로 구현 가능**: seaborn.heatmap()

---

### 2️⃣ **ROC Curve (ROC 곡선)**
```
특징:
- Y축: True Positive Rate (TPR)
- X축: False Positive Rate (FPR)
- 곡선이 좌상단에 가까울수록 우수
- AUC 값 표시 (0.8890)

용도:
- 모델의 분류 성능 평가
- 임계값에 따른 성능 변화 시각화
```

**시각화 요소**:
- 3개 모델 곡선 비교
- 대각선 기준선 (무작위 분류)
- 범례 및 AUC 값
- 그리드 배경

**코드로 구현 가능**: sklearn.metrics.roc_curve()

---

### 3️⃣ **Precision-Recall Curve**
```
특징:
- Y축: Precision (정밀도)
- X축: Recall (재현율)
- 우상단에 가까울수록 우수

용도:
- 클래스 불균형 데이터셋에서 효과적
- 모델의 Precision-Recall 트레이드오프
```

**시각화 요소**:
- 3개 모델 곡선
- 임계값 표시
- 평균 정밀도 점수
- 색상 구분

---

### 4️⃣ **Feature Importance Bar Chart (특성 중요도)**
```
XGBoost 모델에서 추출 가능

특성 중요도 순서:
1. ratio (신용카드 사용률)    ██████████████░ 35.6%
2. income (월 소득)           ████████░░░░░░░ 24.5%
3. prop (부동산 비율)         █████░░░░░░░░░░ 14.3%
4. depen (부양가족 수)        ███░░░░░░░░░░░░ 11.5%
5. age (나이)                ██░░░░░░░░░░░░░ 9.8%
```

**시각화 요소**:
- 수평 바 차트
- 색상 그래디언트
- 백분율 값 표시
- 내림차순 정렬

**코드로 구현 가능**: XGBoost의 feature_importances_

---

### 5️⃣ **Model Performance Radar Chart (레이더 차트)**
```
5개 모든 메트릭을 한눈에 비교

        Accuracy
           /\
          /  \
    Precision─ Recall
        /      \
   ROC-AUC─ F1-Score

3개 모델을 다각형으로 비교
```

**시각화 요소**:
- 5개 축 (Accuracy, Precision, Recall, F1, ROC-AUC)
- 3개 모델 다각형 (투명도 있음)
- 범례 색상 구분
- 0-1 스케일

**코드로 구현 가능**: matplotlib.pyplot.subplot(polar=True)

---

### 6️⃣ **Calibration Curve (캘리브레이션 곡선)**
```
모델 신뢰도 평가

Y축: 실제 양성률
X축: 예측 확률

완벽한 모델: 대각선
과신감 모델: 곡선이 아래쪽
과소신감 모델: 곡선이 위쪽
```

**시각화 요소**:
- 3개 모델 곡선
- 완벽한 모델 대각선
- 히스토그램 (하단)
- Brier Score 표시

---

### 7️⃣ **Learning Curve (학습 곡선)**
```
훈련 샘플 수에 따른 성능 변화

성능
  │     ╱─────── Test Score
  │    ╱
  │   ╱─────────── Train Score
  │  ╱
  └─────────────────────→ 훈련 샘플 수

과적합/과소적합 진단
```

**시각화 요소**:
- 훈련 곡선 (파란색)
- 검증 곡선 (주황색)
- 신뢰도 영역 (음영)
- 교차점 표시

---

### 8️⃣ **Validation Curve (검증 곡선)**
```
하이퍼파라미터에 따른 성능 변화

예: max_depth 변화에 따른 성능

성능
  │       ╱╲
  │      ╱  ╲      최적점
  │     ╱    ╲
  │    ╱      ╲
  └──────────────────→ max_depth
  과소적합   최적   과적합
```

**시각화 요소**:
- 훈련 성능 곡선
- 검증 성능 곡선
- 최적값 표시
- 신뢰도 영역

---

### 9️⃣ **Class Distribution - Violin Plot**
```
클래스별 특성 분포를 바이올린으로 표현

특성: ratio (신용카드 사용률)

빈도 │     ╱╲          ╱╲
     │    ╱  ╲        ╱  ╲
     │   ╱    ╲      ╱    ╲
     │  ╱      ╲    ╱      ╲
     └──────────────────────→ 값
         Class 0    Class 1

시각적 매력도: ⭐⭐⭐⭐⭐
```

**시각화 요소**:
- 분포의 대칭성 표현
- 중앙값 마크
- 사분위수 표시
- 색상 그래디언트

**코드로 구현 가능**: seaborn.violinplot()

---

### 🔟 **Correlation Heatmap (상관계수 히트맵)**
```
모든 특성 간 상관관계 시각화

      prop  age ratio income depen loan
prop  1.00 -0.24  0.16  0.11  0.08 -0.14
age  -0.24  1.00  0.05 -0.11 -0.19  0.10
ratio 0.16  0.05  1.00 -0.13  0.11  0.36
income 0.11 -0.11 -0.13  1.00  0.16  0.25
depen 0.08 -0.19  0.11  0.16  1.00  0.12
loan -0.14  0.10  0.36  0.25  0.12  1.00

색상: 파란색(음의 상관) → 흰색(0) → 빨간색(양의 상관)
```

**시각화 요소**:
- 색상 맵 (cool-warm)
- 상관계수 값 표시
- 대칭 행렬
- 범례

---

### 1️⃣1️⃣ **Decision Boundary Visualization**
```
2D 평면에서 모델의 결정 경계 시각화

예: ratio vs income

income
  │ ┌─────────────┐
  │ │ Class 0     │ ╱╱╱╱  경계선
  │ │    ●   ●    │╱╱╱╱
  │ ├─────────────╱
  │ │    ◆   ◆   │
  │ │ Class 1    │
  │ └─────────────┘
  └─────────────────→ ratio
```

**시각화 요소**:
- 배경 영역 (클래스별 색상)
- 데이터 포인트 (산점도)
- 경계선
- 컬러바

---

### 1️⃣2️⃣ **Threshold Optimization Curve**
```
임계값에 따른 성능 메트릭 변화

성능
  │     Precision
  │    ╱─────────╲
  │   ╱           ╲
  │  ╱  F1-Score   ╲
  │ ╱────────────────╲
  │ Recall ╱─────────╲
  │         ╱       ╲
  └─────────────────────→ 임계값
      0.3  0.5  0.7

최적 임계값 표시
```

**시각화 요소**:
- Precision 곡선 (빨강)
- Recall 곡선 (파랑)
- F1-Score 곡선 (초록)
- 최적값 수직선

---

## 🛠️ **고급 시각화 구현 방법**

### **필요한 라이브러리**
```bash
pip install matplotlib seaborn scikit-learn xgboost pandas numpy
```

### **추천하는 조합**
| 시각화 | 우선순위 | 난이도 | 영향도 |
|--------|---------|--------|--------|
| Confusion Matrix | ⭐⭐⭐⭐⭐ | 쉬움 | 매우 높음 |
| ROC Curve | ⭐⭐⭐⭐⭐ | 중간 | 매우 높음 |
| Feature Importance | ⭐⭐⭐⭐ | 쉬움 | 높음 |
| Radar Chart | ⭐⭐⭐⭐ | 중간 | 높음 |
| Correlation Heatmap | ⭐⭐⭐⭐ | 쉬움 | 높음 |
| Precision-Recall | ⭐⭐⭐ | 중간 | 중간 |
| Violin Plot | ⭐⭐⭐ | 중간 | 중간 |
| Calibration Curve | ⭐⭐ | 어려움 | 낮음 |

---

## 📈 **발표에 추천하는 순서**

1. **데이터 개요 섹션**
   - Confusion Matrix ❌ (모델 후에)
   - Correlation Heatmap ✓
   - Class Distribution (Violin Plot) ✓

2. **모델 성능 섹션**
   - Confusion Matrix (XGBoost) ✓
   - ROC Curve (3개 모델 비교) ✓
   - Feature Importance (Top 5) ✓

3. **모델 비교 섹션**
   - Radar Chart (모든 메트릭) ✓
   - Precision-Recall Curve ✓

4. **고급 분석 섹션**
   - Threshold Optimization ✓
   - Calibration Curve ✓

---

## 💡 **시각화 품질 향상 팁**

### **색상 선택**
✓ 색맹 친화적 팔레트 사용
✓ 대비 충분한 색상 (회색 배경에서도 보이게)
✓ 일관된 색상 스키마 (파란색=모델1, 주황색=모델2, 초록색=모델3)

### **글꼴 및 크기**
✓ 모든 차트 글꼰: Arial 또는 DejaVu
✓ 제목: 14-16pt
✓ 축 레이블: 11-12pt
✓ 범례: 10-11pt

### **레이아웃**
✓ 여백 충분히 확보
✓ 그리드선 (alpha=0.3으로 연하게)
✓ 하나의 그림에 최대 3개 서브플롯

### **정보 표시**
✓ 중요한 값을 직접 표시 (예: 정확도)
✓ 범례 항상 포함
✓ 축 레이블 및 단위 명확히

---

## 🎯 **추천 조합 (시간 제한 있을 때)**

### **최소 패키지 (15분)**
1. Confusion Matrix (2분)
2. ROC Curve (3개 모델) (3분)
3. Feature Importance (2분)

### **표준 패키지 (30분)**
1. Confusion Matrix (2분)
2. ROC Curve (3개 모델) (3분)
3. Feature Importance (2분)
4. Radar Chart (모든 메트릭) (5분)
5. Correlation Heatmap (3분)
6. Class Distribution Violin (3분)

### **프리미엄 패키지 (60분)**
1. Confusion Matrix (2분)
2. ROC & Precision-Recall 곡선 (5분)
3. Feature Importance (2분)
4. Radar Chart (5분)
5. Correlation Heatmap (3분)
6. Violin Plot (Class Distribution) (3분)
7. Threshold Optimization (5분)
8. Decision Boundary (2D) (5분)
9. Learning Curve (시뮬레이션) (3분)

---

**어떤 그래프부터 만들고 싶으신가요? 제가 코드를 작성해드릴 수 있습니다!**
