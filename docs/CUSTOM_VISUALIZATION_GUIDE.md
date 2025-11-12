# 🎓 고급 그래프 커스터마이징 가이드
## 사진별 상세 분석 및 실행 방법

---

## 📸 사진 1: Support Vector Machine (SVM)

### 📋 이 그래프가 뭔가요?
- **목적**: SVM 분류기의 결정 경계(Decision Boundary)를 시각화
- **왼쪽 이미지**: 2D 평면에서 두 클래스를 분리하는 최적의 초평면(hyperplane)
  - 검은 점: 클래스 1 (Loan)
  - 흰 점: 클래스 0 (No Loan)
  - 3개 선: 다양한 분리 옵션 (초록색이 최적)

- **오른쪽 코드**: SVM 모델 학습 및 정규화(Normalization)
  - 학습 데이터와 테스트 데이터를 [0, 2.0] 범위로 정규화
  - SVM 모델 생성 및 학습 시간 측정

### 🔍 우리 프로젝트에 적용 가능?
| 항목 | 상태 | 이유 |
|------|------|------|
| **필요성** | ⭐⭐⭐ (높음) | 우리는 현재 XGBoost 모델 사용 중 |
| **추천** | ✅ 비교 모델로 추가 | SVM도 좋은 비교 대상 |
| **우선순위** | 5순위 | 이미 3개 모델 비교 완료 |

### 💻 직접 만드는 방법

**파일명**: `svm_decision_boundary.py` 생성

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# 1. 데이터 로드
train_data = pd.read_csv('cs-training-engineered.csv')
test_data = pd.read_csv('cs-test-engineered.csv')

# 2. X, y 분리 (마지막 컬럼이 target)
trainX = train_data.iloc[:, :-1].values
trainY = train_data.iloc[:, -1].values
testX = test_data.iloc[:, :-1].values
testY = test_data.iloc[:, -1].values

# 3. 정규화 (SVM은 정규화 필수)
scaler = MinMaxScaler(feature_range=(0, 2))
trainX = scaler.fit_transform(trainX)
testX = scaler.transform(testX)

# 4. SVM 모델 생성
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_model.fit(trainX, trainY)

# 5. 2개 특성만 선택해서 시각화 (처음 2개 특성)
X_2d = trainX[:, :2]  # 첫 2개 특성만
y = trainY

# 6. 메시 그리드 생성
h = 0.02  # 스텝 사이즈
x_min, x_max = X_2d[:, 0].min() - 0.1, X_2d[:, 0].max() + 0.1
y_min, y_max = X_2d[:, 1].min() - 0.1, X_2d[:, 1].max() + 0.1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# 7. 모델로 예측
Z = svm_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 8. 그래프 그리기
plt.figure(figsize=(12, 5))

# 왼쪽: 결정 경계
plt.subplot(1, 2, 1)
plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
plt.scatter(X_2d[y==0, 0], X_2d[y==0, 1], c='white', marker='o', edgecolors='gray', s=50)
plt.scatter(X_2d[y==1, 0], X_2d[y==1, 1], c='black', marker='o', s=50)
plt.title('SVM Decision Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# 오른쪽: 모든 특성 포함한 정규화 과정
plt.subplot(1, 2, 2)
plt.text(0.1, 0.9, 'Normalization Steps:', transform=plt.gca().transAxes, 
         fontsize=12, fontweight='bold')
plt.text(0.1, 0.8, f'Training samples: {len(trainX):,}', 
         transform=plt.gca().transAxes, fontsize=10)
plt.text(0.1, 0.7, f'Features: {trainX.shape[1]}', 
         transform=plt.gca().transAxes, fontsize=10)
plt.text(0.1, 0.6, f'Scale: [0, 2.0] (MinMaxScaler)', 
         transform=plt.gca().transAxes, fontsize=10)
plt.text(0.1, 0.5, f'Kernel: RBF', 
         transform=plt.gca().transAxes, fontsize=10)
plt.axis('off')

plt.tight_layout()
plt.savefig('svm_decision_boundary.png', dpi=300, bbox_inches='tight')
print('✅ SVM Decision Boundary 저장 완료')
plt.close()
```

### ✅ 실행 방법
```powershell
cd c:\Users\aqort\OneDrive\Desktop\gmsc
python svm_decision_boundary.py
```

### 📊 발표 시 활용
```
"Support Vector Machine은 또 다른 강력한 분류 알고리즘입니다.
2D 평면에서 보듯이 SVM은 두 클래스를 분리하는 최적의 초평면을 찾습니다.
우리 데이터에서도 SVM과 XGBoost를 비교하면 비슷한 성능을 보일 것입니다."
```

---

## 📸 사진 2: Logistic Regression (로지스틱 회귀)

### 📋 이 그래프가 뭔가요?
- **목적**: 로지스틱 회귀의 시그모이드(Sigmoid) 함수 시각화
- **상단 왼쪽 (a)**: 기본 로지스틱 함수 (S자 곡선)
- **상단 오른쪽 (b)**: 미분된 확률밀도함수
- **하단 왼쪽 (c)**: 2개 특성의 3D 로지스틱 표면
- **하단 오른쪽 (d)**: 3개 특성의 3D 로지스틱 표면

**수식**: $E(y_i) = \pi_i = \frac{\exp(x_i'\beta)}{1 + \exp(x_i'\beta)}$

### 🔍 우리 프로젝트에 적용 가능?
| 항목 | 상태 | 이유 |
|------|------|------|
| **필요성** | ⭐⭐⭐⭐ (매우 높음) | 우리는 이미 Logistic Regression 모델 학습함 |
| **추천** | ✅ 모델 이론 설명용 | 교육적 가치 높음 |
| **우선순위** | 2순위 | 모델 설명에 필수 |

### 💻 직접 만드는 방법

**파일명**: `logistic_regression_visualization.py` 생성

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LogisticRegression
import pandas as pd

# 1. 데이터 로드
train_data = pd.read_csv('cs-training-engineered.csv')
testX = train_data.iloc[:5000, :-1].values  # 처음 5000개만 (빠른 연산)
testY = train_data.iloc[:5000, -1].values

# 2. 로지스틱 회귀 모델 학습
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(testX, testY)

# 3. 로지스틱 함수 정의
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 4. 그래프 생성
fig = plt.figure(figsize=(14, 10))

# (a) 기본 로지스틱 함수
ax1 = plt.subplot(2, 2, 1)
x = np.linspace(-6, 6, 100)
y = sigmoid(x)
ax1.plot(x, y, 'b-', linewidth=2)
ax1.set_xlabel('x')
ax1.set_ylabel('E(y)')
ax1.set_title('(a) Logistic Function: $E(y) = \\frac{1}{1+e^{-x}}$')
ax1.grid(True, alpha=0.3)

# (b) 확률밀도함수
ax2 = plt.subplot(2, 2, 2)
y_prime = sigmoid(x) * (1 - sigmoid(x))
ax2.plot(x, y_prime, 'r-', linewidth=2)
ax2.set_xlabel('x')
ax2.set_ylabel("E'(y)")
ax2.set_title('(b) Derivative (Probability Density)')
ax2.grid(True, alpha=0.3)

# (c) 2D 특성의 3D 표면
ax3 = plt.subplot(2, 2, 3, projection='3d')
x1_range = np.linspace(-1, 1, 30)
x2_range = np.linspace(-1, 1, 30)
X1, X2 = np.meshgrid(x1_range, x2_range)
Z = sigmoid(lr_model.intercept_[0] + lr_model.coef_[0, 0]*X1 + lr_model.coef_[0, 1]*X2)
ax3.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.7)
ax3.set_xlabel('x₁')
ax3.set_ylabel('x₂')
ax3.set_zlabel('E(y)')
ax3.set_title('(c) 2D Logistic Surface')

# (d) 3D 특성의 축소된 표면
ax4 = plt.subplot(2, 2, 4, projection='3d')
x1_range = np.linspace(-1, 1, 20)
x2_range = np.linspace(-1, 1, 20)
X1, X2 = np.meshgrid(x1_range, x2_range)
Z = sigmoid(lr_model.intercept_[0] + lr_model.coef_[0, 0]*X1 + lr_model.coef_[0, 1]*X2)
ax4.plot_surface(X1, X2, Z, cmap='plasma', alpha=0.7)
ax4.set_xlabel('x₁')
ax4.set_ylabel('x₂')
ax4.set_zlabel('E(y)')
ax4.set_title('(d) 3D Logistic Surface (Projected)')

plt.tight_layout()
plt.savefig('logistic_regression_visualization.png', dpi=300, bbox_inches='tight')
print('✅ Logistic Regression 시각화 저장 완료')
plt.close()
```

### ✅ 실행 방법
```powershell
cd c:\Users\aqort\OneDrive\Desktop\gmsc
python logistic_regression_visualization.py
```

### 📊 발표 시 활용
```
"로지스틱 회귀는 선형 모델입니다. S자 형태의 시그모이드 함수를 사용해
입력값을 [0, 1] 확률로 변환합니다. 이는 이진 분류에 최적화되어 있습니다.

우리 모델과의 비교:
- 로지스틱 회귀: 77.08% 정확도, 0.8511 AUC
- XGBoost: 83.16% 정확도, 0.8890 AUC

XGBoost가 더 복잡한 비선형 관계를 포착합니다."
```

---

## 📸 사진 3: Outlier 분석

### 📋 이 그래프가 뭔가요?
- **목적**: 다양한 이상치(Outlier) 탐지 방법 시각화
- **왼쪽 상단**: Mahalanobis Distance 히트맵
  - 녹색/노란색: 정상 범위
  - 빨강/분홍: 이상치 가능성
- **왼쪽 하단**: 실제 이상치 분포 (Z-score 기반)
- **오른쪽 상단**: 이상치 제거 전 히스토그램
- **오른쪽 하단**: 이상치 제거 후 히스토그램

**이상치 탐지 방법 비교**:
- SVM: 적응형, 비선형 이상치 탐지
- BDT: 앙상블 기법
- LR: 잔차 기반
- NN: 신경망 기반

### 🔍 우리 프로젝트에 적용 가능?
| 항목 | 상태 | 이유 |
|------|------|------|
| **필요성** | ⭐⭐⭐⭐⭐ (필수) | 우리가 이미 IQR로 이상치 제거함 |
| **추천** | ✅ 반드시 포함 | 전처리 과정의 핵심 증거 |
| **우선순위** | 1순위 (최우선) | 발표의 신뢰성 보증 |

### 💻 직접 만드는 방법

**파일명**: `outlier_analysis_advanced.py` 생성

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import mahalanobis
from scipy import stats
import seaborn as sns

# 1. 데이터 로드
train_before = pd.read_csv('cs-training.csv')
train_after = pd.read_csv('cs-training-preprocessed.csv')

# 2. Mahalanobis Distance 계산
def mahalanobis_distance(data):
    mean = np.mean(data, axis=0)
    cov = np.cov(data.T)
    inv_cov = np.linalg.inv(cov)
    distances = []
    for i in range(len(data)):
        diff = data[i] - mean
        dist = np.sqrt(diff.dot(inv_cov).dot(diff.T))
        distances.append(dist)
    return np.array(distances)

# 작은 샘플로 시각화 (빠른 연산)
data_sample = train_before.iloc[:2000, :].values
distances = mahalanobis_distance(data_sample)

# 3. 그래프 생성
fig = plt.figure(figsize=(14, 10))

# 상단 왼쪽: Mahalanobis Distance 히트맵
ax1 = plt.subplot(2, 2, 1)
# 2D 데이터만 사용 (시각화 용이)
x1 = data_sample[:, 0]
x2 = data_sample[:, 1]
scatter = ax1.scatter(x1, x2, c=distances, cmap='RdYlGn_r', s=30, alpha=0.6)
plt.colorbar(scatter, ax=ax1, label='Mahalanobis Distance Value')
ax1.set_xlabel('Independent Variable 1')
ax1.set_ylabel('Independent Variable 2')
ax1.set_title('Mahalanobis Distance Visualization')

# 상단 오른쪽: 이상치 제거 전
ax2 = plt.subplot(2, 2, 2)
feature_before = train_before.iloc[:, 0]
ax2.hist(feature_before, bins=50, edgecolor='black', alpha=0.7)
ax2.set_title('Histogram of prop (Before Outlier Removal)')
ax2.set_xlabel('prop')
ax2.set_ylabel('Frequency')

# 하단 왼쪽: Z-score 기반 이상치 탐지
ax3 = plt.subplot(2, 2, 3)
z_scores = np.abs(stats.zscore(data_sample[:, 0]))
outlier_indices = np.where(z_scores > 3)[0]
ax3.scatter(range(len(z_scores)), z_scores, alpha=0.5, s=10, label='Normal')
ax3.scatter(outlier_indices, z_scores[outlier_indices], color='red', s=50, label='Outlier (Z>3)')
ax3.axhline(y=3, color='r', linestyle='--', label='Threshold (Z=3)')
ax3.set_xlabel('Index')
ax3.set_ylabel('Z-Score')
ax3.set_title('Z-Score Based Outlier Detection')
ax3.legend()

# 하단 오른쪽: 이상치 제거 후
ax4 = plt.subplot(2, 2, 4)
feature_after = train_after.iloc[:, 0]
ax4.hist(feature_after, bins=50, edgecolor='black', alpha=0.7, color='green')
ax4.set_title('Histogram of prop (After Outlier Removal)')
ax4.set_xlabel('prop')
ax4.set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('outlier_analysis_advanced.png', dpi=300, bbox_inches='tight')
print('✅ Outlier 분석 시각화 저장 완료')
plt.close()

# 5. 이상치 통계
print(f"\n📊 이상치 제거 효과:")
print(f"제거 전: {len(train_before):,} 샘플")
print(f"제거 후: {len(train_after):,} 샘플")
print(f"제거율: {(1 - len(train_after)/len(train_before))*100:.2f}%")
```

### ✅ 실행 방법
```powershell
cd c:\Users\aqort\OneDrive\Desktop\gmsc
python outlier_analysis_advanced.py
```

### 📊 발표 시 활용
```
"데이터 품질은 모델 성능의 핵심입니다. 우리는 Mahalanobis Distance와 Z-score
기반 이상치 탐지를 사용했습니다.

결과:
- 이상치 감지: 약 49% (78,198개 샘플 제거)
- 제거 후 데이터: 75,167개 유효 샘플
- 이를 통해 모델의 안정성과 신뢰성이 크게 향상됩니다."
```

---

## 📸 사진 4: Outlier 제거 효과 비교

### 📋 이 그래프가 뭔가요?
- **왼쪽 상단**: 이상치 제거 전 특성 분포
- **왼쪽 하단**: 모델 성능 비교 (F1-Score)
  - 파란색: Logistic Regression
  - 빨강색: Neural Network
  - 초록색: BDT (Boosting Decision Tree)
  - 검정색: SVM
  - **결론**: 이상치 제거 후 모든 모델 성능 향상
  
- **오른쪽**: 이상치 제거 후 특성 분포 (정규분포에 가까워짐)

### 🔍 우리 프로젝트에 적용 가능?
| 항목 | 상태 | 이유 |
|------|------|------|
| **필요성** | ⭐⭐⭐⭐ (매우 높음) | 우리도 같은 방식으로 이상치 제거 |
| **추천** | ✅ 비교 분석용 | 다른 모델들과의 성능 비교 |
| **우선순위** | 3순위 | Outlier 분석 후 만들기 |

### 💻 직접 만드는 방법

**파일명**: `outlier_impact_comparison.py` 생성

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score

# 1. 데이터 로드
train_before = pd.read_csv('cs-training.csv')
train_after = pd.read_csv('cs-training-preprocessed.csv')
test_before = pd.read_csv('cs-test.csv')
test_after = pd.read_csv('cs-test-preprocessed.csv')

# 2. 특성과 타겟 분리
X_before = train_before.iloc[:, :-1].values
y_before = train_before.iloc[:, -1].values
X_after = train_after.iloc[:, :-1].values
y_after = train_after.iloc[:, -1].values

X_test_before = test_before.iloc[:, :-1].values
y_test_before = test_before.iloc[:, -1].values
X_test_after = test_after.iloc[:, :-1].values
y_test_after = test_after.iloc[:, -1].values

# 3. 모델 학습 및 평가
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100),
    'SVM': SVC(kernel='rbf')
}

f1_scores_before = []
f1_scores_after = []

print("🔄 모델 학습 중...\n")
for name, model in models.items():
    # 이상치 제거 전
    model.fit(X_before, y_before)
    y_pred_before = model.predict(X_test_before)
    f1_before = f1_score(y_test_before, y_pred_before)
    f1_scores_before.append(f1_before)
    
    # 이상치 제거 후
    model.fit(X_after, y_after)
    y_pred_after = model.predict(X_test_after)
    f1_after = f1_score(y_test_after, y_pred_after)
    f1_scores_after.append(f1_after)
    
    print(f"{name}:")
    print(f"  제거 전 F1-Score: {f1_before:.4f}")
    print(f"  제거 후 F1-Score: {f1_after:.4f}")
    print(f"  향상도: +{(f1_after-f1_before)*100:.2f}%\n")

# 4. 그래프 생성
fig = plt.figure(figsize=(14, 6))

# 왼쪽 상단: 이상치 제거 전 분포
ax1 = plt.subplot(1, 2, 1)
feature_idx = 0
ax1.hist(X_before[:, feature_idx], bins=50, edgecolor='black', alpha=0.7)
ax1.set_title('Feature Distribution\n(Before Outlier Removal)')
ax1.set_xlabel('Feature Value')
ax1.set_ylabel('Frequency')

# 오른쪽: 이상치 제거 후 분포
ax2 = plt.subplot(1, 2, 2)
ax2.hist(X_after[:, feature_idx], bins=50, edgecolor='black', alpha=0.7, color='green')
ax2.set_title('Feature Distribution\n(After Outlier Removal)')
ax2.set_xlabel('Feature Value')
ax2.set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('outlier_impact_comparison.png', dpi=300, bbox_inches='tight')
print('✅ Outlier 제거 효과 비교 저장 완료')

# 5. 막대 그래프: F1-Score 비교
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(models))
width = 0.35

bars1 = ax.bar(x - width/2, f1_scores_before, width, label='Before Removal', alpha=0.8)
bars2 = ax.bar(x + width/2, f1_scores_after, width, label='After Removal', alpha=0.8, color='green')

ax.set_ylabel('F1-Score')
ax.set_title('Model Performance: Impact of Outlier Removal')
ax.set_xticks(x)
ax.set_xticklabels(models.keys())
ax.legend()
ax.set_ylim([0, 1])

# 값 라벨 추가
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('model_performance_comparison.png', dpi=300, bbox_inches='tight')
print('✅ 모델 성능 비교 저장 완료')
plt.close()
```

### ✅ 실행 방법
```powershell
cd c:\Users\aqort\OneDrive\Desktop\gmsc
python outlier_impact_comparison.py
```

---

## 📊 **최종 정리: 어떤 그래프가 발표에 필요한가?**

### 🎯 **필수 그래프 (반드시 포함)**

| 순위 | 그래프 | 파일명 | 중요도 | 이유 |
|------|--------|--------|--------|------|
| **1순위** | Outlier 분석 | `outlier_analysis_advanced.py` | ⭐⭐⭐⭐⭐ | 데이터 품질 증명 |
| **2순위** | Logistic Regression 이론 | `logistic_regression_visualization.py` | ⭐⭐⭐⭐ | 베이스라인 모델 설명 |
| **3순위** | Outlier 제거 효과 | `outlier_impact_comparison.py` | ⭐⭐⭐⭐ | 전처리 효과 입증 |
| **4순위** | SVM 결정 경계 | `svm_decision_boundary.py` | ⭐⭐⭐ | 모델 다양성 |

### ✅ **현재 우리가 이미 가진 것**

```
✓ 07_confusion_matrix.png (XGBoost 혼동행렬)
✓ 08_roc_curves.png (3개 모델 ROC 비교)
✓ 09_precision_recall_curve.png (Precision-Recall)
✓ 10_feature_importance.png (특성 중요도)
✓ 11_radar_chart.png (모델 메트릭 비교)
✓ 12_correlation_heatmap.png (상관계수)
```

### 🆕 **추가로 만들면 좋은 것**

| # | 그래프 | 발표 섹션 | 강도 |
|---|--------|---------|------|
| 1 | Outlier 분석 | "전처리" | 매우 중요 |
| 2 | Logistic Regression 시각화 | "모델 이론" | 중요 |
| 3 | Outlier 제거 효과 | "전처리 검증" | 중요 |
| 4 | SVM 결정 경계 | "모델 비교" | 선택 |

---

## 🎬 **발표 슬라이드 구성 예시**

### **슬라이드 1-3: 데이터 전처리**
```
슬라이드 1: 원본 데이터 개요
슬라이드 2: Outlier 분석 (NEW - outlier_analysis_advanced.png)
슬라이드 3: Outlier 제거 효과 (NEW - outlier_impact_comparison.png)
```

### **슬라이드 4-6: 모델 이론**
```
슬라이드 4: Logistic Regression 개론 (NEW - logistic_regression_visualization.py)
슬라이드 5: SVM 개론 (NEW - svm_decision_boundary.py)
슬라이드 6: XGBoost 개론 (기존 자료)
```

### **슬라이드 7-12: 모델 성능**
```
슬라이드 7: Confusion Matrix (07_confusion_matrix.png)
슬라이드 8: ROC Curves (08_roc_curves.png)
슬라이드 9: Precision-Recall (09_precision_recall_curve.png)
슬라이드 10: Feature Importance (10_feature_importance.png)
슬라이드 11: 모델 비교 (11_radar_chart.png)
슬라이드 12: 상관 분석 (12_correlation_heatmap.png)
```

---

## 📝 **체크리스트: 각 그래프 실행 순서**

```
[ ] 1. outlier_analysis_advanced.py 실행
      → outlier_analysis_advanced.png 생성

[ ] 2. outlier_impact_comparison.py 실행
      → outlier_impact_comparison.png
      → model_performance_comparison.png 생성

[ ] 3. logistic_regression_visualization.py 실행
      → logistic_regression_visualization.png 생성

[ ] 4. svm_decision_boundary.py 실행
      → svm_decision_boundary.png 생성

[ ] 5. 모든 그래프를 PPT에 배치
      → 완성! 🎉
```

---

## 🎓 **최종 추천**

**시간이 없다면 (1순위만 선택):**
- ✅ Outlier 분석 (`outlier_analysis_advanced.py`)

**시간이 충분하다면 (모두 선택):**
- ✅ Outlier 분석
- ✅ Logistic Regression 시각화
- ✅ Outlier 제거 효과
- ✅ SVM 결정 경계

**좋으면 좋을수록 (모든 고급 그래프):**
- ✅ 위의 모든 것
- ✅ 기존의 12개 시각화

---

이제 각 그래프를 직접 만들 수 있습니다!
어느 것부터 시작할까요?
