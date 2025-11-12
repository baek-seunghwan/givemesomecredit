import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

print("[START] SVM Decision Boundary Visualization")

# 1. 데이터 로드
print("[LOAD] Loading data...")
train_data = pd.read_csv('cs-training-engineered.csv')
test_data = pd.read_csv('cs-test-engineered.csv')

print(f"  Training samples: {len(train_data):,}")
print(f"  Test samples: {len(test_data):,}")

# 2. X, y 분리 (마지막 컬럼이 target)
trainX = train_data.iloc[:10000, :-1].values  # 처음 10000개만 (빠른 학습)
trainY = train_data.iloc[:10000, -1].values
testX = test_data.iloc[:, :-1].values
testY = test_data.iloc[:, -1].values

print(f"  Features: {trainX.shape[1]}")

# 3. 정규화 (SVM은 정규화 필수)
print("[NORMALIZE] Normalizing data...")
scaler = MinMaxScaler(feature_range=(0, 2))
trainX_scaled = scaler.fit_transform(trainX)
testX_scaled = scaler.transform(testX)

# 5. 2개 특성만 선택해서 시각화 (처음 2개 특성)
# 먼저 2개 특성으로만 데이터 준비
X_2d_train = trainX_scaled[:5000, :2]  # 처음 5000개, 처음 2개 특성
y_2d = trainY[:5000]
X_2d_test = testX_scaled[:, :2]
y_2d_test = testY

# 4. SVM 모델 생성 (2개 특성만 사용)
print("[TRAIN] Training SVM model...")
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_model.fit(X_2d_train, y_2d)
print(f"  ✅ SVM trained with {len(svm_model.support_vectors_)} support vectors")

# 변수명 통일
X_2d = X_2d_train
y = y_2d

# 6. 메시 그리드 생성
print("[COMPUTE] Computing decision boundary...")
h = 0.02  # 스텝 사이즈
x_min, x_max = X_2d[:, 0].min() - 0.1, X_2d[:, 0].max() + 0.1
y_min, y_max = X_2d[:, 1].min() - 0.1, X_2d[:, 1].max() + 0.1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# 7. 모델로 예측
print("[PREDICT] Making predictions on grid...")
Z = svm_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 8. 그래프 그리기
print("[VIZ] Creating visualization...")
fig = plt.figure(figsize=(15, 6))

# 왼쪽: 결정 경계
print("  [VIZ-1] Creating decision boundary plot...")
ax1 = plt.subplot(1, 2, 1)

# 배경 색상
contourf = ax1.contourf(xx, yy, Z, alpha=0.3, cmap='viridis', levels=2)

# 데이터 포인트
scatter_0 = ax1.scatter(X_2d[y==0, 0], X_2d[y==0, 1], 
                        c='white', marker='o', edgecolors='gray', s=50, 
                        alpha=0.7, label='Class 0 (No Loan)')
scatter_1 = ax1.scatter(X_2d[y==1, 0], X_2d[y==1, 1], 
                        c='black', marker='o', s=50,
                        alpha=0.7, label='Class 1 (Loan)')

# Support Vector 표시
if hasattr(svm_model, 'support_vectors_'):
    sv_2d = svm_model.support_vectors_[:, :2]
    ax1.scatter(sv_2d[:, 0], sv_2d[:, 1], 
               c='red', marker='x', s=100, linewidths=2, 
               label='Support Vectors')

# 결정 경계선
contour = ax1.contour(xx, yy, Z, colors='red', levels=[0.5], linewidths=2.5)
ax1.clabel(contour, inline=True, fontsize=10)

ax1.set_xlabel('Feature 1 (Normalized)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Feature 2 (Normalized)', fontsize=12, fontweight='bold')
ax1.set_title('SVM Decision Boundary\n(RBF Kernel)', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10, loc='best')
ax1.grid(True, alpha=0.2)

# 오른쪽: 모델 정보 및 성능
print("  [VIZ-2] Creating model information panel...")
ax2 = plt.subplot(1, 2, 2)
ax2.axis('off')

# 모델 성능 평가
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
y_pred = svm_model.predict(X_2d_test)
accuracy = accuracy_score(y_2d_test, y_pred)
precision = precision_score(y_2d_test, y_pred)
recall = recall_score(y_2d_test, y_pred)
f1 = f1_score(y_2d_test, y_pred)

info_text = f"""
SVM MODEL INFORMATION

Kernel: RBF (Radial Basis Function)
C Parameter: 1.0
Gamma: scale

Training Data:
  Samples: {len(X_2d):,}
  Features: 2 (for visualization)
  Total Features: {trainX.shape[1]}

Support Vectors:
  Count: {len(svm_model.support_vectors_)}
  Ratio: {len(svm_model.support_vectors_)/len(X_2d)*100:.2f}%

Model Performance:
  Accuracy: {accuracy:.4f}
  Precision: {precision:.4f}
  Recall: {recall:.4f}
  F1-Score: {f1:.4f}

Normalization:
  Method: MinMaxScaler
  Range: [0, 2.0]
  Features Scaled: All {trainX.shape[1]}

Advantages:
  ✓ Non-linear decision boundary
  ✓ Effective with normalized data
  ✓ Good generalization
  ✓ Handles complex patterns
"""

ax2.text(0.05, 0.95, info_text, transform=ax2.transAxes, 
         fontsize=10, verticalalignment='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7, pad=1))

plt.tight_layout()
plt.savefig('17_svm_decision_boundary.png', dpi=300, bbox_inches='tight')
print("✅ 17_svm_decision_boundary.png saved")

# 추가 보너스: 3개 커널 비교
print("\n[VIZ-BONUS] Creating kernel comparison visualization...")
fig2 = plt.figure(figsize=(15, 5))

kernels = ['linear', 'rbf', 'poly']
h_small = 0.05  # 더 세밀한 그리드

for idx, kernel in enumerate(kernels):
    print(f"  Training SVM with {kernel} kernel...")
    
    # 메시 그리드 생성 (더 작은 영역)
    x_min_s, x_max_s = 0, 2
    y_min_s, y_max_s = 0, 2
    xx_s, yy_s = np.meshgrid(np.arange(x_min_s, x_max_s, h_small),
                             np.arange(y_min_s, y_max_s, h_small))
    
    # 커널별 SVM 모델
    svm_k = SVC(kernel=kernel, C=1.0, gamma='scale')
    svm_k.fit(X_2d, y)
    
    # 예측
    Z_k = svm_k.predict(np.c_[xx_s.ravel(), yy_s.ravel()])
    Z_k = Z_k.reshape(xx_s.shape)
    
    # 플로팅
    ax = plt.subplot(1, 3, idx + 1)
    ax.contourf(xx_s, yy_s, Z_k, alpha=0.3, cmap='viridis', levels=2)
    ax.scatter(X_2d[y==0, 0], X_2d[y==0, 1], c='white', marker='o', 
              edgecolors='gray', s=30, alpha=0.6)
    ax.scatter(X_2d[y==1, 0], X_2d[y==1, 1], c='black', marker='o', s=30, alpha=0.6)
    ax.contour(xx_s, yy_s, Z_k, colors='red', levels=[0.5], linewidths=2)
    
    ax.set_xlabel('Feature 1', fontsize=11, fontweight='bold')
    ax.set_ylabel('Feature 2', fontsize=11, fontweight='bold')
    ax.set_title(f'SVM Decision Boundary\n({kernel.upper()} Kernel)', 
                fontsize=12, fontweight='bold')
    ax.set_xlim([x_min_s, x_max_s])
    ax.set_ylim([y_min_s, y_max_s])
    ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('18_svm_kernel_comparison.png', dpi=300, bbox_inches='tight')
print("✅ 18_svm_kernel_comparison.png saved (BONUS)")

print("\n[COMPLETE] SVM Decision Boundary Visualization Complete")
print("\n📊 Summary:")
print(f"  Generated 2 visualizations (17, 18)")
print(f"  SVM Test Accuracy: {accuracy:.4f}")
print(f"  SVM Test F1-Score: {f1:.4f}")
