import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LogisticRegression
import pandas as pd

print("[START] Logistic Regression Visualization")

# 1. 데이터 로드
print("[LOAD] Loading data...")
train_data = pd.read_csv('cs-training-engineered.csv')
testX = train_data.iloc[:5000, :-1].values  # 처음 5000개만 (빠른 연산)
testY = train_data.iloc[:5000, -1].values

print(f"  Samples: {len(testX):,}")
print(f"  Features: {testX.shape[1]}")

# 2. 로지스틱 회귀 모델 학습
print("[TRAIN] Training Logistic Regression model...")
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(testX, testY)
print("  ✅ Model trained")

# 3. 로지스틱 함수 정의
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 4. 그래프 생성
print("[VIZ] Creating Logistic Regression visualizations...")
fig = plt.figure(figsize=(14, 10))

# (a) 기본 로지스틱 함수
print("  [VIZ-1] Creating basic sigmoid function...")
ax1 = plt.subplot(2, 2, 1)
x = np.linspace(-6, 6, 100)
y = sigmoid(x)
ax1.plot(x, y, 'b-', linewidth=3)
ax1.fill_between(x, 0, y, alpha=0.2, color='blue')
ax1.set_xlabel('x', fontsize=12, fontweight='bold')
ax1.set_ylabel('E(y)', fontsize=12, fontweight='bold')
ax1.set_title('(a) Logistic Function: $E(y) = \\frac{1}{1+e^{-x}}$', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0, 1.1])

# (b) 확률밀도함수 (미분)
print("  [VIZ-2] Creating probability density function...")
ax2 = plt.subplot(2, 2, 2)
y_prime = sigmoid(x) * (1 - sigmoid(x))
ax2.plot(x, y_prime, 'r-', linewidth=3)
ax2.fill_between(x, 0, y_prime, alpha=0.2, color='red')
ax2.set_xlabel('x', fontsize=12, fontweight='bold')
ax2.set_ylabel("E'(y)", fontsize=12, fontweight='bold')
ax2.set_title('(b) Derivative (Probability Density)', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)

# (c) 2D 특성의 3D 표면
print("  [VIZ-3] Creating 2D logistic surface...")
ax3 = plt.subplot(2, 2, 3, projection='3d')
x1_range = np.linspace(-1, 1, 30)
x2_range = np.linspace(-1, 1, 30)
X1, X2 = np.meshgrid(x1_range, x2_range)

# 2개 특성만 사용하여 3D 표면 생성
Z = sigmoid(lr_model.intercept_[0] + 
            lr_model.coef_[0, 0]*X1 + 
            lr_model.coef_[0, 1]*X2)

surface1 = ax3.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.8, edgecolor='none')
ax3.set_xlabel('x₁', fontsize=10, fontweight='bold')
ax3.set_ylabel('x₂', fontsize=10, fontweight='bold')
ax3.set_zlabel('E(y)', fontsize=10, fontweight='bold')
ax3.set_title('(c) 2D Logistic Surface', fontsize=12, fontweight='bold')
ax3.view_init(elev=25, azim=45)

# (d) 다른 각도의 3D 표면
print("  [VIZ-4] Creating 3D logistic surface (rotated)...")
ax4 = plt.subplot(2, 2, 4, projection='3d')
x1_range = np.linspace(-1, 1, 25)
x2_range = np.linspace(-1, 1, 25)
X1, X2 = np.meshgrid(x1_range, x2_range)

Z = sigmoid(lr_model.intercept_[0] + 
            lr_model.coef_[0, 0]*X1 + 
            lr_model.coef_[0, 1]*X2)

surface2 = ax4.plot_surface(X1, X2, Z, cmap='plasma', alpha=0.8, edgecolor='none')
ax4.set_xlabel('x₁', fontsize=10, fontweight='bold')
ax4.set_ylabel('x₂', fontsize=10, fontweight='bold')
ax4.set_zlabel('E(y)', fontsize=10, fontweight='bold')
ax4.set_title('(d) 3D Logistic Surface (Rotated)', fontsize=12, fontweight='bold')
ax4.view_init(elev=20, azim=120)

plt.tight_layout()
plt.savefig('14_logistic_regression_visualization.png', dpi=300, bbox_inches='tight')
print("✅ 14_logistic_regression_visualization.png saved")

# 모델 정보 출력
print("\n📊 Logistic Regression Model Info:")
print(f"  Intercept: {lr_model.intercept_[0]:.6f}")
print(f"  Coefficients: {lr_model.coef_[0]}")
print(f"  Classes: {lr_model.classes_}")

print("\n[COMPLETE] Logistic Regression Visualization Complete")
plt.close()
