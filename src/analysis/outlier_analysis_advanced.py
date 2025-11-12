import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import mahalanobis
from scipy import stats
import seaborn as sns

print("[START] Outlier Analysis Advanced Visualization")

# 1. 데이터 로드
print("[LOAD] Loading data...")
train_before = pd.read_csv('cs-training.csv')
train_after = pd.read_csv('cs-training-preprocessed.csv')

print(f"  Before: {len(train_before):,} samples")
print(f"  After: {len(train_after):,} samples")

# 2. Mahalanobis Distance 계산
print("[COMPUTE] Calculating Mahalanobis Distance...")
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
print("[VIZ-1] Creating Outlier Analysis visualization...")
fig = plt.figure(figsize=(14, 10))

# 상단 왼쪽: Mahalanobis Distance 히트맵
ax1 = plt.subplot(2, 2, 1)
x1 = data_sample[:, 0]
x2 = data_sample[:, 1]
scatter = ax1.scatter(x1, x2, c=distances, cmap='RdYlGn_r', s=30, alpha=0.6)
cbar = plt.colorbar(scatter, ax=ax1, label='Mahalanobis Distance Value')
ax1.set_xlabel('Independent Variable 1', fontsize=11, fontweight='bold')
ax1.set_ylabel('Independent Variable 2', fontsize=11, fontweight='bold')
ax1.set_title('Mahalanobis Distance Visualization', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.2)

# 상단 오른쪽: 이상치 제거 전
ax2 = plt.subplot(2, 2, 2)
feature_before = train_before.iloc[:, 0]
ax2.hist(feature_before, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
ax2.set_title('Histogram of prop\n(Before Outlier Removal)', fontsize=12, fontweight='bold')
ax2.set_xlabel('prop', fontsize=11, fontweight='bold')
ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax2.grid(True, alpha=0.2, axis='y')

# 하단 왼쪽: Z-score 기반 이상치 탐지
print("[VIZ-2] Creating Z-score based outlier detection...")
ax3 = plt.subplot(2, 2, 3)
z_scores = np.abs(stats.zscore(data_sample[:, 0]))
outlier_indices = np.where(z_scores > 3)[0]
ax3.scatter(range(len(z_scores)), z_scores, alpha=0.5, s=10, label='Normal', color='blue')
ax3.scatter(outlier_indices, z_scores[outlier_indices], color='red', s=50, label='Outlier (Z>3)', zorder=5)
ax3.axhline(y=3, color='r', linestyle='--', linewidth=2, label='Threshold (Z=3)')
ax3.set_xlabel('Index', fontsize=11, fontweight='bold')
ax3.set_ylabel('Z-Score', fontsize=11, fontweight='bold')
ax3.set_title('Z-Score Based Outlier Detection', fontsize=12, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.2)

# 하단 오른쪽: 이상치 제거 후
print("[VIZ-3] Creating post-removal histogram...")
ax4 = plt.subplot(2, 2, 4)
feature_after = train_after.iloc[:, 0]
ax4.hist(feature_after, bins=50, edgecolor='black', alpha=0.7, color='lightgreen')
ax4.set_title('Histogram of prop\n(After Outlier Removal)', fontsize=12, fontweight='bold')
ax4.set_xlabel('prop', fontsize=11, fontweight='bold')
ax4.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax4.grid(True, alpha=0.2, axis='y')

plt.tight_layout()
plt.savefig('13_outlier_analysis_advanced.png', dpi=300, bbox_inches='tight')
print("✅ 13_outlier_analysis_advanced.png saved")

# 5. 이상치 통계
print("\n📊 Outlier Removal Statistics:")
print(f"  Before: {len(train_before):,} samples")
print(f"  After: {len(train_after):,} samples")
print(f"  Removal Rate: {(1 - len(train_after)/len(train_before))*100:.2f}%")
print(f"  Removed: {len(train_before) - len(train_after):,} outliers")
print(f"\n✅ Outliers Detected (Z-score > 3): {len(outlier_indices)} in sample")

print("\n[COMPLETE] Outlier Analysis Advanced Visualization Complete")
plt.close()
