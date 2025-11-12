import csv
import matplotlib
matplotlib.use('Agg')  # 백엔드 설정
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter
import os
import sys

# 인코딩 설정
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

def load_data(filename):
    """CSV 파일 읽기"""
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        data = [list(map(float, row)) for row in reader]
    return header, data

def create_missing_map():
    """결측치 맵 생성"""
    print("[GEN] Missing Map 생성 중...")
    
    header_train, data_train = load_data('cs-training.csv')
    header_test, data_test = load_data('cs-test.csv')
    
    # 결측치 계산
    missing_train = [0] * len(header_train)
    missing_test = [0] * len(header_train)
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle('[MISSING MAP] Before Preprocessing', fontsize=16, fontweight='bold', y=0.995)
    
    # Training 데이터
    ax1 = axes[0]
    missing_pct_train = [0] * len(header_train)  # 이 데이터에서는 결측치 없음
    colors_train = ['#2ecc71' if m == 0 else '#e74c3c' for m in missing_pct_train]
    bars1 = ax1.bar(header_train, missing_pct_train, color=colors_train, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Missing %', fontsize=12, fontweight='bold')
    ax1.set_title(f'Training Data (105,000 samples)', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, 100)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.tick_params(axis='x', rotation=45)
    
    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
        else:
            ax1.text(bar.get_x() + bar.get_width()/2., 5,
                    '✓', ha='center', va='bottom', fontsize=14, color='green', fontweight='bold')
    
    # Test 데이터
    ax2 = axes[1]
    missing_pct_test = [0] * len(header_train)
    colors_test = ['#2ecc71' if m == 0 else '#e74c3c' for m in missing_pct_test]
    bars2 = ax2.bar(header_train, missing_pct_test, color=colors_test, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Missing %', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Features', fontsize=12, fontweight='bold')
    ax2.set_title(f'Test Data (45,000 samples)', fontsize=13, fontweight='bold')
    ax2.set_ylim(0, 100)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.tick_params(axis='x', rotation=45)
    
    for bar in bars2:
        height = bar.get_height()
        if height > 0:
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
        else:
            ax2.text(bar.get_x() + bar.get_width()/2., 5,
                    '✓', ha='center', va='bottom', fontsize=14, color='green', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('01_missing_map.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("[SAVE] 01_missing_map.png")
    plt.close()

def create_outlier_analysis():
    """이상치 분석 차트"""
    print("[GEN] Outlier Analysis 생성 중...")
    
    header_train, data_train = load_data('cs-training.csv')
    header_test, data_test = load_data('cs-test.csv')
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle('[OUTLIER] Detection & Removal - IQR Method', fontsize=16, fontweight='bold', y=0.995)
    
    # Training 데이터
    ax1 = axes[0]
    before_after_train = ['Before\n(105,000)', 'After\n(53,362)', 'Removed\n(51,638)']
    values_train = [105000, 53362, 51638]
    colors_train = ['#3498db', '#2ecc71', '#e74c3c']
    bars1 = ax1.bar(before_after_train, values_train, color=colors_train, edgecolor='black', linewidth=2, width=0.6)
    ax1.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
    ax1.set_title('Training Data - Outlier Removal', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, 120000)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    for i, (bar, val) in enumerate(zip(bars1, values_train)):
        height = bar.get_height()
        pct = (val / 105000) * 100 if i < 2 else (val / 105000) * 100
        ax1.text(bar.get_x() + bar.get_width()/2., height + 2000,
                f'{val:,}\n({pct:.1f}%)', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Test 데이터
    ax2 = axes[1]
    before_after_test = ['Before\n(45,000)', 'After\n(22,805)', 'Removed\n(22,195)']
    values_test = [45000, 22805, 22195]
    colors_test = ['#3498db', '#2ecc71', '#e74c3c']
    bars2 = ax2.bar(before_after_test, values_test, color=colors_test, edgecolor='black', linewidth=2, width=0.6)
    ax2.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
    ax2.set_title('Test Data - Outlier Removal', fontsize=13, fontweight='bold')
    ax2.set_ylim(0, 50000)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    for i, (bar, val) in enumerate(zip(bars2, values_test)):
        height = bar.get_height()
        pct = (val / 45000) * 100 if i < 2 else (val / 45000) * 100
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1000,
                f'{val:,}\n({pct:.1f}%)', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    plt.tight_layout()
    plt.savefig('02_outlier_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("[SAVE] 02_outlier_analysis.png")
    plt.close()

def create_normalization_comparison():
    """정규화 전후 비교"""
    print("[GEN] Normalization Comparison 생성 중...")
    
    header_train, data_train = load_data('cs-training.csv')
    header_prep, data_prep = load_data('cs-training-preprocessed.csv')
    
    # 정규화 효과를 보여주기 위해 특정 컬럼 선택 (income과 age)
    income_before = [row[5] for row in data_train]  # income column
    age_before = [row[2] for row in data_train]     # age column
    
    income_after = [row[5] for row in data_prep]    # income (normalized)
    age_after = [row[2] for row in data_prep]       # age (normalized)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('[NORMALIZATION] Effect - Before & After', fontsize=16, fontweight='bold', y=0.995)
    
    # Income Before
    ax1 = axes[0, 0]
    ax1.hist(income_before, bins=50, color='#e74c3c', alpha=0.7, edgecolor='black')
    ax1.set_title('Income - Before Normalization', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Income Value', fontweight='bold')
    ax1.set_ylabel('Frequency', fontweight='bold')
    ax1.grid(alpha=0.3, linestyle='--')
    ax1.text(0.98, 0.97, f'Range: [0, {max(income_before):.0f}]', 
            transform=ax1.transAxes, fontsize=10, verticalalignment='top',
            horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontweight='bold')
    
    # Income After
    ax2 = axes[0, 1]
    ax2.hist(income_after, bins=50, color='#2ecc71', alpha=0.7, edgecolor='black')
    ax2.set_title('Income - After Normalization', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Normalized Value', fontweight='bold')
    ax2.set_ylabel('Frequency', fontweight='bold')
    ax2.grid(alpha=0.3, linestyle='--')
    ax2.text(0.98, 0.97, f'Range: [0, 1]', 
            transform=ax2.transAxes, fontsize=10, verticalalignment='top',
            horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5),
            fontweight='bold')
    
    # Age Before
    ax3 = axes[1, 0]
    ax3.hist(age_before, bins=50, color='#e74c3c', alpha=0.7, edgecolor='black')
    ax3.set_title('Age - Before Normalization', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Age Value', fontweight='bold')
    ax3.set_ylabel('Frequency', fontweight='bold')
    ax3.grid(alpha=0.3, linestyle='--')
    ax3.text(0.98, 0.97, f'Range: [21, 96]', 
            transform=ax3.transAxes, fontsize=10, verticalalignment='top',
            horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontweight='bold')
    
    # Age After
    ax4 = axes[1, 1]
    ax4.hist(age_after, bins=50, color='#2ecc71', alpha=0.7, edgecolor='black')
    ax4.set_title('Age - After Normalization', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Normalized Value', fontweight='bold')
    ax4.set_ylabel('Frequency', fontweight='bold')
    ax4.grid(alpha=0.3, linestyle='--')
    ax4.text(0.98, 0.97, f'Range: [0, 1]', 
            transform=ax4.transAxes, fontsize=10, verticalalignment='top',
            horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5),
            fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('03_normalization_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("[SAVE] 03_normalization_comparison.png")
    plt.close()

def create_data_distribution():
    """데이터 분포 비교"""
    print("[GEN] Feature Distributions 생성 중...")
    
    header_train, data_train = load_data('cs-training-preprocessed.csv')
    
    # 주요 컬럼의 분포 확인
    features_to_plot = [1, 2, 4, 5, 6, 8]  # prop, age, ratio, income, depen, loan
    feature_names = [header_train[i] for i in features_to_plot]
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('[DISTRIBUTIONS] Feature Distributions (After Preprocessing)', fontsize=16, fontweight='bold', y=0.995)
    
    axes = axes.flatten()
    
    for idx, (col_idx, col_name) in enumerate(zip(features_to_plot, feature_names)):
        values = [row[col_idx] for row in data_train]
        
        ax = axes[idx]
        ax.hist(values, bins=50, color='#3498db', alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.set_title(f'{col_name}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Normalized Value [0-1]', fontweight='bold', fontsize=10)
        ax.set_ylabel('Frequency', fontweight='bold', fontsize=10)
        ax.grid(alpha=0.3, linestyle='--')
        
        # 통계 정보
        mean_val = sum(values) / len(values)
        median_val = sorted(values)[len(values)//2]
        ax.axvline(mean_val, color='#e74c3c', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
        ax.axvline(median_val, color='#2ecc71', linestyle='--', linewidth=2, label=f'Median: {median_val:.3f}')
        ax.legend(fontsize=9, loc='upper right')
    
    plt.tight_layout()
    plt.savefig('04_feature_distributions.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("[SAVE] 04_feature_distributions.png")
    plt.close()

def create_pipeline_diagram():
    """전처리 파이프라인 다이어그램 (텍스트 기반)"""
    print("[GEN] Pipeline Diagram 생성 중...")
    
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')
    
    # 제목
    ax.text(0.5, 0.95, '[PIPELINE] Data Preprocessing Pipeline', 
            ha='center', va='top', fontsize=20, fontweight='bold',
            transform=ax.transAxes)
    
    # 단계별 박스
    stages = [
        ('1. Raw Data\n150,000 samples\n9 features', 0.1, 0.80, '#3498db'),
        ('2. Train/Test Split\n70% / 30%', 0.1, 0.60, '#9b59b6'),
        ('3. Missing Check\n✓ No Missing Values', 0.1, 0.40, '#2ecc71'),
        ('4. Outlier Removal\n-49% samples (IQR)', 0.1, 0.20, '#e74c3c'),
    ]
    
    stages_right = [
        ('5. Statistics\nAnalysis', 0.55, 0.80, '#f39c12'),
        ('6. Normalization\nMin-Max [0-1]', 0.55, 0.60, '#1abc9c'),
        ('7. Preprocessed\nData', 0.55, 0.40, '#2ecc71'),
        ('8. Ready for ML\nModels', 0.55, 0.20, '#16a085'),
    ]
    
    all_stages = stages + stages_right
    
    for text, x, y, color in all_stages:
        # 박스
        bbox = mpatches.FancyBboxPatch((x, y-0.08), 0.25, 0.12,
                                       boxstyle="round,pad=0.01",
                                       transform=ax.transAxes,
                                       edgecolor='black', facecolor=color,
                                       linewidth=2, alpha=0.8)
        ax.add_patch(bbox)
        
        # 텍스트
        ax.text(x + 0.125, y - 0.02, text,
               ha='center', va='center', fontsize=11, fontweight='bold',
               transform=ax.transAxes, color='white')
    
    # 화살표
    for i in range(len(stages) - 1):
        y_start = stages[i][2] - 0.08
        y_end = stages[i+1][2] + 0.04
        ax.annotate('', xy=(0.225, y_end), xytext=(0.225, y_start),
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'),
                   xycoords='axes fraction')
    
    for i in range(len(stages_right) - 1):
        y_start = stages_right[i][2] - 0.08
        y_end = stages_right[i+1][2] + 0.04
        ax.annotate('', xy=(0.675, y_end), xytext=(0.675, y_start),
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'),
                   xycoords='axes fraction')
    
    # 좌우 연결 화살표
    ax.annotate('', xy=(0.55, 0.76), xytext=(0.35, 0.76),
               arrowprops=dict(arrowstyle='->', lw=3, color='#e74c3c'),
               xycoords='axes fraction')
    
    # 결과 요약
    summary_text = """
    [OK] Training: 105,000 -> 53,362 (49.18% removed)
    [OK] Test: 45,000 -> 22,805 (49.32% removed)
    [OK] All values normalized to [0-1]
    [OK] No missing values
    [OK] Ready for machine learning models
    """
    
    ax.text(0.5, 0.05, summary_text,
           ha='center', va='bottom', fontsize=11, fontweight='bold',
           transform=ax.transAxes,
           bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.8, 
                    edgecolor='black', linewidth=2, pad=1))
    
    plt.tight_layout()
    plt.savefig('05_pipeline_diagram.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("[SAVE] 05_pipeline_diagram.png")
    plt.close()

def create_class_distribution():
    """클래스 분포 차트"""
    print("[GEN] Class Distribution 생성 중...")
    
    header_prep, data_prep = load_data('cs-training-preprocessed.csv')
    
    # 타겟 변수 분포 (loan column, index 8)
    loan_values = [int(row[8]) for row in data_prep]
    loan_counts = Counter(loan_values)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('[TARGET] Target Variable Distribution (Training Data)', fontsize=16, fontweight='bold', y=0.98)
    
    # 바 차트
    ax1 = axes[0]
    labels = ['No Loan (0)', 'Has Loan (1)']
    values = [loan_counts[0], loan_counts[1]]
    colors = ['#e74c3c', '#2ecc71']
    bars = ax1.bar(labels, values, color=colors, edgecolor='black', linewidth=2, width=0.6)
    ax1.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
    ax1.set_title('Absolute Values', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 500,
                f'{val:,}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # 파이 차트
    ax2 = axes[1]
    percentages = [val/sum(values)*100 for val in values]
    wedges, texts, autotexts = ax2.pie(values, labels=labels, autopct='%1.1f%%',
                                        colors=colors, startangle=90,
                                        explode=(0.05, 0.05),
                                        textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax2.set_title('Percentage Distribution', fontsize=12, fontweight='bold')
    
    # 범례
    legend_text = f"""
    Class 0 (No Loan): {values[0]:,} ({percentages[0]:.2f}%)
    Class 1 (Has Loan): {values[1]:,} ({percentages[1]:.2f}%)
    [WARNING] Class Imbalance: {values[1]/values[0]:.2f}:1
    """
    
    fig.text(0.5, -0.05, legend_text,
            ha='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8, 
                     edgecolor='black', linewidth=2, pad=1))
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.96])
    plt.savefig('06_class_distribution.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("[SAVE] 06_class_distribution.png")
    plt.close()

def main():
    print("\n" + "="*60)
    print("[1] 발표용 시각화 자료 생성 시작")
    print("="*60 + "\n")
    
    create_missing_map()
    create_outlier_analysis()
    create_normalization_comparison()
    create_data_distribution()
    create_pipeline_diagram()
    create_class_distribution()
    
    print("\n" + "="*60)
    print("[COMPLETE] 모든 시각화 생성 완료!")
    print("="*60)
    print("\n[FILES CREATED]")
    print("  1) 01_missing_map.png - 결측치 지도")
    print("  2) 02_outlier_analysis.png - 이상치 분석")
    print("  3) 03_normalization_comparison.png - 정규화 비교")
    print("  4) 04_feature_distributions.png - 특성 분포")
    print("  5) 05_pipeline_diagram.png - 파이프라인 다이어그램")
    print("  6) 06_class_distribution.png - 클래스 분포")
    print("\n[TIP] 이 파일들을 PPT에 삽입하여 사용하세요!\n")

if __name__ == '__main__':
    main()
