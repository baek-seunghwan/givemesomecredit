import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, roc_auc_score
import seaborn as sns

# 스타일 설정
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_data(filename):
    """CSV 파일 읽기"""
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        data = [list(map(float, row)) for row in reader]
    return header, data

def prepare_dataset(data):
    """입력과 타겟으로 데이터 분리"""
    X = [row[:-1] for row in data]
    y = [int(row[-1]) for row in data]
    return X, y

def train_models(X_train, y_train):
    """모델 훈련"""
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
        'XGBoost': XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, verbosity=0)
    }
    
    trained_models = {}
    for name, model in models.items():
        print(f"  Training {name}...", end=" ")
        model.fit(X_train, y_train)
        trained_models[name] = model
        print("OK")
    
    return trained_models

def create_confusion_matrix_plot(y_true, y_pred, model_name):
    """혼동 행렬 히트맵"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    
    # 혼동 행렬
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Loan', 'Loan'],
                yticklabels=['No Loan', 'Loan'],
                cbar_kws={'label': 'Count'},
                annot_kws={'size': 14, 'weight': 'bold'})
    
    plt.title(f'[{model_name}] Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Actual', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted', fontsize=12, fontweight='bold')
    
    # 정확도 표시
    accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
    plt.text(1, -0.15, f'Accuracy: {accuracy:.4f}', 
            transform=plt.gca().transAxes, fontsize=11, fontweight='bold',
            ha='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    return plt.gcf()

def create_roc_curves_plot(y_true, predictions_dict):
    """ROC 곡선 비교"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = {'Logistic Regression': '#1f77b4', 'Random Forest': '#ff7f0e', 'XGBoost': '#2ca02c'}
    
    for model_name, y_pred_proba in predictions_dict.items():
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, color=colors[model_name], lw=2.5,
               label=f'{model_name} (AUC = {roc_auc:.4f})')
    
    # 무작위 분류 기준선
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier', alpha=0.5)
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title('[ROC CURVE] Model Performance Comparison', fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_precision_recall_curve(y_true, predictions_dict):
    """Precision-Recall 곡선"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = {'Logistic Regression': '#1f77b4', 'Random Forest': '#ff7f0e', 'XGBoost': '#2ca02c'}
    
    for model_name, y_pred_proba in predictions_dict.items():
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        
        ax.plot(recall, precision, color=colors[model_name], lw=2.5,
               label=f'{model_name}')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax.set_title('[PRECISION-RECALL] Trade-off Analysis', fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_feature_importance_plot(model, feature_names):
    """특성 중요도"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        return None
    
    indices = np.argsort(importances)[::-1][:5]  # Top 5
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors_bar = plt.cm.viridis(np.linspace(0, 1, len(indices)))
    bars = ax.barh(range(len(indices)), importances[indices], color=colors_bar, edgecolor='black', linewidth=1.5)
    
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices], fontsize=11)
    ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
    ax.set_title('[TOP 5] Feature Importance', fontsize=16, fontweight='bold', pad=20)
    
    # 값 표시
    for i, (idx, bar) in enumerate(zip(indices, bars)):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, 
               f' {importances[idx]:.4f}', 
               ha='left', va='center', fontweight='bold', fontsize=10)
    
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    return fig

def create_radar_chart(metrics_dict):
    """모델 성능 비교 차트"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = list(metrics_dict.keys())
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    
    x = np.arange(len(metrics_names))
    width = 0.25
    
    colors = {'Logistic Regression': '#1f77b4', 'Random Forest': '#ff7f0e', 'XGBoost': '#2ca02c'}
    
    for i, model_name in enumerate(models):
        values = [metrics_dict[model_name]['accuracy'],
                 metrics_dict[model_name]['precision'],
                 metrics_dict[model_name]['recall'],
                 metrics_dict[model_name]['f1'],
                 metrics_dict[model_name]['roc_auc']]
        
        offset = (i - 1) * width
        bars = ax.bar(x + offset, values, width, label=model_name, color=colors[model_name], edgecolor='black', linewidth=1.5)
        
        # 값 표시
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('[COMPARISON] Model Performance Metrics', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_correlation_heatmap(X_train, feature_names):
    """상관계수 히트맵"""
    # 상관계수 계산
    n = len(X_train[0])
    corr_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            col_i = [row[i] for row in X_train]
            col_j = [row[j] for row in X_train]
            
            mean_i = sum(col_i) / len(col_i)
            mean_j = sum(col_j) / len(col_j)
            
            cov = sum((col_i[k] - mean_i) * (col_j[k] - mean_j) for k in range(len(col_i))) / len(col_i)
            
            std_i = (sum((x - mean_i)**2 for x in col_i) / len(col_i)) ** 0.5
            std_j = (sum((x - mean_j)**2 for x in col_j) / len(col_j)) ** 0.5
            
            if std_i > 0 and std_j > 0:
                corr_matrix[i][j] = cov / (std_i * std_j)
            else:
                corr_matrix[i][j] = 1 if i == j else 0
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
                center=0, vmin=-1, vmax=1,
                xticklabels=feature_names,
                yticklabels=feature_names,
                cbar_kws={'label': 'Correlation'},
                annot_kws={'size': 10},
                square=True, ax=ax)
    
    plt.title('[CORRELATION] Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    return fig

def main():
    print("\n" + "="*80)
    print("[START] Advanced Visualization Generation")
    print("="*80 + "\n")
    
    # 데이터 로드
    print("[LOAD] Loading Data...")
    header_train, data_train = load_data('cs-training-engineered.csv')
    header_test, data_test = load_data('cs-test-engineered.csv')
    
    X_train, y_train = prepare_dataset(data_train)
    X_test, y_test = prepare_dataset(data_test)
    
    feature_names = header_train[:-1]
    
    # 모델 훈련
    print("\n[TRAIN] Training Models...")
    trained_models = train_models(X_train, y_train)
    
    # 1. Confusion Matrix (XGBoost만)
    print("\n[VIZ-1] Creating Confusion Matrix...", end=" ")
    y_pred_xgb = trained_models['XGBoost'].predict(X_test)
    fig1 = create_confusion_matrix_plot(y_test, y_pred_xgb, 'XGBoost')
    plt.savefig('07_confusion_matrix.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("OK")
    
    # 2. ROC Curves
    print("[VIZ-2] Creating ROC Curves...", end=" ")
    predictions_dict = {}
    for model_name, model in trained_models.items():
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        predictions_dict[model_name] = y_pred_proba
    
    fig2 = create_roc_curves_plot(y_test, predictions_dict)
    plt.savefig('08_roc_curves.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("OK")
    
    # 3. Precision-Recall Curves
    print("[VIZ-3] Creating Precision-Recall Curves...", end=" ")
    fig3 = create_precision_recall_curve(y_test, predictions_dict)
    plt.savefig('09_precision_recall_curve.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("OK")
    
    # 4. Feature Importance (XGBoost)
    print("[VIZ-4] Creating Feature Importance Chart...", end=" ")
    fig4 = create_feature_importance_plot(trained_models['XGBoost'], feature_names)
    plt.savefig('10_feature_importance.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("OK")
    
    # 5. Radar Chart
    print("[VIZ-5] Creating Radar Chart...", end=" ")
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    metrics_dict = {}
    for model_name, model in trained_models.items():
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        metrics_dict[model_name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
    
    fig5 = create_radar_chart(metrics_dict)
    plt.savefig('11_radar_chart.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("OK")
    
    # 6. Correlation Heatmap
    print("[VIZ-6] Creating Correlation Heatmap...", end=" ")
    fig6 = create_correlation_heatmap(X_train, feature_names)
    plt.savefig('12_correlation_heatmap.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("OK")
    
    print("\n" + "="*80)
    print("[COMPLETE] Advanced Visualization Generation Finished")
    print("="*80)
    print("\n[FILES CREATED]")
    print("  07_confusion_matrix.png")
    print("  08_roc_curves.png")
    print("  09_precision_recall_curve.png")
    print("  10_feature_importance.png")
    print("  11_radar_chart.png")
    print("  12_correlation_heatmap.png\n")

if __name__ == '__main__':
    main()
