import csv
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import numpy as np

def load_data(filename):
    """CSV 파일 읽기"""
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        data = [list(map(float, row)) for row in reader]
    return header, data

def prepare_dataset(data):
    """입력과 타겟으로 데이터 분리"""
    X = [row[:-1] for row in data]  # 모든 특성
    y = [int(row[-1]) for row in data]  # 타겟
    return X, y

def train_model(model_name, model, X_train, y_train):
    """모델 학습"""
    print(f"\n  [TRAIN] {model_name} 학습 중...", end=" ")
    model.fit(X_train, y_train)
    print("[OK]")
    return model

def evaluate_model(model_name, model, X_test, y_test):
    """모델 평가"""
    print(f"\n  [EVAL] {model_name} 평가 중...")
    
    # 예측
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # 메트릭 계산
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # 결과 출력
    print(f"    Accuracy:  {accuracy:.4f}")
    print(f"    Precision: {precision:.4f}")
    print(f"    Recall:    {recall:.4f}")
    print(f"    F1-Score:  {f1:.4f}")
    print(f"    ROC-AUC:   {roc_auc:.4f}")
    
    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }

def handle_class_imbalance(X_train, y_train):
    """클래스 불균형 처리 (클래스 가중치 계산)"""
    print(f"\n  [BALANCE] 클래스 불균형 처리 중...")
    
    class_0_count = sum(1 for y in y_train if y == 0)
    class_1_count = sum(1 for y in y_train if y == 1)
    
    # 클래스 가중치 계산
    total = len(y_train)
    weight_0 = total / (2 * class_0_count)
    weight_1 = total / (2 * class_1_count)
    
    class_weights = {0: weight_0, 1: weight_1}
    
    print(f"    Class 0: {class_0_count:,} | Weight: {weight_0:.4f}")
    print(f"    Class 1: {class_1_count:,} | Weight: {weight_1:.4f}")
    
    return class_weights

def main():
    print("\n" + "="*80)
    print("[START] Model Training Pipeline")
    print("="*80)
    
    # 데이터 로드
    print("\n[LOAD] Loading Data...")
    header_train, data_train = load_data('cs-training-engineered.csv')
    header_test, data_test = load_data('cs-test-engineered.csv')
    
    print(f"  Training: {len(data_train):,} samples")
    print(f"  Test: {len(data_test):,} samples")
    print(f"  Features: {len(header_train) - 1}")
    
    # 데이터 준비
    X_train, y_train = prepare_dataset(data_train)
    X_test, y_test = prepare_dataset(data_test)
    
    # 클래스 불균형 처리
    print(f"\n{'='*80}")
    print("[CLASS IMBALANCE] Handling Imbalanced Dataset")
    print(f"{'='*80}")
    class_weights = handle_class_imbalance(X_train, y_train)
    
    # 모델 정의
    print(f"\n{'='*80}")
    print("[MODELS] Training Multiple Models")
    print(f"{'='*80}")
    
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            class_weight=class_weights,
            random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=10,
            class_weight=class_weights,
            random_state=42,
            n_jobs=-1
        ),
        'XGBoost': XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=class_weights[0] / class_weights[1],
            random_state=42,
            verbosity=0
        )
    }
    
    # 모델 학습 및 평가
    results = []
    
    for model_name, model in models.items():
        print(f"\n[{model_name.upper()}]")
        trained_model = train_model(model_name, model, X_train, y_train)
        result = evaluate_model(model_name, trained_model, X_test, y_test)
        results.append(result)
    
    # 모델 성능 비교
    print(f"\n\n{'='*80}")
    print("[COMPARISON] Model Performance Comparison")
    print(f"{'='*80}\n")
    
    print(f"{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'ROC-AUC':<12}")
    print("-" * 80)
    
    for result in results:
        print(f"{result['model_name']:<20} "
              f"{result['accuracy']:<12.4f} "
              f"{result['precision']:<12.4f} "
              f"{result['recall']:<12.4f} "
              f"{result['f1']:<12.4f} "
              f"{result['roc_auc']:<12.4f}")
    
    # 최고 성능 모델
    print(f"\n{'='*80}")
    print("[BEST MODEL] Selecting Best Model")
    print(f"{'='*80}\n")
    
    best_result = max(results, key=lambda x: x['f1'])  # F1-Score 기준
    
    print(f"  Model: {best_result['model_name']}")
    print(f"  F1-Score: {best_result['f1']:.4f}")
    print(f"  Accuracy: {best_result['accuracy']:.4f}")
    print(f"  ROC-AUC: {best_result['roc_auc']:.4f}")
    
    # 상세 분석
    print(f"\n{'='*80}")
    print("[ANALYSIS] Detailed Analysis of Best Model")
    print(f"{'='*80}\n")
    
    y_pred = best_result['y_pred']
    
    # Confusion Matrix 계산
    tp = sum(1 for i in range(len(y_test)) if y_test[i] == 1 and y_pred[i] == 1)
    tn = sum(1 for i in range(len(y_test)) if y_test[i] == 0 and y_pred[i] == 0)
    fp = sum(1 for i in range(len(y_test)) if y_test[i] == 0 and y_pred[i] == 1)
    fn = sum(1 for i in range(len(y_test)) if y_test[i] == 1 and y_pred[i] == 0)
    
    print("[Confusion Matrix]")
    print(f"  True Positives:  {tp:,}")
    print(f"  True Negatives:  {tn:,}")
    print(f"  False Positives: {fp:,}")
    print(f"  False Negatives: {fn:,}")
    
    # 추가 메트릭
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    print(f"\n[Additional Metrics]")
    print(f"  Sensitivity (Recall): {sensitivity:.4f}")
    print(f"  Specificity: {specificity:.4f}")
    print(f"  False Positive Rate: {fp/(fp+tn):.4f}")
    print(f"  False Negative Rate: {fn/(fn+tp):.4f}")
    
    print(f"\n{'='*80}")
    print("[COMPLETE] Model Training Finished")
    print("="*80 + "\n")

if __name__ == '__main__':
    main()
