import csv
from collections import Counter

def load_preprocessed_data(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        data = [list(map(float, row)) for row in reader]
    return header, data

def generate_analysis_report():
    print("\n" + "="*70)
    print(" " * 20 + "📊 전처리 데이터 최종 분석")
    print("="*70)
    
    # Training 데이터
    header_train, data_train = load_preprocessed_data('cs-training-preprocessed.csv')
    header_test, data_test = load_preprocessed_data('cs-test-preprocessed.csv')
    
    print("\n📌 Training 데이터 분석")
    print("-" * 70)
    print(f"총 샘플 수: {len(data_train):,}")
    print(f"총 특성 수: {len(header_train)}")
    print(f"\n컬럼명 및 범위:")
    for i, col_name in enumerate(header_train):
        values = [row[i] for row in data_train]
        print(f"  {i+1}. {col_name:10} → [{min(values):.4f}, {max(values):.4f}]")
    
    print("\n\n📌 Test 데이터 분석")
    print("-" * 70)
    print(f"총 샘플 수: {len(data_test):,}")
    print(f"총 특성 수: {len(header_test)}")
    print(f"\n컬럼명 및 범위:")
    for i, col_name in enumerate(header_test):
        values = [row[i] for row in data_test]
        print(f"  {i+1}. {col_name:10} → [{min(values):.4f}, {max(values):.4f}]")
    
    # 타겟 변수 분포
    print("\n\n📌 타겟 변수 분포")
    print("-" * 70)
    
    loan_train = Counter([int(row[-1]) for row in data_train])
    loan_test = Counter([int(row[-1]) for row in data_test])
    
    print("Training 데이터:")
    for val in sorted(loan_train.keys()):
        count = loan_train[val]
        pct = count / len(data_train) * 100
        print(f"  loan={val}: {count:,} ({pct:.2f}%)")
    
    print("\nTest 데이터:")
    for val in sorted(loan_test.keys()):
        count = loan_test[val]
        pct = count / len(data_test) * 100
        print(f"  loan={val}: {count:,} ({pct:.2f}%)")
    
    # 데이터 품질 체크
    print("\n\n📌 데이터 품질 체크")
    print("-" * 70)
    
    # Training 데이터
    print("Training 데이터:")
    null_count_train = 0
    for row in data_train:
        for val in row:
            if val is None or (isinstance(val, float) and (val != val)):  # NaN check
                null_count_train += 1
    print(f"  ✅ 결측치: {null_count_train}개")
    
    # 중복 행 확인
    unique_rows_train = len(set(tuple(row) for row in data_train))
    duplicate_rows_train = len(data_train) - unique_rows_train
    print(f"  ℹ️  중복 행: {duplicate_rows_train}개")
    
    # Test 데이터
    print("\nTest 데이터:")
    null_count_test = 0
    for row in data_test:
        for val in row:
            if val is None or (isinstance(val, float) and (val != val)):  # NaN check
                null_count_test += 1
    print(f"  ✅ 결측치: {null_count_test}개")
    
    unique_rows_test = len(set(tuple(row) for row in data_test))
    duplicate_rows_test = len(data_test) - unique_rows_test
    print(f"  ℹ️  중복 행: {duplicate_rows_test}개")
    
    # 모델 학습 권장사항
    print("\n\n🎯 모델 학습 권장사항")
    print("-" * 70)
    print("""
1. 특성 선택:
   - 'gg', '3059', 'Defaul' 제거 권장 (상수값)
   - 7개 특성으로 축소: prop, age, ratio, income, depen, loan + 1개 타겟

2. 모델 후보:
   ✓ Logistic Regression (해석 가능성)
   ✓ Random Forest (특성 중요도)
   ✓ Gradient Boosting (성능)
   ✓ SVM (정규화된 데이터에 효과적)

3. 검증 전략:
   ✓ K-Fold Cross Validation (k=5 또는 10)
   ✓ Stratified Split (클래스 불균형 고려)

4. 메트릭:
   ✓ Accuracy (전체 정확도)
   ✓ Precision/Recall (신용도 중요)
   ✓ F1-Score (균형 지표)
   ✓ ROC-AUC (분류 성능)

5. 클래스 불균형 처리:
   ✓ Class Weight Adjustment
   ✓ SMOTE (오버샘플링)
   ✓ 임계값 조정
""")
    
    print("\n" + "="*70)
    print(" " * 25 + "✨ 분석 완료!")
    print("="*70 + "\n")

if __name__ == '__main__':
    generate_analysis_report()
