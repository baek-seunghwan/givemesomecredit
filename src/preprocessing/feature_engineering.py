import csv

def load_data(filename):
    """CSV 파일 읽기"""
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        data = [list(map(float, row)) for row in reader]
    return header, data

def perform_feature_engineering(filename, output_filename, data_type="Training"):
    """특성 공학 수행"""
    print(f"\n{'='*80}")
    print(f"[FEATURE ENGINEERING] {data_type} Data Processing")
    print(f"{'='*80}\n")
    
    header, data = load_data(filename)
    
    print(f"[ORIGINAL] Features: {len(header)}")
    print(f"  {header}\n")
    
    # 1. 상수 특성 제거 (표준편차가 0인 특성)
    print("[STEP 1] Remove Constant Features")
    print("  특성별 표준편차 계산 중...\n")
    
    const_features_idx = []
    for col_idx, col_name in enumerate(header):
        values = [row[col_idx] for row in data]
        mean_val = sum(values) / len(values)
        variance = sum((x - mean_val)**2 for x in values) / len(values)
        std_val = variance ** 0.5
        
        if std_val == 0:
            print(f"  [REMOVE] {col_name}: Std Dev = {std_val:.4f} (상수)")
            const_features_idx.append(col_idx)
        else:
            print(f"  [KEEP]   {col_name}: Std Dev = {std_val:.4f}")
    
    # 2. 상수 특성 제외
    print(f"\n[STEP 2] Apply Feature Selection")
    print(f"  제거할 특성 수: {len(const_features_idx)}")
    
    if const_features_idx:
        print(f"  제거할 인덱스: {const_features_idx}\n")
    
    # 새 헤더와 데이터 생성
    new_header = [col for i, col in enumerate(header) if i not in const_features_idx]
    new_data = []
    
    for row in data:
        new_row = [val for i, val in enumerate(row) if i not in const_features_idx]
        new_data.append(new_row)
    
    print(f"  [ORIGINAL] Features: {len(header)}")
    print(f"  [AFTER REMOVAL] Features: {len(new_header)}\n")
    
    # 3. 특성 순서 재조정 (타겟을 마지막에)
    print("[STEP 3] Reorder Features (Target at the end)")
    print(f"  현재 헤더: {new_header}")
    
    # loan (타겟)이 마지막인지 확인
    if new_header[-1] == 'loan':
        print(f"  [OK] Target variable 'loan' is already at the end\n")
    else:
        print(f"  [WARNING] Target variable needs to be reordered\n")
        loan_idx = new_header.index('loan')
        new_header.append(new_header.pop(loan_idx))
        
        for row in new_data:
            row.append(row.pop(loan_idx))
    
    # 4. 새로운 특성 생성 (선택사항)
    print("[STEP 4] Create New Features (Optional)")
    print(f"  Interaction features 또는 polynomial features 생성 가능")
    print(f"  현재: 기본 특성 사용 (나중에 추가 가능)\n")
    
    # 5. 최종 특성 정보
    print("[FINAL FEATURES] Selected Features")
    print(f"  Total Features: {len(new_header)}\n")
    for i, col_name in enumerate(new_header):
        marker = "[TARGET]" if col_name == 'loan' else "[FEATURE]"
        print(f"  {i+1}. {marker} {col_name}")
    
    # 6. 데이터 저장
    print(f"\n[SAVE] Writing to {output_filename}")
    with open(output_filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(new_header)
        writer.writerows(new_data)
    
    print(f"  [OK] {len(new_data):,} samples saved\n")
    
    return new_header, new_data

def generate_feature_report(header, data):
    """특성 보고서 생성"""
    print(f"\n{'='*80}")
    print("[FEATURE REPORT] Final Dataset Summary")
    print(f"{'='*80}\n")
    
    print(f"[DATASET INFO]")
    print(f"  Total Samples: {len(data):,}")
    print(f"  Total Features: {len(header)}")
    print(f"  Input Features: {len(header) - 1}")
    print(f"  Target Variable: 1 (loan)\n")
    
    print(f"[FEATURE BREAKDOWN]")
    print(f"  Numerical Features: {len(header) - 1}")
    print(f"  Categorical Features: 0\n")
    
    print(f"[TARGET VARIABLE DISTRIBUTION]")
    target_col = [int(row[-1]) for row in data]
    class_0 = sum(1 for t in target_col if t == 0)
    class_1 = sum(1 for t in target_col if t == 1)
    
    print(f"  Class 0: {class_0:,} ({class_0/len(data)*100:.2f}%)")
    print(f"  Class 1: {class_1:,} ({class_1/len(data)*100:.2f}%)")
    print(f"  Class Imbalance Ratio: {class_1/class_0:.2f}:1\n")
    
    print(f"[FEATURE STATISTICS]")
    for col_idx, col_name in enumerate(header[:-1]):  # 타겟 제외
        values = [row[col_idx] for row in data]
        mean_val = sum(values) / len(values)
        min_val = min(values)
        max_val = max(values)
        
        print(f"  {col_name:10} -> Min: {min_val:.4f}, Max: {max_val:.4f}, Mean: {mean_val:.4f}")

def main():
    print("\n" + "="*80)
    print("[START] Feature Engineering Pipeline")
    print("="*80)
    
    # Training 데이터 처리
    print("\n[PROCESSING] Training Data")
    header_train, data_train = perform_feature_engineering(
        'cs-training-preprocessed.csv',
        'cs-training-engineered.csv',
        'Training'
    )
    
    # Test 데이터 처리
    print("\n[PROCESSING] Test Data")
    header_test, data_test = perform_feature_engineering(
        'cs-test-preprocessed.csv',
        'cs-test-engineered.csv',
        'Test'
    )
    
    # 최종 보고서
    print("\n" + "="*80)
    print("[TRAINING DATA] Feature Summary")
    print("="*80)
    generate_feature_report(header_train, data_train)
    
    print("\n" + "="*80)
    print("[TEST DATA] Feature Summary")
    print("="*80)
    generate_feature_report(header_test, data_test)
    
    # 출력 요약
    print("\n" + "="*80)
    print("[COMPLETE] Feature Engineering Finished")
    print("="*80)
    print("\n[OUTPUT FILES]")
    print("  1. cs-training-engineered.csv")
    print("  2. cs-test-engineered.csv\n")
    
    print("[NEXT STEP]")
    print("  모델 학습 준비 완료!")
    print("  - Logistic Regression")
    print("  - Random Forest")
    print("  - XGBoost\n")

if __name__ == '__main__':
    main()
