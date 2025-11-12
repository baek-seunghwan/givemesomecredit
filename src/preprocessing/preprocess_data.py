import csv
import statistics
from collections import defaultdict

# 1. 데이터 읽기
def load_data(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        data = [list(map(float, row)) for row in reader]
    return header, data

# 2. 결측치 처리 (결측치 없는지 확인)
def check_missing_values(data, header):
    print("=" * 60)
    print("1️⃣  결측치 검사")
    print("=" * 60)
    missing_count = defaultdict(int)
    for row in data:
        for i, val in enumerate(row):
            if val is None or (isinstance(val, str) and val.strip() == ''):
                missing_count[header[i]] += 1
    
    if missing_count:
        print("❌ 결측치 발견:")
        for col, count in missing_count.items():
            print(f"  {col}: {count}개")
    else:
        print("✅ 결측치 없음")
    return len(missing_count) == 0

# 3. 이상치 탐지 및 제거 (IQR 방식)
def remove_outliers(data, header):
    print("\n" + "=" * 60)
    print("2️⃣  이상치 탐지 및 제거 (IQR 방식)")
    print("=" * 60)
    
    before_count = len(data)
    outlier_indices = set()
    
    # 각 컬럼별로 이상치 탐지
    for col_idx in range(len(header)):
        values = [row[col_idx] for row in data if row[col_idx] is not None]
        
        if len(values) > 0:
            values_sorted = sorted(values)
            q1_idx = len(values_sorted) // 4
            q3_idx = 3 * len(values_sorted) // 4
            q1 = values_sorted[q1_idx]
            q3 = values_sorted[q3_idx]
            iqr = q3 - q1
            
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            for row_idx, row in enumerate(data):
                val = row[col_idx]
                if val < lower_bound or val > upper_bound:
                    outlier_indices.add(row_idx)
    
    # 이상치 행 제거
    data_cleaned = [row for idx, row in enumerate(data) if idx not in outlier_indices]
    after_count = len(data_cleaned)
    removed = before_count - after_count
    
    print(f"이전 행 수: {before_count}")
    print(f"이상치 제거: {removed}개")
    print(f"이후 행 수: {after_count}")
    print(f"제거율: {removed/before_count*100:.2f}%")
    
    return data_cleaned

# 4. 기본 통계 분석
def print_statistics(data, header):
    print("\n" + "=" * 60)
    print("3️⃣  데이터 통계")
    print("=" * 60)
    print(f"총 샘플: {len(data)}")
    print(f"총 특성: {len(header)}")
    print(f"\n컬럼별 통계:")
    print("-" * 60)
    
    for col_idx, col_name in enumerate(header):
        values = [row[col_idx] for row in data]
        if len(values) > 0:
            mean_val = statistics.mean(values)
            min_val = min(values)
            max_val = max(values)
            median_val = statistics.median(values)
            try:
                std_val = statistics.stdev(values)
            except:
                std_val = 0
            
            print(f"\n{col_name}:")
            print(f"  평균: {mean_val:.4f}")
            print(f"  중앙값: {median_val:.4f}")
            print(f"  표준편차: {std_val:.4f}")
            print(f"  최소: {min_val:.4f}")
            print(f"  최대: {max_val:.4f}")

# 5. 정규화 (0-1 범위)
def normalize_data(data, header):
    print("\n" + "=" * 60)
    print("4️⃣  데이터 정규화 (0-1 범위)")
    print("=" * 60)
    
    normalized_data = []
    min_max = []
    
    # 각 컬럼의 최소/최대 계산
    for col_idx in range(len(header)):
        values = [row[col_idx] for row in data]
        min_val = min(values)
        max_val = max(values)
        min_max.append((min_val, max_val))
    
    # 정규화 적용
    for row in data:
        normalized_row = []
        for col_idx, val in enumerate(row):
            min_val, max_val = min_max[col_idx]
            if max_val - min_val == 0:
                normalized_val = 0
            else:
                normalized_val = (val - min_val) / (max_val - min_val)
            normalized_row.append(normalized_val)
        normalized_data.append(normalized_row)
    
    print("✅ 정규화 완료 (모든 값이 0-1 범위로 변환)")
    return normalized_data

# 6. 데이터 저장
def save_data(data, header, filename):
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)
    print(f"✅ 저장 완료: {filename}")

# 메인 전처리 파이프라인
def main():
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 15 + "데이터 전처리 파이프라인" + " " * 17 + "║")
    print("╚" + "=" * 58 + "╝")
    
    # Training 데이터 전처리
    print("\n\n📊 cs-training.csv 전처리 중...")
    header_train, data_train = load_data('cs-training.csv')
    
    check_missing_values(data_train, header_train)
    data_train = remove_outliers(data_train, header_train)
    print_statistics(data_train, header_train)
    data_train_normalized = normalize_data(data_train, header_train)
    save_data(data_train_normalized, header_train, 'cs-training-preprocessed.csv')
    
    # Test 데이터 전처리
    print("\n\n📊 cs-test.csv 전처리 중...")
    header_test, data_test = load_data('cs-test.csv')
    
    check_missing_values(data_test, header_test)
    data_test = remove_outliers(data_test, header_test)
    print_statistics(data_test, header_test)
    data_test_normalized = normalize_data(data_test, header_test)
    save_data(data_test_normalized, header_test, 'cs-test-preprocessed.csv')
    
    print("\n\n" + "=" * 60)
    print("✨ 모든 전처리 완료!")
    print("=" * 60)
    print(f"Training: 원본 {len(data_train)} → 정규화 완료")
    print(f"Test: 원본 {len(data_test)} → 정규화 완료")
    print("\n생성된 파일:")
    print("  - cs-training-preprocessed.csv")
    print("  - cs-test-preprocessed.csv")

if __name__ == '__main__':
    main()
