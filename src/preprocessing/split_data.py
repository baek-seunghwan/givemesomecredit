import csv

# CSV 파일 읽기
with open('givemesomecredit_renamed.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    header = next(reader)
    data = list(reader)

print(f"전체 행 수: {len(data)}")
print(f"컬럼 수: {len(header)}")
print(f"컬럼명: {header}")

# 약 70% train, 30% test로 분할
total_rows = len(data)
train_count = int(total_rows * 0.7)
test_count = total_rows - train_count

print(f"\nTrain 행 수: {train_count}")
print(f"Test 행 수: {test_count}")

# cs-training.csv 생성
with open('cs-training.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(data[:train_count])

# cs-test.csv 생성
with open('cs-test.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(data[train_count:])

print("\n✓ cs-training.csv 생성 완료")
print("✓ cs-test.csv 생성 완료")
