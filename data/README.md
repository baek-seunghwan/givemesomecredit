# Data Directory

이 디렉토리에는 프로젝트에 사용되는 데이터 파일들이 저장됩니다.

## 필요한 파일들

### 원본 데이터
- `givemesomecredit_renamed.csv` - 원본 "Give Me Some Credit" 데이터셋

### 전처리된 데이터
- `cs-training-preprocessed.csv` - 전처리된 훈련 데이터
- `cs-test-preprocessed.csv` - 전처리된 테스트 데이터

## 사용 방법

1. 위의 CSV 파일들을 이 디렉토리에 배치합니다.
2. 프로젝트 스크립트들이 자동으로 이 디렉토리에서 데이터를 읽어옵니다.

## 참고

- 데이터 파일은 `.gitignore`에 의해 Git에 제외됩니다.
- 개인 프로젝트 환경에서는 이 파일들을 직접 관리해야 합니다.
- 공유 시에는 데이터 소스와 처리 방법만 문서화합니다.
