# Give Me Some Credit - Credit Risk Prediction Project

머신러닝을 이용한 신용위험(Credit Risk) 예측 프로젝트입니다.

## 📊 프로젝트 개요

이 프로젝트는 "Give Me Some Credit" 데이터셋을 활용하여 고객의 신용 부도 위험을 예측하는 머신러닝 모델을 개발합니다.



### 주요 특징
- **탐색적 데이터 분석(EDA)**: 데이터 분포, 상관관계, 이상치 분석
- **데이터 전처리**: 정규화, 특성 공학, 데이터 분할
- **모델 훈련**: 로지스틱 회귀, SVM 등 다양한 알고리즘 적용
- **시각화**: 의사결정 경계, 모델 성능 시각화
- **분석 리포트**: 상세한 모델 평가 및 인사이트 제공

## 🗂️ 프로젝트 구조

```
project-root/
├── src/
│   ├── preprocessing/          # 데이터 전처리
│   │   ├── preprocess_data.py
│   │   ├── feature_engineering.py
│   │   └── split_data.py
│   ├── models/                 # 모델 훈련
│   │   └── model_training.py
│   ├── analysis/               # 데이터 분석
│   │   ├── eda_analysis.py
│   │   ├── outlier_analysis_advanced.py
│   │   ├── outlier_impact_comparison.py
│   │   └── analysis_report.py
│   └── visualization/          # 시각화
│       ├── visualization.py
│       ├── logistic_regression_visualization.py
│       ├── svm_decision_boundary.py
│       └── advanced_visualization.py
├── docs/                       # 문서
│   ├── PRESENTATION_GUIDE.md
│   ├── CUSTOM_VISUALIZATION_GUIDE.md
│   ├── ADVANCED_VISUALIZATION_GUIDE.md
│   ├── MODEL_EVALUATION_REPORT.md
│   ├── preprocessing_report.md
│   ├── ADVANCED_VISUALIZATION_REPORT.md
│   └── VISUALIZATION_COMPLETION_SUMMARY.md
├── data/                       # 데이터 (CSV 파일)
├── README.md                   # 프로젝트 설명
└── requirements.txt            # 의존성 패키지
```

## � 시작하기

### 1. 환경 설정

```bash
# 저장소 복제
git clone https://github.com/your-username/give-me-some-credit.git
cd give-me-some-credit

# 가상환경 생성 (선택사항)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 또는
venv\Scripts\activate  # Windows

# 필요한 패키지 설치
pip install -r requirements.txt
```

### 2. 데이터 준비

`data/` 폴더에 다음 파일들을 배치합니다:
- `givemesomecredit_renamed.csv` - 원본 데이터
- `cs-training-preprocessed.csv` - 전처리된 훈련 데이터
- `cs-test-preprocessed.csv` - 전처리된 테스트 데이터

### 3. 실행 순서

```bash
# 1. 탐색적 데이터 분석
python src/analysis/eda_analysis.py

# 2. 데이터 전처리
python src/preprocessing/preprocess_data.py

# 3. 특성 공학
python src/preprocessing/feature_engineering.py

# 4. 데이터 분할
python src/preprocessing/split_data.py

# 5. 모델 훈련
python src/models/model_training.py

# 6. 시각화
python src/visualization/visualization.py
python src/visualization/logistic_regression_visualization.py
python src/visualization/svm_decision_boundary.py
python src/visualization/advanced_visualization.py
```

## � 주요 분석 내용

### 데이터 통계
- **훈련 샘플**: 약 53,000개 (이상치 제거 후)
- **테스트 샘플**: 약 23,000개 (이상치 제거 후)
- **특성**: 9개 (타겟 변수 포함)
- **클래스 분포**: 불균형 데이터 (부도율 약 74%)

### 모델 성능
- **로지스틱 회귀**: 기준선 모델
- **SVM**: 고성능 분류
- **앙상블 모델**: 추가 개선 가능

### 주요 인사이트
- 특정 특성들의 높은 상관관계 분석
- 이상치 탐지 및 영향도 평가
- 클래스 불균형 처리 방안 제시

## 📚 문서

자세한 정보는 `docs/` 폴더의 다음 문서들을 참고하세요:

- **PRESENTATION_GUIDE.md**: 프로젝트 전체 가이드
- **CUSTOM_VISUALIZATION_GUIDE.md**: 시각화 커스터마이징
- **ADVANCED_VISUALIZATION_GUIDE.md**: 고급 시각화 기법
- **MODEL_EVALUATION_REPORT.md**: 모델 평가 상세 리포트
- **preprocessing_report.md**: 전처리 과정 상세 설명

## 🔧 기술 스택

- **Python 3.8+**
- **Pandas**: 데이터 조작 및 분석
- **NumPy**: 수치 계산
- **Scikit-learn**: 머신러닝 알고리즘
- **Matplotlib & Seaborn**: 데이터 시각화

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

---

**마지막 업데이트**: 2025년 11월 12일

**🎓 프로젝트 완료! 모델 학습을 시작할 수 있습니다.** ✨
