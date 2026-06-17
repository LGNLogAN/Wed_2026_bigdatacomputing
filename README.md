# 다중 특성 회귀 모델 파이프라인 구축 및 Streamlit 웹 서비스 배포

## 사용 데이터셋

WHO 기대수명 데이터셋을 사용합니다.

```python
https://github.com/dongupak/DataML/raw/main/csv/life_expectancy.csv
```

## 사용한 독립변수

`Schooling` 컬럼은 독립변수로 사용하지 않았습니다.

- Adult Mortality
- BMI
- GDP
- Alcohol
- Polio

## 사용한 모델

- Linear: `StandardScaler + LinearRegression`
- Poly: `StandardScaler + PolynomialFeatures(degree=3) + LinearRegression`
- Ridge: `StandardScaler + PolynomialFeatures(degree=3) + Ridge(alpha=1.0)`

모델은 전체 훈련 데이터가 아니라 훈련 데이터에서 무작위로 추출한 50개 샘플만 사용해 학습합니다.

## 파일 구성

```text
life_expectancy_assignment.ipynb
app.py
linear_model.pkl
poly_model.pkl
ridge_model.pkl
README.md
```

## 실행 방법

필요 라이브러리를 설치합니다.

```bash
pip install streamlit pyngrok joblib pandas scikit-learn matplotlib
```

Streamlit 앱을 실행합니다.

```bash
streamlit run app.py
```

## Colab에서 Streamlit 실행

Colab 노트북에서 다음 명령어로 Streamlit을 실행할 수 있습니다.

```python
!streamlit run app.py --server.port 8501 &
```

## ngrok 배포 방법

Colab에서 ngrok으로 외부 접속 URL을 생성합니다.

```python
from pyngrok import ngrok

# ngrok authtoken이 필요한 경우 아래 주석을 해제하고 본인 토큰을 입력합니다.
# !ngrok config add-authtoken "본인_ngrok_authtoken"

public_url = ngrok.connect(8501)
print(public_url)
```

## 제출 안내

결과 화면 캡처는 `life_expectancy_assignment.ipynb`에 첨부해야 합니다.
