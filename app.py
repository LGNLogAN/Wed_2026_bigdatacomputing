import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


# WHO 기대수명 데이터셋 URL
DATA_URL = "https://github.com/dongupak/DataML/raw/main/csv/life_expectancy.csv"

# 예측 대상 컬럼
TARGET = "Life expectancy"

# 과제에서 지정한 5개 독립변수
# 주의: Schooling 컬럼은 독립변수로 사용하지 않는다.
FEATURES = [
    "Adult Mortality",
    "BMI",
    "GDP",
    "Alcohol",
    "Polio",
]

MODEL_FILES = {
    "Linear": "linear_model.pkl",
    "Poly": "poly_model.pkl",
    "Ridge": "ridge_model.pkl",
}


@st.cache_data
def load_data():
    """데이터를 불러오고 컬럼명 정리 및 결측치 제거를 수행한다."""
    df = pd.read_csv(DATA_URL)
    df.columns = df.columns.str.strip()
    df = df.rename(columns={"Adult mortality": "Adult Mortality"})
    df = df.dropna()
    return df


@st.cache_data
def prepare_data(df):
    """독립변수/종속변수 분리, train/test split, 훈련 샘플 50개 추출을 수행한다."""
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    # 과대적합 관찰을 위해 전체 훈련 데이터가 아니라 50개 샘플만 사용한다.
    X_train_sample = X_train.sample(n=50, random_state=42)
    y_train_sample = y_train.loc[X_train_sample.index]

    return X_train_sample, X_test, y_train_sample, y_test


def build_models():
    """과제 요구사항에 맞는 3개 파이프라인 모델을 생성한다."""
    linear_model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("linear", LinearRegression()),
        ]
    )

    poly_model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("poly", PolynomialFeatures(degree=3, include_bias=False)),
            ("linear", LinearRegression()),
        ]
    )

    ridge_model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("poly", PolynomialFeatures(degree=3, include_bias=False)),
            ("ridge", Ridge(alpha=1.0)),
        ]
    )

    return {
        "Linear": linear_model,
        "Poly": poly_model,
        "Ridge": ridge_model,
    }


def train_and_save_models(X_train_sample, y_train_sample):
    """모델을 50개 훈련 샘플로 학습한 뒤 pkl 파일로 저장한다."""
    models = build_models()

    for model_name, model in models.items():
        model.fit(X_train_sample, y_train_sample)
        joblib.dump(model, MODEL_FILES[model_name])

    return models


def load_or_train_models(X_train_sample, y_train_sample):
    """pkl 파일이 있으면 로드하고, 없으면 학습 후 저장한다."""
    if all(os.path.exists(path) for path in MODEL_FILES.values()):
        return {
            model_name: joblib.load(path)
            for model_name, path in MODEL_FILES.items()
        }

    return train_and_save_models(X_train_sample, y_train_sample)


def get_complexity(model, model_name, X_train_sample):
    """모델이 최종적으로 사용하는 특성 개수를 계산한다."""
    if model_name == "Linear":
        return len(FEATURES)

    scaled_sample = model.named_steps["scaler"].transform(X_train_sample.iloc[:1])
    poly = model.named_steps["poly"]
    return poly.transform(scaled_sample).shape[1]


def evaluate_models(models, X_train_sample, X_test, y_train_sample, y_test):
    """Train/Test R2, MSE, Complexity를 계산해 DataFrame으로 반환한다."""
    results = []

    for model_name, model in models.items():
        train_pred = model.predict(X_train_sample)
        test_pred = model.predict(X_test)

        results.append(
            {
                "Model": model_name,
                "Train R2": r2_score(y_train_sample, train_pred),
                "Test R2": r2_score(y_test, test_pred),
                "Train MSE": mean_squared_error(y_train_sample, train_pred),
                "Test MSE": mean_squared_error(y_test, test_pred),
                "Complexity": get_complexity(model, model_name, X_train_sample),
            }
        )

    result_df = pd.DataFrame(results)
    return result_df


def make_sidebar_inputs(df):
    """5개 특성에 대한 Streamlit 사이드바 슬라이더를 생성한다."""
    st.sidebar.header("입력값 설정")

    adult_mortality = st.sidebar.slider(
        "Adult Mortality",
        float(df["Adult Mortality"].min()),
        float(df["Adult Mortality"].max()),
        float(df["Adult Mortality"].mean()),
    )

    bmi = st.sidebar.slider(
        "BMI",
        float(df["BMI"].min()),
        float(df["BMI"].max()),
        float(df["BMI"].mean()),
    )

    gdp = st.sidebar.slider(
        "GDP",
        float(df["GDP"].min()),
        float(df["GDP"].max()),
        float(df["GDP"].mean()),
    )

    alcohol = st.sidebar.slider(
        "Alcohol",
        float(df["Alcohol"].min()),
        float(df["Alcohol"].max()),
        float(df["Alcohol"].mean()),
    )

    polio = st.sidebar.slider(
        "Polio",
        float(df["Polio"].min()),
        float(df["Polio"].max()),
        float(df["Polio"].mean()),
    )

    input_data = pd.DataFrame(
        [
            {
                "Adult Mortality": adult_mortality,
                "BMI": bmi,
                "GDP": gdp,
                "Alcohol": alcohol,
                "Polio": polio,
            }
        ]
    )

    return input_data


def main():
    st.title("WHO 기대수명 예측 웹 서비스")

    df = load_data()
    X_train_sample, X_test, y_train_sample, y_test = prepare_data(df)
    models = load_or_train_models(X_train_sample, y_train_sample)
    result_df = evaluate_models(
        models,
        X_train_sample,
        X_test,
        y_train_sample,
        y_test,
    )

    input_data = make_sidebar_inputs(df)

    model_choice = st.selectbox(
        "사용할 모델을 선택하세요",
        ["Linear", "Poly", "Ridge"],
    )

    selected_model = models[model_choice]
    prediction = selected_model.predict(input_data)[0]

    st.metric("예측 기대수명", f"{prediction:.2f} 세")

    st.subheader("모델 성능 비교")
    st.dataframe(
        result_df.style.format(
            {
                "Train R2": "{:.4f}",
                "Test R2": "{:.4f}",
                "Train MSE": "{:.4f}",
                "Test MSE": "{:.4f}",
            }
        ),
        use_container_width=True,
    )

    st.subheader("Test R2 Score Comparison")
    fig, ax = plt.subplots()
    ax.bar(result_df["Model"], result_df["Test R2"])
    ax.set_xlabel("Model")
    ax.set_ylabel("Test R2")
    ax.set_title("Test R2 Score Comparison")
    st.pyplot(fig)

    st.subheader("입력 데이터")
    st.dataframe(input_data, use_container_width=True)


if __name__ == "__main__":
    main()
