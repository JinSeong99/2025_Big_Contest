from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
import requests  # 🔹 추가 (GitHub에서 폰트 다운로드용)

# =====================================
# 🔤 NanumGothic 폰트 GitHub에서 불러오기
# =====================================
# 🔹 Cloud 환경에 설치된 나눔고딕 경로
font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"

if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams["font.family"] = "NanumGothic"
    print("✅ NanumGothic 폰트 설정 완료 (시스템 폰트 사용)")
else:
    plt.rcParams["font.family"] = "DejaVu Sans"
    print("⚠️ NanumGothic 경로 없음, 기본 폰트로 대체")

plt.rcParams["axes.unicode_minus"] = False
sns.set_style("whitegrid")



# ==========================
# 데이터 불러오기
# ==========================
try:
    last_df = pd.read_excel("KPI_file.xlsx")
    threshold_df = pd.read_excel("threshold.xlsx")
except FileNotFoundError as e:
    raise FileNotFoundError(
        f"❌ 파일을 찾을 수 없습니다. 같은 폴더(main)에 KPI_file.xlsx와 threshold.xlsx가 있는지 확인하세요.\n세부 오류: {e}"
    )

threshold_df['지표'] = threshold_df['지표'].astype(str).str.replace(" ", "")
threshold_df.set_index('지표', inplace=True)



# ==========================
# 문자열 정규화 함수 (공백/유니코드 공백 제거)
# ==========================
def _norm(s: object) -> str:
    """
    문자열을 정규화:
    - None / float 값 대응
    - 일반 공백, NBSP(\u00A0), zero-width space(\u200B) 제거
    - 대소문자/공백 무시 일관 처리
    """
    return re.sub(r'\s+', '', str(s)).replace('\u00A0', '').replace('\u200B', '').strip()

# 인덱스/컬럼 정규화
last_df.columns = [str(c).strip() for c in last_df.columns]
threshold_df.index = [str(i).strip() for i in threshold_df.index]
threshold_df.columns = [str(c).strip() for c in threshold_df.columns]

# ==========================
# 안전 매칭을 위한 맵 구성
# ==========================
_idx_map = {_norm(i): i for i in threshold_df.index}
_col_map = {_norm(c): c for c in threshold_df.columns}

# 임계치 컬럼 자동 인식
warn_col = _col_map.get(_norm("경고임계치"), "경고임계치")
danger_col = _col_map.get(_norm("위험임계치"), "위험임계치")

# ==========================
# 평가 지표 함수
# ==========================
def evaluate_forecast(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-8, None))) * 100
    return mae, rmse, mape

# ==========================
# Prophet 기반 시계열 예측 함수
# ==========================
def evaluate_forecast_model_prophet(last_df, threshold_df, forecast_months=10, pre_close_months=6):
    """
    Prophet-based time-series forecasting (with English-only visualization)
    - last_df: merchant KPI data
    - threshold_df: threshold values per KPI
    """

    df = last_df.copy()
    df["ds"] = pd.to_datetime(df["기준년월"], format="%Y%m")

    # Split alive and closed merchants
    alive_df = df[df["폐업여부"] == 0].copy()
    closed_df = df[df["폐업여부"] == 1].copy()

    # Keep only last n months for closed stores
    closed_pre = (
        closed_df.sort_values(["가맹점구분번호", "ds"])
        .groupby("가맹점구분번호", group_keys=False)
        .apply(lambda g: g.tail(pre_close_months))
    )

    total_df = pd.concat([alive_df, closed_pre], axis=0).reset_index(drop=True)
    total_df = total_df.sort_values(["가맹점구분번호", "ds"])

    # KPI list
    indicators = ["매출안정성지표", "경쟁우위 지표", "고객 충성도 지표"]
    results = []

    for target in indicators:
        key = _norm(target)
        matched_idx = _idx_map.get(key, None)

        if matched_idx is None:
            print(f"⚠️ Could not find threshold for '{target}' (normalized='{key}') → skipped")
            continue

        # Column check
        if target not in total_df.columns:
            alt = [c for c in total_df.columns if _norm(c) == key]
            if alt:
                target = alt[0]
            else:
                print(f"⚠️ '{target}' not found in last_df → skipped")
                continue

        sub = total_df[["ds", target]].dropna().sort_values("ds").copy()
        if len(sub) < 10:
            print(f"⚠️ Not enough data for '{target}' (len={len(sub)}) → skipped")
            continue

        prophet_df = sub.rename(columns={target: "y"})

        # Prophet model
        m = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            changepoint_prior_scale=1
        )
        m.fit(prophet_df)

        # Forecast
        future = m.make_future_dataframe(periods=forecast_months, freq="MS")
        forecast = m.predict(future)

        # Evaluation
        y_true = prophet_df["y"].iloc[-min(forecast_months, len(prophet_df)):]
        y_pred = forecast["yhat"].iloc[-min(forecast_months, len(forecast)):]
        mae, rmse, mape = evaluate_forecast(y_true, y_pred)

        # Thresholds
        warn_th = float(threshold_df.loc[matched_idx, warn_col])
        danger_th = float(threshold_df.loc[matched_idx, danger_col])

        # ==============================
        # 🎨 Visualization (English only)
        # ==============================
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(forecast["ds"], forecast["yhat"], color="#1f77b4", label="Predicted Trend")
        ax.axhline(y=warn_th, color="orange", linestyle="--", label=f"Warning {warn_th:.3f}")
        ax.axhline(y=danger_th, color="red", linestyle="--", label=f"Danger {danger_th:.3f}")
        ax.axvspan(
            forecast["ds"].iloc[-forecast_months],
            forecast["ds"].iloc[-1],
            color="khaki",
            alpha=0.2
        )

        # English titles / labels
        english_title = {
            "매출안정성지표": "Sales Stability Index",
            "경쟁우위 지표": "Competitive Advantage Index",
            "고객 충성도 지표": "Customer Loyalty Index"
        }.get(target, target)

        ax.set_title(f"📈 {english_title} (Prophet Forecast)")
        ax.set_xlabel("Month")
        ax.set_ylabel("KPI Value")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()

        # ✅ Save results
        results.append({
            "Model": "Prophet",
            "Indicator": english_title,
            "Forecast Mean": y_pred.mean(),
            "MAE": mae,
            "RMSE": rmse,
            "MAPE(%)": mape,
            "Warning Threshold": warn_th,
            "Danger Threshold": danger_th,
            "fig": fig
        })

    # If no results
    if not results:
        print("⚠️ No forecast results available.")
        return pd.DataFrame(columns=[
            "Model", "Indicator", "Forecast Mean", "MAE", "RMSE", "MAPE(%)",
            "Warning Threshold", "Danger Threshold"
        ])

    print(f"✅ {len(results)} KPI forecasts completed (English only)")
    return pd.DataFrame(results)












