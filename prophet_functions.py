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
FONT_URL = "https://github.com/naver/nanumfont/blob/master/NanumGothic.ttf?raw=true"
FONT_PATH = "/tmp/NanumGothic.ttf"

try:
    # 🔹 GitHub에서 폰트 다운로드
    if not os.path.exists(FONT_PATH):
        print("📥 NanumGothic 폰트 다운로드 중...")
        r = requests.get(FONT_URL)
        r.raise_for_status()
        with open(FONT_PATH, "wb") as f:
            f.write(r.content)

    # 🔹 Matplotlib에 폰트 등록
    fm.fontManager.addfont(FONT_PATH)
    plt.rcParams["font.family"] = "NanumGothic"
    plt.rcParams["axes.unicode_minus"] = False

    # ✅ 폰트 캐시 재빌드 (Cloud 환경에서 중요!!)
    try:
        fm._rebuild()
    except Exception:
        pass

    # ✅ 등록된 폰트 확인 로그
    print("✅ NanumGothic 폰트 등록 및 캐시 재생성 완료")

except Exception as e:
    print(f"⚠️ 폰트 로드 실패: {e}")
    plt.rcParams["font.family"] = "DejaVu Sans"

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
    Prophet 기반 시계열 예측 (생존 + 폐업 전 구간 포함)
    - last_df: 가맹점별 지표 및 폐업여부 포함
    - 각 지표별 Prophet 학습 → 예측 → 임계치 비교 및 시각화
    """

    df = last_df.copy()
    df["ds"] = pd.to_datetime(df["기준년월"], format="%Y%m")

    # 생존 / 폐업 데이터 분리
    alive_df = df[df["폐업여부"] == 0].copy()
    closed_df = df[df["폐업여부"] == 1].copy()

    # 폐업 매장: 폐업 전 n개월만 포함
    closed_pre = (
        closed_df.sort_values(["가맹점구분번호", "ds"])
        .groupby("가맹점구분번호", group_keys=False)
        .apply(lambda g: g.tail(pre_close_months))
    )

    total_df = pd.concat([alive_df, closed_pre], axis=0).reset_index(drop=True)
    total_df = total_df.sort_values(["가맹점구분번호", "ds"])

    indicators = ["매출안정성지표", "경쟁우위 지표", "고객 충성도 지표"]
    results = []

    # ==============================
    # Prophet 학습 및 예측 (지표별)
    # ==============================
    for target in indicators:
        key = _norm(target)
        matched_idx = _idx_map.get(key, None)

        if matched_idx is None:
            print(f"⚠️ 임계치 테이블에서 '{target}'(정규화='{key}') 찾지 못함 → 스킵")
            continue

        # 컬럼 존재 확인
        if target not in total_df.columns:
            alt = [c for c in total_df.columns if _norm(c) == key]
            if alt:
                target = alt[0]
            else:
                print(f"⚠️ last_df에 '{target}' 컬럼이 없어 스킵합니다.")
                continue

        sub = total_df[["ds", target]].dropna().sort_values("ds").copy()
        if len(sub) < 10:
            print(f"⚠️ '{target}' 데이터가 충분하지 않아 스킵합니다. (len={len(sub)})")
            continue

        prophet_df = sub.rename(columns={target: "y"})

        # Prophet 모델 생성 및 학습
        m = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            changepoint_prior_scale=1
        )
        m.fit(prophet_df)

        future = m.make_future_dataframe(periods=forecast_months, freq="MS")
        forecast = m.predict(future)

        # 성능 계산
        y_true = prophet_df["y"].iloc[-min(forecast_months, len(prophet_df)):]
        y_pred = forecast["yhat"].iloc[-min(forecast_months, len(forecast)):]
        mae, rmse, mape = evaluate_forecast(y_true, y_pred)

        # 임계치 값 읽기 (float 변환 보장)
        warn_th = float(threshold_df.loc[matched_idx, warn_col])
        danger_th = float(threshold_df.loc[matched_idx, danger_col])

        # 시각화
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(forecast["ds"], forecast["yhat"], color="#1f77b4", label="예측 추세")
        ax.axhline(y=warn_th, color="orange", linestyle="--", label=f"경고 {warn_th:.3f}")
        ax.axhline(y=danger_th, color="red", linestyle="--", label=f"위험 {danger_th:.3f}")
        ax.axvspan(
            forecast["ds"].iloc[-forecast_months],
            forecast["ds"].iloc[-1],
            color="khaki",
            alpha=0.2
        )
        ax.set_title(f"📈 {target} Prophet 예측")
        ax.set_xlabel("기준년월")
        ax.set_ylabel(target)
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()

        # ✅ Streamlit 환경에서 표시할 수 있게 반환
        results.append({
            "모델": "Prophet",
            "지표": target,
            "예측 평균": y_pred.mean(),
            "MAE": mae,
            "RMSE": rmse,
            "MAPE(%)": mape,
            "경고임계치": warn_th,
            "위험임계치": danger_th,
            "fig": fig  #장
        })

    # 빈 결과 방지
    if not results:
        print("⚠️ 예측 결과가 비어 있습니다. (데이터 또는 임계치 불일치 가능)")
        return pd.DataFrame(columns=[
            "모델", "지표", "예측 평균", "MAE", "RMSE", "MAPE(%)", "경고임계치", "위험임계치"
        ])

    print(f"✅ {len(results)}개의 지표 예측 완료")
    return pd.DataFrame(results)








