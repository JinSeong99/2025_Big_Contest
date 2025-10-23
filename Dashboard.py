import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import matplotlib.font_manager as fm

from prophet_functions import evaluate_forecast_model_prophet, last_df, threshold_df

# -----------------------------
# 📄 기본 설정 (한글 폰트 + 스타일)
# -----------------------------

# 한글 폰트 설정

# 1️⃣ 시스템 폰트 디렉터리에서 나눔고딕 폰트 경로 직접 탐색
font_paths = fm.findSystemFonts(fontpaths=['/usr/share/fonts', '/usr/local/share/fonts'])

nanum_fonts = [f for f in font_paths if 'Nanum' in f or 'nanum' in f]
if nanum_fonts:
    font_path = nanum_fonts[0]
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = fm.FontProperties(fname=font_path).get_name()
else:
    # fallback: 기본 sans-serif
    plt.rcParams['font.family'] = 'DejaVu Sans'

# 2️⃣ 마이너스 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False

# 3️⃣ 캐시 재로드
try:
    fm._rebuild()
except Exception:
    pass

# 4️⃣ Seaborn & Streamlit 기본 스타일
sns.set_style("whitegrid")
st.set_page_config(page_title="KPI 예측 대시보드", layout="wide")
# -----------------------------
# 📌 헤더
# -----------------------------
st.markdown(
    """
    <h1 style='text-align:center; color:#2E86C1;'>📈 KPI 예측 대시보드</h1>
    <p style='text-align:center; color:#555;'>
    생존한 가맹점의 평균 예측 결과와 자신의 가맹점의 미래 위험 상태를 함께 확인하세요.<br>
    
    </p>
    """,
    unsafe_allow_html=True
)

# ======================================
# ⚙️ 1️⃣ Prophet 예측 결과 생성 (기존 코드 유지)
# ======================================
st.info("🔄 예측 실행 중입니다...")
results_df = evaluate_forecast_model_prophet(last_df, threshold_df)

if results_df is None or results_df.empty or "지표" not in results_df.columns:
    st.error("⚠️ 예측 결과를 불러오지 못했습니다. 데이터나 임계치 구성이 올바른지 확인하세요.")
    st.stop()

indicator_list = results_df["지표"].tolist()
selected_indicator = st.radio(
    "📊 지표 선택",
    indicator_list,
    horizontal=True,
    key="indicator_select",
    label_visibility="visible"
)

# ======================================
# ⚙️ 2️⃣ 가맹점 상태 파일 불러오기
# ======================================


# 같은 폴더 내에 있는 CSV 파일 경로 지정
csv_path = "result_prophet_storewise.csv"  


try:
    store_df = pd.read_csv(csv_path)
    store_df.columns = [c.strip() for c in store_df.columns]  # 공백 제거
    st.success(f"✅ 가맹점 상태 데이터 불러오기 완료")
    st.caption(f"평균 가맹점 KPI와 맞춤형 KPI를 비교해보세요!")
except Exception as e:
    st.error(f"❌ 가맹점 상태 데이터를 불러올 수 없습니다: {e}")
    store_df = pd.DataFrame(columns=['가맹점', '지표', '미래상태'])

# ======================================
# ⚙️ 3️⃣ 메인 레이아웃 (왼쪽 Prophet 시각화 + 오른쪽 가맹점 상태)
# ======================================
col1, col2 = st.columns([1.6, 1])  # 왼쪽 시각화 넓게, 오른쪽 검색 좁게

# -----------------------------
# ⬅️ 왼쪽 Prophet 시각화 (크기 축소)
# -----------------------------
with col1:
    st.markdown(f"### 🔍 선택한 지표: **{selected_indicator}**")

    indicator_row = results_df[results_df["지표"] == selected_indicator].iloc[0]
    mae = indicator_row["MAE"]
    rmse = indicator_row["RMSE"]
    warn = indicator_row.get("경고임계치", None)
    danger = indicator_row.get("위험임계치", None)

    # Prophet 시각화 (크기 축소)
    if "fig" in indicator_row and indicator_row["fig"] is not None:
        fig = indicator_row["fig"]
        fig.set_size_inches(6, 3)  # 🔹 시각화 축소
        st.pyplot(fig, use_container_width=False)
    else:
        st.warning("⚠️ 그래프 객체(fig)가 없습니다. Prophet 함수가 fig를 반환하도록 수정하세요.")

    # 성능 요약 카드
    st.markdown("### 📊 예측 성능 요약")
    c1, c2, c3 = st.columns(3)
    c1.metric("📏 MAE", f"{mae:.3f}")
    c2.metric("📐 RMSE", f"{rmse:.3f}")
    if warn and danger:
        c3.markdown(
            f"""
            <div style='background-color:#f9f9f9; padding:10px; border-radius:10px; text-align:center;'>
                <b>⚠️ 임계치</b><br>
                <span style='color:orange;'>경고: {warn:.3f}</span><br>
                <span style='color:red;'>위험: {danger:.3f}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

# -----------------------------
# ➡️ 오른쪽 : 가맹점 ID 검색 + 미래 상태 표시
# -----------------------------
with col2:
    st.markdown("### 🏪 내 가맹점 미래 상태 조회")

    store_id_input = st.text_input("가맹점 ID 입력", placeholder="예: 000F03E44A")

    if store_id_input:
        filtered = store_df[
            store_df["가맹점"].astype(str).str.contains(store_id_input.strip(), case=False, na=False)
        ]

        if filtered.empty:
            st.warning("❌ 해당 가맹점 ID를 찾을 수 없습니다.")
        else:
            st.success(f"✅ {store_id_input} 검색 결과 ({len(filtered)}개 지표)")
            st.markdown("---")

            # 색상 매핑
            color_map = {
                "안전": "#1f77b4",  # 파랑
                "경고": "#ffbf00",  # 노랑
                "위험": "#d62728"   # 빨강
            }

            # 각 지표별 미래 상태 표시
            for _, row in filtered.iterrows():
                indicator = str(row["지표"]).strip()
                status = str(row["미래상태"]).strip()
                color = color_map.get(status, "#555")
                st.markdown(
                    f"""
                    <div style='background-color:{color}22;
                                border-left:5px solid {color};
                                padding:10px;
                                border-radius:8px;
                                margin-bottom:8px;'>
                        <b style='color:{color}; font-size:16px;'>{indicator}</b><br>
                        <span style='color:{color}; font-weight:bold; font-size:18px;'>{status}</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    else:
        st.info("🔍 오른쪽 입력창에 가맹점 ID를 입력하면 상태가 표시됩니다.")

# ======================================
# ⚙️ 4️⃣ 하단 문구
# ======================================
st.markdown(
    """
    <hr style='margin-top:40px; margin-bottom:10px;'>
    <p style='text-align:center; color:gray; font-size:13px;'>
    © 2025 Prophet Forecast Visualization | Streamlit Dashboard by Data Analyst
    </p>
    """,
    unsafe_allow_html=True
)








