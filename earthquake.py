import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import os
import joblib

# 1. 페이지 설정
st.set_page_config(
    page_title="BLACK MATRIX AI", 
    page_icon="🌌",
    layout="wide"
)

# 2. 파일 경로 설정
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(CURRENT_DIR, "Earthquakes.csv")
MODEL_PATH = os.path.join(CURRENT_DIR, "model.pkl")
SCALER_PATH = os.path.join(CURRENT_DIR, "scaler.pkl")

# 3. 데이터 로드 함수 (joblib 바이너리로 읽기)
@st.cache_data
def load_data():
    if not os.path.exists(CSV_PATH):
        return None, f"파일 없음: {CSV_PATH}"
    try:
        df = joblib.load(CSV_PATH)
        return df, "✅ 데이터 로드 성공"
    except Exception as e:
        return None, f"❌ 로드 실패: {e}"

# 4. 데이터 로드 실행
df_new, load_msg = load_data()

# 5. CSS 스타일링
st.markdown("""
    <style>
        .stApp { background-color: #000000 !important; }
        h1, h2, h3, p, span, label {
            color: #FFFFFF !important;
            font-family: 'Courier New', Courier, monospace !important;
        }
        .stNumberInput label { color: #00F0FF !important; font-weight: bold; }
        .cyber-title {
            font-size: 2.5rem !important;
            font-weight: 900 !important;
            color: #00F0FF !important;
            text-shadow: 0 0 10px #00F0FF;
            text-align: center;
            margin-bottom: 0px;
        }
        .result-box {
            background-color: #080808;
            border: 2px solid #FF007F;
            border-radius: 10px;
            padding: 40px 20px;
            text-align: center;
            box-shadow: 0 0 20px #FF007F;
            margin-top: 50px;
        }
    </style>
""", unsafe_allow_html=True)

# 6. 헤더
st.markdown('<p class="cyber-title">⚡ BLACK-MATRIX SEISMIC GRID</p>', unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#FF007F !important;'>SYSTEM STATUS: ONLINE // NEURAL-LINK ACTIVE</p>", unsafe_allow_html=True)

# 파일 로드 실패 시 중단
if df_new is None:
    st.error(f"❌ 데이터 로드 실패 — {load_msg}")
    st.stop()

# 7. 컬럼명 (확인 완료: 한글)
col_lat = '위도'
col_lon = '경도'
col_mag = '규모'
col_imp = '영향도'
col_dep = '진원깊이'
col_cls = 'cluster' if 'cluster' in df_new.columns else None

# 8. 세션 상태 초기화
if 'clicked' not in st.session_state:
    st.session_state.clicked = False

# 9. 입력 UI
st.markdown("<div style='background-color:#111; padding:20px; border-radius:10px; border:1px solid #333;'>", unsafe_allow_html=True)
input_col1, input_col2, input_col3 = st.columns([2, 2, 1])
with input_col1:
    lat = st.number_input("TARGET LATITUDE (위도)", value=37.5665, format="%.4f")
with input_col2:
    lon = st.number_input("TARGET LONGITUDE (경도)", value=126.9780, format="%.4f")
with input_col3:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("RUN SCAN", use_container_width=True):
        st.session_state.clicked = True
st.markdown("</div>", unsafe_allow_html=True)

# 설정값
risk_dict = {0: 'CRITICAL (높음)', 1: 'STABLE (낮음)', 2: 'WARNING (중간)'}
colors = {0: '#FF0055', 1: '#00F0FF', 2: '#39FF14'}
card_styles = {
    0: {"text": "#FF0055", "shadow": "#FF0055"},
    1: {"text": "#00F0FF", "shadow": "#00F0FF"},
    2: {"text": "#39FF14", "shadow": "#39FF14"}
}

# 10. 메인 레이아웃
if st.session_state.clicked:
    col_left, col_right = st.columns([1, 2])

    with col_left:
        near_df = df_new[
            (df_new[col_lat] >= lat - 5) & (df_new[col_lat] <= lat + 5) &
            (df_new[col_lon] >= lon - 5) & (df_new[col_lon] <= lon + 5)
        ]

        if near_df.empty:
            st.error("🚨 NO DATA IN SCAN RANGE")
        else:
            try:
                model = joblib.load(MODEL_PATH)
                scaler = joblib.load(SCALER_PATH)

                avg_magnitude = near_df[col_mag].mean()
                avg_impact    = near_df[col_imp].mean()
                avg_depth     = near_df[col_dep].mean()

                model_cols = list(scaler.feature_names_in_)
                temp_dict = {
                    col_lat: lat, col_lon: lon,
                    col_mag: avg_magnitude,
                    col_imp: avg_impact,
                    col_dep: avg_depth
                }
                pred_features = pd.DataFrame([temp_dict])[model_cols]

                features_scaled = scaler.transform(pred_features)
                predicted_cluster = model.predict(pd.DataFrame(features_scaled, columns=model_cols))[0]

                main_cluster = int(predicted_cluster)
                risk_label = risk_dict.get(main_cluster, "UNKNOWN")
                style = card_styles.get(main_cluster, {"text": "#FFFFFF", "shadow": "#FFFFFF"})

                st.markdown(f"""
                    <div class="result-box" style="border-color:{style['text']}; box-shadow:0 0 20px {style['shadow']};">
                        <p style="font-size:0.9rem; color:#888 !important;">// ANALYSIS RESULT</p>
                        <h2 style="color:{style['text']} !important; text-shadow:0 0 10px {style['text']}; font-size:2.2rem;">
                            {risk_label}
                        </h2>
                        <p style="color:#555 !important; font-size:0.8rem; margin-top:20px;">Grid Coords: {lat}, {lon}</p>
                    </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"MATRIX ERROR: {e}")

    with col_right:
        m = folium.Map(location=[lat, lon], zoom_start=5, tiles="CartoDB dark_matter")

        sample_df = df_new.sample(n=min(1000, len(df_new)), random_state=42)
        for _, row in sample_df.iterrows():
            c_idx = int(row[col_cls]) if col_cls and col_cls in row else 0
            m_color = colors.get(c_idx, '#555')
            folium.CircleMarker(
                location=[row[col_lat], row[col_lon]],
                radius=3, color=m_color, fill=True, fill_color=m_color, fill_opacity=0.7
            ).add_to(m)

        folium.Marker(
            location=[lat, lon],
            icon=folium.Icon(color='darkpurple', icon='star', icon_color='#39FF14')
        ).add_to(m)

        st_folium(m, width="100%", height=600, key="main_map")

else:
    st.markdown("""
        <div style="margin-top:100px; text-align:center; border:1px dashed #444; padding:50px;">
            <h3 style="color:#444 !important;">AWAITING SCAN COMMAND...</h3>
            <p style="color:#333 !important;">상단에 위도와 경도를 입력하고 RUN SCAN 버튼을 누르십시오.</p>
        </div>
    """, unsafe_allow_html=True)
