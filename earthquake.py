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
    layout="wide" # 가로로 넓게 쓰기 위해 설정
)

# 2. 파일 경로 설정
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(CURRENT_DIR, "Earthquakes.csv")
MODEL_PATH = os.path.join(CURRENT_DIR, "model.pkl")
SCALER_PATH = os.path.join(CURRENT_DIR, "scaler.pkl")

# 3. 데이터 로드 함수
@st.cache_data
def load_data():
    if not os.path.exists(CSV_PATH):
        st.error(f"❌ 파일을 찾을 수 없습니다: {CSV_PATH}")
        return None
    
    # 시도할 인코딩 목록 확장
    encodings = ['utf-8', 'utf-8-sig', 'cp949', 'euc-kr', 'latin1', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            df = pd.read_csv(CSV_PATH, encoding=encoding)
            st.success(f"✅ '{encoding}' 인코딩으로 파일을 읽었습니다.")
            return df
        except (UnicodeDecodeError, Exception):
            continue
    
    # 최후 수단: errors='replace'로 강제 읽기
    # 최후 수단: errors='replace'로 강제 읽기
    try:
        df = pd.read_csv(CSV_PATH, encoding='utf-8', errors='replace')
        st.warning("⚠️ 일부 글자가 깨질 수 있습니다 (강제 읽기 모드).")
        return df
    except Exception as e:
        st.error(f"❌ 파일 읽기 최종 실패: {e}")
        return None

# ✅ 여기에 추가
df_new = load_data()
st.write("CSV 경로:", CSV_PATH)
st.write("파일 존재 여부:", os.path.exists(CSV_PATH))
if df_new is None:
    st.error("❌ 파일을 읽을 수 없습니다.")
    st.stop()

# 4. 세션 상태 초기화
if 'clicked' not in st.session_state:
    st.session_state.clicked = False

# --- 🛠️ 블랙 & 네온 CSS 스타일링 ---
st.markdown("""
    <style>
        /* 배경을 완전한 블랙으로 고정 */
        .stApp {
            background-color: #000000 !important;
        }
        /* 모든 텍스트 색상을 밝게 */
        h1, h2, h3, p, span, label {
            color: #FFFFFF !important;
            font-family: 'Courier New', Courier, monospace !important;
        }
        /* 입력창 라벨 색상 */
        .stNumberInput label {
            color: #00F0FF !important;
            font-weight: bold;
        }
        /* 사이버펑크 타이틀 */
        .cyber-title {
            font-size: 2.5rem !important;
            font-weight: 900 !important;
            color: #00F0FF !important;
            text-shadow: 0 0 10px #00F0FF;
            text-align: center;
            margin-bottom: 0px;
        }
        /* 결과 카드 디자인 */
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

# 5. 상단 헤더 구역
st.markdown('<p class="cyber-title">⚡ BLACK-MATRIX SEISMIC GRID</p>', unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#FF007F !important;'>SYSTEM STATUS: ONLINE // NEURAL-LINK ACTIVE</p>", unsafe_allow_html=True)

# 6. [상단 배치] 입력 UI (사이드바 대신 메인 상단 컬럼 사용)
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

# 7. 메인 레이아웃 (좌측 결과, 우측 지도)
if st.session_state.clicked:
    # 화면 분할
    col_left, col_right = st.columns([1, 2]) # 1:2 비율

    with col_left:
        # --- 🛠️ 디버그: 실제 컬럼명 확인 ---
        st.write("📋 CSV 컬럼 목록:", df_new.columns.tolist())
        st.write("📄 첫 3행 미리보기:", df_new.head(3))

        # --- AI 분석 로직 ---
        # ✅ 컬럼명 자동 감지 (한글/영어 둘 다 대응)
        col_lat  = '위도'       if '위도'       in df_new.columns else 'latitude'
        col_lon  = '경도'       if '경도'       in df_new.columns else 'longitude'
        col_mag  = '규모'       if '규모'       in df_new.columns else 'magnitude'
        col_imp  = '영향도'     if '영향도'     in df_new.columns else 'impact'
        col_dep  = '진원깊이'   if '진원깊이'   in df_new.columns else 'depth'
        col_cls  = 'cluster'

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
                
                avg_magnitude = near_df[col_mag].mean() if col_mag in near_df.columns else 3.0
                avg_impact    = near_df[col_imp].mean() if col_imp in near_df.columns else 1.0
                avg_depth     = near_df[col_dep].mean() if col_dep in near_df.columns else 10.0
                
                temp_dict = {'위도': lat, '경도': lon, '규모': avg_magnitude, '영향도': avg_impact, '진원깊이': avg_depth}
                model_cols = list(scaler.feature_names_in_)
                pred_features = pd.DataFrame([temp_dict])[model_cols]
                
                features_scaled = scaler.transform(pred_features)
                predicted_cluster = model.predict(pd.DataFrame(features_scaled, columns=model_cols))[0]
                
                main_cluster = int(predicted_cluster)
                risk_label = risk_dict.get(main_cluster, "UNKNOWN")
                style = card_styles.get(main_cluster, {"text": "#FFFFFF", "shadow": "#FFFFFF"})

                st.markdown(f"""
                    <div class="result-box" style="border-color: {style['text']}; box-shadow: 0 0 20px {style['shadow']};">
                        <p style="font-size:0.9rem; color:#888 !important;">// ANALYSIS RESULT</p>
                        <h2 style="color:{style['text']} !important; text-shadow: 0 0 10px {style['text']}; font-size:2.2rem;">
                            {risk_label}
                        </h2>
                        <p style="color:#555 !important; font-size:0.8rem; margin-top:20px;">Grid Coords: {lat}, {lon}</p>
                    </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"MATRIX ERROR: {e}")

    with col_right:
        # 우측 지도 출력
        m = folium.Map(location=[lat, lon], zoom_start=5, tiles="CartoDB dark_matter")
        
        sample_df = df_new.sample(n=min(1000, len(df_new)), random_state=42)
        for _, row in sample_df.iterrows():
            c_idx = int(row['cluster']) if 'cluster' in row else 0
            m_color = colors.get(c_idx, '#555')
            folium.CircleMarker(
                location=[row['위도'], row['경도']],
                radius=3, color=m_color, fill=True, fill_color=m_color, fill_opacity=0.7
            ).add_to(m)
            
        folium.Marker(
            location=[lat, lon],
            icon=folium.Icon(color='darkpurple', icon='star', icon_color='#39FF14')
        ).add_to(m)
        
        st_folium(m, width="100%", height=600, key="main_map")

else:
    # 초기 화면 안내 (전체 너비 사용)
    st.markdown("""
        <div style="margin-top:100px; text-align:center; border:1px dashed #444; padding:50px;">
            <h3 style="color:#444 !important;">AWAITING SCAN COMMAND...</h3>
            <p style="color:#333 !important;">상단에 위도와 경도를 입력하고 RUN SCAN 버튼을 누르십시오.</p>
        </div>
    """, unsafe_allow_html=True)
