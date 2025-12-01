import streamlit as st
import pandas as pd
import numpy as np
import io

# Streamlit 앱의 제목 설정
st.title("🏃‍♀️ 체력 데이터 상관관계 분석기")
st.markdown("---")

# 파일 로드 및 전처리 함수
@st.cache_data
def load_and_preprocess_data(uploaded_file):
    """업로드된 CSV 파일을 로드하고 분석에 필요한 전처리를 수행합니다."""
    # 파일 로드 (첫 번째 행을 헤더로 사용)
    df = pd.read_csv(uploaded_file)
    
    # 1. 컬럼 이름 정리: 불필요한 공백/특수 문자 제거 및 영문으로 변경 (분석의 편의성)
    # 한글 컬럼명이 복잡하므로, 분석에 사용할 주요 컬럼을 식별하고 전처리합니다.
    # 사용자가 업로드한 데이터의 스니펫을 기반으로 주요 체력 측정 항목을 선택하고 전처리합니다.
    
    # 분석에 사용할 수치형 컬럼 목록
    # 스니펫에 나타난 한글 컬럼명을 기반으로 코드를 작성합니다.
    # CSV 파일이므로 엑셀의 빈 칸이나 특수문자를 포함할 수 있어, 정확한 컬럼명 확인이 필요합니다.
    # 스니펫을 바탕으로 주요 컬럼을 선택합니다.
    
    columns_to_analyze = [
        '신장', '체중', '체지방율', '허리둘레', 
        '악력_좌', '악력_우', '윗몸말아올리기', '반복점프',
        '앉아윗몸앞으로굽히기', '제자리 멀리뛰기', 'BMI', 
        '상대악력', '허리둘레-신장비'
    ]
    
    # 데이터프레임에 분석 컬럼이 모두 있는지 확인
    missing_cols = [col for col in columns_to_analyze if col not in df.columns]
    if missing_cols:
        st.error(f"⚠️ **오류:** 데이터 파일에 다음 컬럼들이 누락되었습니다: {', '.join(missing_cols)}")
        st.stop()
        
    # 2. 수치형 데이터만 선택 및 결측치 처리
    df_analysis = df[columns_to_analyze].copy()
    
    # 모든 값을 숫자로 변환 (변환할 수 없는 값은 NaN으로 설정)
    for col in df_analysis.columns:
        df_analysis[col] = pd.to_numeric(df_analysis[col], errors='coerce')
        
    # 결측치가 너무 많은 행은 제거 (선택적)
    df_analysis = df_analysis.dropna()
    
    if df_analysis.empty:
        st.error("⚠️ **오류:** 전처리 후 분석에 사용할 수 있는 데이터가 없습니다. 원본 데이터에 결측치나 비수치형 값이 많을 수 있습니다.")
        st.stop()
        
    return df_analysis

# --- 데이터 로드 ---
# 사용자가 올린 파일을 메모리 내에서 로드
# 로컬 파일이 아닌, 사용자 업로드 파일을 사용하는 방식
# Streamlit의 file_uploader 기능을 사용하여 사용자에게 파일 업로드를 요청합니다.

# 단, 깃허브에 추가할 코드이므로, 사용자가 직접 파일을 업로드하는 방식으로 구현합니다.
uploaded_file_name = "fitness data.xlsx - KS_NFA_FTNESS_MESURE_ITEM_MESUR.csv"

# 로컬 환경 테스트용 (주석 처리): 
# try:
#     # 깃허브에 올릴 때 테스트할 수 있도록 로컬 파일을 읽는 코드
#     df_original = pd.read_csv(uploaded_file_name)
# except FileNotFoundError:
#     st.warning("⚠️ **주의:** 이 코드는 깃허브에 추가될 때 'fitness data.xlsx - KS_NFA_FTNESS_MESURE_ITEM_MESUR.csv' 파일이 같은 폴더에 있다고 가정합니다.")
#     st.warning("로컬에서 실행 시 파일을 찾을 수 없어 임시 데이터로 대체합니다. 실제 분석을 위해 파일을 업로드하거나 같은 경로에 위치시켜 주세요.")
#     # 임시 데이터 생성 (파일이 없을 경우 대비)
#     data = {'신장': [170, 180, 165, 175], '체중': [70, 80, 60, 75], '윗몸말아올리기': [40, 50, 30, 45], '제자리 멀리뛰기': [200, 250, 180, 230]}
#     df_original = pd.DataFrame(data)
#     df_original['체지방율'] = np.random.rand(4) * 20
#     # ... 임시 컬럼 추가 필요
    
# uploaded_file = uploaded_file_name # 실제 파일 경로 대신 파일명 문자열 사용
# df_analysis = load_and_preprocess_data(uploaded_file) # 로컬 파일 읽기 시도

# 깃허브/배포 환경을 위한 Streamlit 파일 업로더
uploaded_file = st.file_uploader(
    "📤 CSV 파일 업로드",
    type="csv",
    help="분석할 'fitness data.xlsx - KS_NFA_FTNESS_MESURE_ITEM_MESUR.csv' 파일을 업로드해주세요."
)

if uploaded_file is None:
    st.info("⬆️ 분석을 시작하려면 CSV 파일을 업로드해주세요.")
else:
    # 파일을 메모리에 로드
    df_analysis = load_and_preprocess_data(uploaded_file)
    
    st.subheader("📊 전처리된 데이터 미리보기")
    st.dataframe(df_analysis.head())
    st.write(f"**분석 데이터 크기:** {df_analysis.shape[0]} 행, {df_analysis.shape[1]} 열")
    st.markdown("---")
    
    # 3. 상관관계 분석
    corr_matrix = df_analysis.corr()
    
    # 4. 상관관계 데이터를 Series로 변환 후 정렬
    # 자기 자신과의 상관관계 (값=1)와 중복 쌍을 제거
    corr_unstacked = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)).stack().sort_values(ascending=False)
    
    st.subheader("🔎 상관관계 분석 결과")
    
    # --- 양의 상관관계 버튼 ---
    if st.button("➕ 가장 높은 **양의 상관관계** 데이터 쌍 보기"):
        if not corr_unstacked.empty:
            # 상위 5개 출력
            st.success("🥇 **가장 높은 양의 상관관계 (Top 5)**")
            st.table(
                corr_unstacked.head(5).rename("상관계수").reset_index().rename(columns={'level_0': '변수 1', 'level_1': '변수 2'})
            )
            st.write("상관계수가 **1**에 가까울수록 두 변수는 **강한 비례 관계**를 가집니다.")
        else:
            st.warning("분석할 수 있는 상관관계 데이터 쌍이 없습니다.")
            
    # --- 음의 상관관계 버튼 ---
    if st.button("➖ 가장 높은 **음의 상관관계** 데이터 쌍 보기"):
        if not corr_unstacked.empty:
            # 하위 5개 출력 (가장 음의 상관관계가 강한 것, 즉 -1에 가까운 것)
            st.error("📉 **가장 높은 음의 상관관계 (Bottom 5)**")
            st.table(
                corr_unstacked.tail(5).rename("상관계수").reset_index().rename(columns={'level_0': '변수 1', 'level_1': '변수 2'})
            )
            st.write("상관계수가 **-1**에 가까울수록 두 변수는 **강한 반비례 관계**를 가집니다.")
        else:
            st.warning("분석할 수 있는 상관관계 데이터 쌍이 없습니다.")
            
    st.markdown("---")
    st.info("💡 **상관계수 (Correlation Coefficient)**:\n"
            "* **+1**: 완벽한 양의 상관관계 (비례)\n"
            "* **0**: 상관관계 없음\n"
            "* **-1**: 완벽한 음의 상관관계 (반비례)")

    # --- 전체 상관관계 히트맵 (선택 사항) ---
    st.subheader("🎨 전체 상관관계 히트맵 (시각화)")
    
    # 시각화 라이브러리 (matplotlib, seaborn) 사용 예시
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', ax=ax, cbar_kws={'label': '상관계수'})
        ax.set_title('데이터 변수 간 상관관계 히트맵')
        st.pyplot(fig)
        
        st.info("➡️ 히트맵을 통해 모든 변수 쌍의 상관관계를 한눈에 확인할 수 있습니다.")
        
    except ImportError:
        st.warning("⚠️ **경고:** 시각화를 위해 `matplotlib`와 `seaborn` 라이브러리가 필요합니다. `requirements.txt`에 추가했습니다.")
