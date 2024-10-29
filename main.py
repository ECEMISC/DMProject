import streamlit as st
import pandas as pd
from catboost import CatBoostClassifier

import base64

st.set_page_config(
    page_title="Attrition Predictor",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="auto"
)


# CSS ile dairesel resimler ve arka plan ekliyoruz

st.markdown(
    """
    <style>
    .stApp {
        background-color: #D3D3D3;
        background-image: linear-gradient(to bottom, #ffffff, #D3D3D3);
    }
    </style>
    """,
    unsafe_allow_html=True
)


def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")
    return base64_image

image_base64 = image_to_base64("arkaplan.JPG")

# CSS ile sidebar başlığını küçültüyoruz

# Sidebar içeriğine yukarıdan boşluk ekliyoruz
st.sidebar.markdown("""
    <style>
    .sidebar-content {
        margin-top: 53px;  /* Yukarıdan boşluk bırakmak için */
    }
    </style>
    <div class="sidebar-content">
    """, unsafe_allow_html=True)


st.sidebar.markdown("""
    <style>
    .sidebar-title {
        font-size: 19px;
        font-weight: bold;
        color: black;
    }
    </style>
    <div class="sidebar-title">Page Selection</div>
    """, unsafe_allow_html=True)


# Sidebar'a seçici ekliyoruz
page = st.sidebar.selectbox("Choose a Page", ["Project Information", "Prediction"])





# Arka plan resmi eklemek için CSS
st.markdown(
    f"""
    <style>
    [data-testid="stSidebar"] {{
        background-image: url("data:image/jpeg;base64,{image_base64}");
        background-size: cover;
    }}
    </style>
    """,
    unsafe_allow_html=True
)


# Proje Bilgileri sayfası
if page == "Project Information":
    # Büyük başlık: HR Attrition Predictor
    st.markdown("<h1 style='text-align: left; color: black; font-size: 30px;'>Attrition Predictor</h1>",
                unsafe_allow_html=True)
    # Küçük başlık: A Data-Driven Approach to Workforce Stability
    st.markdown("<h4 style='text-align: left; color: grey;'>A Data-Driven Approach to Workforce Stability</h4>",
                unsafe_allow_html=True)


    # Takım üyeleri ve fotoğraflar
    st.markdown("<h4 style='text-align: left; color: black;'>Our Team</h4>", unsafe_allow_html=True)


    # 3 üstte, 2 altta olacak şekilde fotoğrafları ve isimleri yerleştirme
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("ecem.png", width=150, caption="Ecem Zeynep Iscanli")
    with col2:
        st.image("kosmas.png", width=150, caption="Kosmas Konomis")
    with col3:
        st.image("shivesh.png", width=150, caption="Shevish")

    col_left, col_right = st.columns([0.5, 1])

    with col_left:
        st.image("rhythm.png", width=150, caption="Rhythm Chaudhary")
    with col_right:
        st.image("ira.png", width=150, caption="Ira Ira")

    # Proje adı ve amacı
    st.subheader("Project Information")
    st.write("This project aims to predict employee attrition using advanced machine learning techniques. Attrition, also known as employee turnover, is a critical issue for organizations as it can lead to increased costs, loss of talent, and a negative impact on team morale. By predicting which employees are most likely to leave the company, HR departments can take proactive measures to improve retention, address employee concerns, and ensure workforce stability.")

    st.subheader("Project Purpose")
    st.write("""
    This project uses a CatBoost model to predict the likelihood of employee attrition based on factors like salary, job satisfaction, and work-life balance. By identifying employees at risk of leaving, the model provides HR teams with insights to improve retention strategies, reduce turnover costs, and make more informed decisions to retain key talent.
    """)

# Tahmin Sayfası
elif page == "Prediction":
    st.title("Attrition Predictior")

    # Modelin eğitildiği özellikleri öğrenelim (modeldeki sütunlar)
    Cat_model = CatBoostClassifier()
    Cat_model.load_model('catboost_model.cbm')  # Daha önce kaydedilen modeli yükle
    model_features = Cat_model.feature_names_  # Modelin beklediği tüm sütunlar

    # Başlık boyutunu küçültmek için HTML kullanıyoruz
    st.markdown("<h3 style='text-align: left; color: black;'>Employee Data</h3>", unsafe_allow_html=True)

    # Kullanıcıdan alınacak girdiler
    age = st.number_input("Age", min_value=18, max_value=65, value=30)
    distance_from_home = st.number_input("Distance From Home", min_value=0, max_value=100, value=5)
    monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=20000, value=5000)
    years_at_company = st.number_input("Years At Company", min_value=0, max_value=40, value=5)
    years_in_current_role = st.number_input("Years In Current Role", min_value=0, max_value=20, value=5)
    years_since_last_promotion = st.number_input("Years Since Last Promotion", min_value=0, max_value=20, value=2)
    overtime = st.selectbox("Overtime", ['Yes', 'No'])
    environment_satisfaction = st.radio("Environment Satisfaction (1-4)", options=[1, 2, 3, 4], index=2)
    job_involvement = st.radio("Job Involvement (1-4)", options=[1, 2, 3, 4], index=2)
    job_satisfaction = st.radio("Job Satisfaction (1-4)", options=[1, 2, 3, 4], index=2)


    # Kullanıcı girdilerini bir DataFrame'e dönüştürme
    input_data = pd.DataFrame({
        'Age': [age],
        'DistanceFromHome': [distance_from_home],
        'MonthlyIncome': [monthly_income],
        'YearsAtCompany': [years_at_company],
        'YearsInCurrentRole': [years_in_current_role],
        'YearsSinceLastPromotion': [years_since_last_promotion],
        'EnvironmentSatisfaction': [environment_satisfaction],
        'JobInvolvement': [job_involvement],
        'JobSatisfaction': [job_satisfaction]
    })

    # Overtime sütunu için dummy (one-hot encoding)
    input_data['OverTime_Yes'] = 1 if overtime == 'Yes' else 0

    # Kullanıcıdan gelen veriyi 'Processed_data.csv' verisi ile eşleştirme
    input_data = pd.concat([df, input_data], axis=0, ignore_index=True).tail(1)  # CSV'den son satır al

    # Sütunları modeldeki sıraya göre yeniden düzenleyelim
    input_data = input_data[model_features]

    # Kullanıcı girdilerini kullanarak tahmin yapma
    if st.button('Predict Attrition'):
        prediction = Cat_model.predict(input_data)  # Kullanıcı verileri ile tahmin yap
        if prediction == 1:
            st.error('The employee may leave the company.')
        else:
            st.success('The employee is likely to stay.')

    # Kullanıcıdan alınan verileri göster
    st.write("Entered Employee Data:")
    st.write(input_data)
