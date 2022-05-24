import streamlit as st
import pandas as pd
import pickle
import sklearn
import PIL
from PIL import Image


st.write("**Project Goal**: A multinational had requested a study on the annual income for prospective \
    employees and was provided a dataset  extracted by Barry Becker from the 1994 US Census Database.\
    The dataset consists of anonymous information such as occupation, age, capital gain,  education and more.")


st.write("**Study of professionals' annual income for a multinational company.**")
st.write("Please, input the data below:")



#@st.cache(allow_output_mutation=True)


col1, col2 = st.columns(2)
with col1:
    idade = st.number_input('Age', min_value=16, max_value=98, value=16)
    
    escolaridade= st.selectbox('Education', (' Bachelors', ' HS-grad', ' 11th', ' Masters', ' 9th',
       ' Some-college', ' Assoc-acdm', ' 7th-8th', ' Doctorate',
       ' Assoc-voc', ' Prof-school', ' 5th-6th', ' 10th', ' Preschool',
       ' 12th', ' 1st-4th'))

    horas_trabalhadas = st.number_input('Hour peer week:', min_value=30, max_value=100)


with col2:
    estado_civil= st.selectbox("Marital status", (' Divorced', ' Married',' Single', ' Widowed'))
    

    profissao = st.selectbox('Occupation: ', (' Exec-managerial', ' Handlers-cleaners', ' Adm-clerical',\
       ' Sales', ' Transport-moving', ' Machine-op-inspct',\
       ' Tech-support', ' Protective-serv'))
    
    ganho_capital= st.selectbox('Capital Gain:', ('yes', 'no'))

lista = [{"age": idade, 
        "education": escolaridade,
        "marital_status": estado_civil,
        "occupation": profissao , 
        "capital_gain": ganho_capital,
        "hours_per_week":horas_trabalhadas}]
dados = pd.DataFrame(lista)


#Botão para predição dos dados:
if st.button('Annual Salary'):
    # predição dos novos dados
    loaded_model = pickle.load(open('LR_model_OHE_MaxAbsScaler_13_05.pkl', 'rb'))
    #print(loaded_model)
    y_pred = loaded_model.predict(dados)
    #print(y_pred)



    if y_pred == ' <=50K':
        st.write('The annual income is less than U$50k')
    else:
        st.write('The annual income is more than U$50k')
        
barra = Image.open('./image/barVertical.png')
st.image(barra)


st.subheader("Exploratory data analysis:")



df= pd.read_csv("./dataset/salary_f1.csv")
#print(df)

#counts:
work= df.work_class.value_counts()
country = df.native_country.value_counts()
education= df.education.value_counts(ascending=True, sort=False )
hour = df.hours_per_week.mean()

st.bar_chart(work)
st.bar_chart(country)
st.bar_chart(education)




with st.expander("Notes:"):
     st.write(" We can see that Americans with HS graduate degrees are in the majority and work on average ", round(hour,2),   "hours per week.")

    
