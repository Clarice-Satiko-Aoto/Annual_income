import streamlit as st
import pandas as pd
import pickle




st.header("Estudo do salário anual dos profissionais de uma multinacional")



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
if st.button('classificar'):
    # predição dos novos dados
    loaded_model = pickle.load(open('LR_model_OHE_MaxAbsScaler_13_05.pkl', 'rb'))
    #print(loaded_model)
    y_pred = loaded_model.predict(dados)
    #print(y_pred)

    if y_pred == ' <=50K':
        st.write('O salário anual é igual ou inferior a U$50k')
    else:
        st.write('O salário anual é maior que U$50k')
    
