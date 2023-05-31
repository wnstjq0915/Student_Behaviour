import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def main():
    st.title('학생 행동 분석')
    df = pd.read_csv('data/Student_Behaviour.csv')
    menu = ['데이터 분석', '데이터 예측', '데이터 분류']
    choise = st.sidebar.selectbox('목록', menu)

    if choise == menu[0]:
        st.header('데이터 분석')
        st.dataframe(df)
        st.subheader('데이터의 갯수')
        select_count = st.selectbox('데이터를 선택해주세요.', df.columns)
        # 데이터 종류 제한하고, 간격 조정하기
        fig = plt.figure()
        sns.countplot(data=df, x = select_count)
        st.pyplot(fig)

if __name__ == '__main__':
    main()