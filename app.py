import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def main():
    st.title('학생 행동 분석')

    df = pd.read_csv('data/Student_Behaviour.csv')
    menu = ['개요', '데이터 분석', '데이터 예측', '데이터 분류']
    choise = st.sidebar.selectbox('목록', menu)
    df_onehot = pd.read_csv('data/Student_Behaviour3.csv')
    X_df = pd.read_csv('data/Student_Behaviour3_1.csv')
    onehot_dict = {
    'Department' : ['B.com Accounting and Finance ', 'B.com ISM', 'BCA', 'Commerce'],
    'hobbies' : ['Cinema', 'Reading books', 'Sports', 'Video Games'],
    'prefer_to_study_in' : ['Anytime', 'Morning', 'Night']
    }

    if choise == menu[0]:
        st.header('개요') # 데이터를 가져온 링크 적기.
        st.subheader('대학생들 정보를 분석한 사이트')
        st.dataframe(df.head())
        st.text("""
        Certification Course : 자격증 보유여부
        Gender : 성별
        Department : 학과
        Height(CM) : 키
        Weight(KG) : 몸무게
        10th Mark : 10살때의 성적
        12th Mark : 12살때의 성적
        college mark : 대학성적
        hobbies : 취미
        daily study : 하루 공부시간
        prefer to study in : 공부하는 시간대
        salary expectation : 원하는 연봉
        Do you like your degree : 학위 만족도
        willingness to pursue a career based on their degree(%) : 학위를 바탕으로 직업을 추구하려는 의지
        social medai & video : 소셜미디어와 영상시청에 하루에 할당하는 시간
        Travelling Time : 등교시간(왕복)
        stress Level : 스트레스 지수
        Financial Status : 자금 정도
        part-time job : 아르바이트 활동여부
        """)
        st.subheader('출처')
        st.text('kaggle Student Behavior')
        st.text('https://www.kaggle.com/datasets/gunapro/student-behavior?resource=download')

    if choise == menu[1]:
        st.header('데이터 분석')
        st.dataframe(df)
        st.subheader('데이터의 갯수')
        select_count = st.selectbox('데이터를 선택해주세요.', df.columns)
        # 데이터 종류 제한하고, 간격 조정하기
        fig = plt.figure()
        sns.countplot(data=df, x = select_count)
        st.pyplot(fig)

        st.subheader('상관관계')
        df_corr = df_onehot.corr()
        for i in df_onehot.columns:
            df_corr.loc[abs(df_corr[i]) < 0.1, i] = np.NaN

        sel_corr = st.selectbox('상관관계를 볼 데이터를 선택해주세요.', df.columns)
        if sel_corr in onehot_dict.keys():
            sel_corr = st.selectbox('값을 선택해주세요.', onehot_dict[sel_corr])

        df_sel_corr = (df_corr[sel_corr].dropna() * 100)
        if len(df_sel_corr.keys()) == 1:
            st.text('관계 있는 데이터가 없습니다.')
        else:
            for i in df_sel_corr.sort_values(ascending=False).keys()[1:]:
                if df_sel_corr[i] > 0:
                    st.text(f'{i}값과 {int(df_sel_corr[i])}% 비례관계')
                else:
                    st.text(f'{i}값과 {int(abs(df_sel_corr[i]))}% 반비례관계')


    elif choise == menu[2]: # 따로 파일 만들어서 import하기
        st.header('데이터 예측')
        st.text('입력 받을 데이터와 예측할 데이터를 정하면 해당 값을 출력합니다.')
        st.subheader('입력할 데이터')
        Travelling_Time = {'0 - 30 minutes' : 0,
                            '30 - 60 minutes' : 1,
                            '1 - 1.30 hour' : 2,
                            '1.30 - 2 hour' : 3,
                            '2 - 2.30 hour' : 4,
                            '2.30 - 3 hour' : 5,
                            'more than 3 hour' : 6}
        social_medai_video = {'0 Minute' : 0,
                            '1 - 30 Minute' : 1,
                            '30 - 60 Minute' : 2,
                            '1 - 1.30 hour' : 3,
                            '1.30 - 2 hour' : 4,
                            'More than 2 hour' : 5}
        daily_studing_time = {'0 - 30 minute' : 0,
                            '30 - 60 minute' : 1,
                            '1 - 2 Hour' : 2,
                            '2 - 3 hour' : 3,
                            '3 - 4 hour' : 4,
                            'More Than 4 hour' : 5}
        label_dict = {'Yes' : 1, 
                    'No': 0, 
                    'Male' : 1, 
                    'Female' : 0}
        Financial_Status = {'Awful':0, 'Bad':1, 'good':2, 'Fabulous':3}
        Stress_Level = {'Awful':0, 'Bad':1, 'Good':2, 'fabulous':3}

# 한번에 선택할 수 있는 기능 만들기
        pred_choise = st.multiselect('입력받을 데이터를 정해주세요.', df.columns, max_selections=len(df.columns) - 1)
        new_data = []
        X_choise = list(set(pred_choise) - {'Department', 'hobbies', 'prefer to study in'})
        if pred_choise:
            if 'Certification Course' in pred_choise:
                new_data.append(label_dict[st.select_slider('자격증 보유여부를 정해주세요.', ['Yes', 'No'], value=df['Certification Course'].value_counts().first_valid_index())])
            if 'Gender' in pred_choise:
                new_data.append(label_dict[st.select_slider('성별을 정해주세요.', ['Male', 'Female'], value=df['Gender'].value_counts().first_valid_index())])
            if 'Height(CM)' in pred_choise:
                new_data.append(st.number_input('키를 입력해주세요.', min_value=0, max_value=250, value=int(df['Height(CM)'].mean())))
            if 'Weight(KG)' in pred_choise:
                new_data.append(st.number_input('몸무게를 입력해주세요.', min_value=0, max_value=250, value=int(df['Weight(KG)'].mean())))
            if '10th Mark' in pred_choise:
                new_data.append(st.number_input('10살 때의 성적을 입력해주세요.', min_value=0, max_value=100, value=int(df['10th Mark'].mean())))
            if '12th Mark' in pred_choise:
                new_data.append(st.number_input('12살 때의 성적을 입력해주세요.', min_value=0, max_value=100, value=int(df['12th Mark'].mean())))
            if 'college mark' in pred_choise:
                new_data.append(st.number_input('대학생 때의 성적을 입력해주세요.', min_value=0, max_value=100, value=int(df['college mark'].mean())))
            if 'daily studing time' in pred_choise:
                new_data.append(daily_studing_time[st.selectbox('하루 공부시간을 정해주세요.', daily_studing_time.keys())])
            if 'salary expectation' in pred_choise:
                new_data.append(st.number_input('원하는 연봉($)을 정해주세요.', min_value=0, value=int(df['salary expectation'].median())))
            if 'Do you like your degree?' in pred_choise:
                new_data.append(label_dict[st.select_slider('학력이 마음에 드시나요?', ['Yes', 'No'], value=df['Do you like your degree?'].value_counts().first_valid_index())])
            if 'willingness to pursue a career based on their degree(%)' in pred_choise:
                new_data.append(st.number_input('학위에 따라 경력을 추구하려는 의지(%)를 입력해주세요.', min_value=0, max_value=100, value=int(df['willingness to pursue a career based on their degree(%)'].mean())))
            if 'social medai & video' in pred_choise:
                new_data.append(social_medai_video[st.selectbox('소셜미디어와 영상시청에 하루에 할당하는 시간을 정해주세요.', social_medai_video.keys())])
            if 'Travelling Time' in pred_choise:
                new_data.append(Travelling_Time[st.selectbox('등교시간(왕복)을 정해주세요.', Travelling_Time.keys())])
            if 'Stress Level' in pred_choise:
                new_data.append(Stress_Level[st.selectbox('스트레스 레벨을 정해주세요.', Stress_Level.keys())])
            if 'Financial Status' in pred_choise:
                new_data.append(Financial_Status[st.selectbox('자금상태를 정해주세요.', Financial_Status.keys())])
            if 'part-time job' in pred_choise:
                new_data.append(label_dict[st.select_slider('아르바이트를 하시나요?', ['Yes', 'No'], value=df['part-time job'].value_counts().first_valid_index())])

            if 'Department' in pred_choise:
                sel_department = st.selectbox('학과를 정해주세요.', onehot_dict['Department'])
                if sel_department == 'B.com Accounting and Finance ':
                    new_data.extend([1, 0, 0])
                elif sel_department == 'B.com ISM':
                    new_data.extend([0, 1, 0])
                elif sel_department == 'BCA':
                    new_data.extend([0, 0, 1])
                elif sel_department == 'Commerce':
                    new_data.extend([0, 0, 0])
                X_choise.extend(onehot_dict['Department'][:-1])

            if 'hobbies' in pred_choise:
                sel_hobbies = st.selectbox('취미를 정해주세요.', onehot_dict['hobbies'])
                if sel_hobbies == 'Cinema':
                    new_data.extend([1, 0, 0])
                elif sel_hobbies == 'Reading books':
                    new_data.extend([0, 1, 0])
                elif sel_hobbies == 'Sports':
                    new_data.extend([0, 0, 1])
                elif sel_hobbies == 'Video Games':
                    new_data.extend([0, 0, 0])
                X_choise.extend(onehot_dict['hobbies'][:-1])

            if 'prefer to study in' in pred_choise:
                sel_prefer = st.selectbox('공부하는 시간대를 정해주세요.', onehot_dict['prefer_to_study_in'])
                if sel_prefer == 'Anytime':
                    new_data.extend([1, 0])
                elif sel_prefer == 'Morning':
                    new_data.extend([0, 1])
                elif sel_prefer == 'Night':
                    new_data.extend([0, 0])
                X_choise.extend(onehot_dict['prefer_to_study_in'][:-1])

            st.subheader('예측할 데이터')
            y_choise = st.selectbox('예측할 데이터를 선택해주세요.', set(df.columns) - set(pred_choise) - {'Department', 'hobbies', 'prefer to study in'})

            X = X_df[X_choise]
            y = X_df[y_choise]


            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=600)

            from sklearn.linear_model import LinearRegression
            regressor = LinearRegression()
            regressor.fit(X_train.values, y_train.values)

            st.text('정확도') # 값 표현방식 바꾸기
            y_pred = regressor.predict(X_test)
            st.text(st.text(((y_test - y_pred) **2).mean()))

            new_data = np.array(new_data)
            st.write(new_data)
            new_data = new_data.reshape(1, len(new_data))
            new_data = pd.DataFrame(new_data)
            st.text('결과') # 소수점 2자리까지.
            st.text(regressor.predict(new_data.values)[0])
        else:
            st.text('선택된 값이 없습니다.')



    elif choise == menu[3]:
        pass

if __name__ == '__main__':
    main()