import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def main():
    st.title('학생 행동 분석')
    df = pd.read_csv('data/Student_Behaviour.csv')
    menu = ['데이터 분석', '데이터 예측', '데이터 분류']
    choise = st.sidebar.selectbox('목록', menu)
    X_df = pd.read_csv('data/Student_Behaviour3_1.csv')

    if choise == menu[0]:
        st.header('데이터 분석')
        st.dataframe(df)
        st.subheader('데이터의 갯수')
        select_count = st.selectbox('데이터를 선택해주세요.', df.columns)
        # 데이터 종류 제한하고, 간격 조정하기
        fig = plt.figure()
        sns.countplot(data=df, x = select_count)
        st.pyplot(fig)


    elif choise == menu[1]:
        st.subheader('데이터 예측')
        st.text('데이터를 입력하고 예측할 값을 정하면 해당 값을 반환합니다.')
        Travelling_Time = ['0 - 30 minutes',
                            '30 - 60 minutes',
                            '1 - 1.30 hour',
                            '1.30 - 2 hour',
                            '2 - 2.30 hour',
                            '2.30 - 3 hour',
                            'more than 3 hour']
        social_medai_video = ['0 Minute',
                            '1 - 30 Minute',
                            '30 - 60 Minute',
                            '1 - 1.30 hour',
                            '1.30 - 2 hour',
                            'More than 2 hour']
        daily_studing_time = ['0 - 30 minute',
                            '30 - 60 minute',
                            '1 - 2 Hour',
                            '2 - 3 hour',
                            '3 - 4 hour',
                            'More Than 4 hour']
        Financial_Status = ['Awful', 'Bad', 'good', 'Fabulous']
        Stress_Level = ['Awful', 'Bad', 'Good', 'fabulous']
        Department = ['B.com Accounting and Finance ', 'B.com ISM', 'BCA', 'Commerce']
        hobbies = ['Cinema', 'Reading books', 'Sports', 'Video Games']
        prefer_to_study_in = ['Anytime', 'Morning', 'Night']

# 한번에 선택할 수 있는 기능 만들기
        pred_choise = st.multiselect('입력받을 데이터를 정해주세요.', df.columns, max_selections=len(df.columns) - 1)
        X = []
        label_col = ['Do you like your degree?', 'Certification Course', 'part-time job']

        if pred_choise:
            st.subheader('입력할 데이터')
            if 'Certification Course' in pred_choise:
                X.append(st.select_slider('자격증 보유여부를 정해주세요.', ['Yes', 'No']))
            if 'Gender' in pred_choise:
                X.append(st.select_slider('성별을 정해주세요.', ['Male', 'Female']))
            if 'Height(CM)' in pred_choise:
                X.append(st.number_input('키를 입력해주세요.', min_value=0, max_value=250, value=170))
            if 'Weight(KG)' in pred_choise:
                X.append(st.number_input('몸무게를 입력해주세요.', min_value=0, max_value=250, value=65))
            if '10th Mark' in pred_choise:
                X.append(st.number_input('10살 때의 성적을 입력해주세요.', min_value=0, max_value=100, value=50))
            if '12th Mark' in pred_choise:
                X.append(st.number_input('12살 때의 성적을 입력해주세요.', min_value=0, max_value=100, value=50))
            if 'college Mark' in pred_choise:
                X.append(st.number_input('대학생 때의 성적을 입력해주세요.', min_value=0, max_value=100, value=50))
            if 'daily studing time' in pred_choise:
                X.append(st.selectbox('하루 공부시간을 정해주세요.', daily_studing_time))
            if 'salary expectation' in pred_choise:
                X.append(st.number_input('원하는 연봉($)을 정해주세요.', min_value=0, value=25000))
            if 'Do you like your degree?' in pred_choise:
                X.append(st.select_slider('학력이 마음에 드시나요?', ['Yes', 'No']))
            if 'willingness to pursue a career based on their degree(%)' in pred_choise:
                X.append(st.number_input('학위에 따라 경력을 추구하려는 의지(%)를 입력해주세요.', min_value=0, max_value=100, value=50))
            if 'social medai & video' in pred_choise:
                X.append(st.selectbox('소셜미디어와 영상시청에 하루에 할당하는 시간을 정해주세요.', social_medai_video))
            if 'Travelling Time' in pred_choise:
                X.append(st.selectbox('등교시간(왕복)을 정해주세요.', Travelling_Time))
            if 'Stress Level' in pred_choise:
                X.append(st.selectbox('스트레스 레벨을 정해주세요.', Stress_Level))
            if 'Financial Status' in pred_choise:
                X.append(st.selectbox('자금상태를 정해주세요.', Financial_Status))
            if 'part-time job' in pred_choise:
                X.append(st.select_slider('아르바이트를 하시나요?', ['Yes', 'No']))

            if 'Department' in pred_choise:
                sel_department = st.selectbox('학과를 정해주세요.', Department)
                if sel_department == 'B.com Accounting and Finance':
                    X.extend([1, 0, 0])
                elif sel_department == 'B.com ISM':
                    X.extend([0, 1, 0])
                elif sel_department == 'BCA':
                    X.extend([0, 0, 1])
                elif sel_department == 'Commerce':
                    X.extend([0, 0, 0])

            if 'hobbies' in pred_choise:
                sel_hobbies = st.selectbox('취미를 정해주세요.', hobbies)
                if sel_hobbies == 'Cinema':
                    X.extend([1, 0, 0])
                elif sel_hobbies == 'Reading books':
                    X.extend([0, 1, 0])
                elif sel_hobbies == 'Sports':
                    X.extend([0, 0, 1])
                elif sel_hobbies == 'Video Games':
                    X.extend([0, 0, 0])

            if 'prefer to study in' in pred_choise:
                sel_prefer = st.selectbox('공부하는 시간대를 정해주세요.', prefer_to_study_in)
                if sel_prefer == 'Anytime':
                    X.extend([1, 0])
                elif sel_prefer == 'Morning':
                    X.extend([0, 1])
                elif sel_prefer == 'Night':
                    X.extend([0, 0])
            
            st.text(X)
            st.text(len(X))


            y = st.selectbox('예측할 데이터를 선택해주세요.', set(X_df.columns) - set(pred_choise))

            # 1. X와 y 나누기
            # X = df.loc[ : , 'Gender' : 'Net Worth']



            # # 2. 학습시킬 데이터와 테스트할 데이터를 나누기
            # from sklearn.model_selection import train_test_split
            # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=50)

            # # 3. 학습시키기
            # from sklearn.linear_model import LinearRegression
            # regressor = LinearRegression()
            # regressor.fit(X_train, y_train)
            # # regressor.fit(X_train.values, y_train.values)

            # # 4. 예측한 값과 실제값을 비교하기
            # y_pred = regressor.predict(X_test)
            # ((y_test - y_pred) **2).mean()

            # # 5. X가 아닌 다른 값으로 예측해보기.
            # new_data = np.array([1, 50, 40000, 50000, 200000])
            # new_data = new_data.reshape(1, 5) # 오류를 막기 위해 차원 변경.
            # regressor.predict(new_data)
        else:
            st.text('선택된 값이 없습니다.')



    elif choise == menu[2]:
        pass

if __name__ == '__main__':
    main()

# """
# # 1. X와 y 나누기
# y = df['Car Purchase Amount'] # 멀티셀렉트로 컬럼 어떤거 선택힐지 정하고, 각각 값을 넣을 수 있도록 함.
# X = df.loc[ : , 'Gender' : 'Net Worth']

# # 2. 학습시킬 데이터와 테스트할 데이터를 나누기
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=50)

# # 3. 학습시키기
# from sklearn.linear_model import LinearRegression
# regressor = LinearRegression()
# regressor.fit(X_train, y_train)
# # regressor.fit(X_train.values, y_train.values)

# # 4. 예측한 값과 실제값을 비교하기
# y_pred = regressor.predict(X_test)
# ((y_test - y_pred) **2).mean()

# # 5. X가 아닌 다른 값으로 예측해보기.
# new_data = np.array([1, 50, 40000, 50000, 200000])
# new_data = new_data.reshape(1, 5) # 오류를 막기 위해 차원 변경.
# regressor.predict(new_data)
# """