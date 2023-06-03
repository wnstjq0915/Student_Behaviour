import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def main():
    st.title('학생 행동 분석')

    df = pd.read_csv('data/Student_Behaviour_kor.csv')
    menu = ['개요', '데이터 분석', '데이터 예측', '데이터 분류']
    choise = st.sidebar.selectbox('목록', menu)
    df_onehot = pd.read_csv('data/Student_Behaviour_kor3.csv')
    X_df = pd.read_csv('data/Student_Behaviour_kor3_1.csv')
    onehot_dict = {
    '학과' : ['B.com Accounting and Finance', 'B.com ISM', 'BCA', 'Commerce'],
    '취미' : ['영화', '책읽기', '운동', '비디오게임'],
    '공부 시간대' : ['아무때나', '아침', '저녁']
    }
    label_list = ['자격증', '성별', '학위만족도', '아르바이트 여부']
    onehot_list = ['학과', '취미', '공부 시간대']
    int_list = ['공부시간', '미디어 이용시간', '왕복통학시간', '스트레스 지수', '자산상황']
    normal_list = ['신장(cm)', '몸무게(kg)', '10살 성적', '12살 성적', '대학성적', '희망연봉', '학위기반 취업고려(%)']

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
        10살 성적 : 10살때의 성적
        12살 성적 : 12살때의 성적
        대학성적 : 대학성적
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

        dict_key = None
        sel_corr = st.selectbox('상관관계를 볼 데이터를 선택해주세요.', df.columns)
        if sel_corr in onehot_dict.keys():
            dict_key = sel_corr
            sel_corr = st.selectbox('값을 선택해주세요.', onehot_dict[sel_corr])

        df_sel_corr = (df_corr[sel_corr].dropna() * 100)
        if len(df_sel_corr.keys()) == 1:
            st.text('관계 있는 데이터가 없습니다.')
        else:
            for i in df_sel_corr.sort_values(ascending=False).keys()[1:]:
                if dict_key:
                    if i in onehot_dict[dict_key]:
                        continue
                if df_sel_corr[i] > 0:
                    st.text(f"'{i}'값과 {int(df_sel_corr[i])}% 비례관계")
                else:
                    st.text(f"'{i}'값과 {int(abs(df_sel_corr[i]))}% 반비례관계")


    elif choise == menu[2]: # 따로 파일 만들어서 import하기
        st.header('데이터 예측')
        st.text('입력 받을 데이터와 예측할 데이터를 정하면 해당 값을 출력합니다.')
        st.subheader('입력할 데이터')
        Travelling_Time = {
                            ' ~ 0.5' : 0,
                            '0.5 ~ 1' : 1,
                            '1 ~ 1.5' : 2,
                            '1.5 ~ 2' : 3,
                            '2 ~ 2.5' : 4,
                            '2.5 ~ 3' : 5,
                            '3 ~ ' : 6
                            }
        social_medai_video = {
                            '0' : 0,
                            ' ~ 0.5' : 1,
                            '0.5 ~ 1' : 2,
                            '1 ~ 1.5' : 3,
                            '1.5 ~ 2' : 4,
                            '2 ~ ' : 5
                            }
        daily_studing_time = {
                            ' ~ 0.5' : 0,
                            '0.5 ~ 1' : 1,
                            '1 ~ 2' : 2,
                            '2 ~ 3' : 3,
                            '3 ~ 4' : 4,
                            '4 ~ ' : 5
                            }
        label_dict = {
                    'No': 0, 
                    'Yes' : 1, 
                    '여자' : 0,
                    '남자' : 1 
                    }
        Financial_Status = {'매우나쁨':0, '나쁨':1, '좋음':2, '매우좋음':3}
        Stress_Level = {'매우나쁨':0, '나쁨':1, '좋음':2, '매우좋음':3}

# 한번에 선택할 수 있는 기능 만들기
        pred_choise = st.multiselect('입력받을 데이터를 정해주세요.', df.columns, max_selections=len(df.columns) - 1)
        new_data = []
        X_choise = list(set(pred_choise) - set(onehot_list))
        if pred_choise:
            if '자격증' in pred_choise:
                new_data.append(label_dict[st.select_slider('자격증 보유여부를 정해주세요.', ['Yes', 'No'], value=df['자격증'].value_counts().first_valid_index())])
            if '성별' in pred_choise:
                new_data.append(label_dict[st.select_slider('성별을 정해주세요.', ['남자', '여자'], value=df['성별'].value_counts().first_valid_index())])
            if '신장(cm)' in pred_choise:
                new_data.append(st.number_input('키를 입력해주세요.', min_value=0, max_value=250, value=int(df['신장(cm)'].mean())))
            if '몸무게(kg)' in pred_choise:
                new_data.append(st.number_input('몸무게를 입력해주세요.', min_value=0, max_value=250, value=int(df['몸무게(kg)'].mean())))
            if '10살 성적' in pred_choise:
                new_data.append(st.number_input('10살 때의 성적을 입력해주세요.', min_value=0, max_value=100, value=int(df['10살 성적'].mean())))
            if '12살 성적' in pred_choise:
                new_data.append(st.number_input('12살 때의 성적을 입력해주세요.', min_value=0, max_value=100, value=int(df['12살 성적'].mean())))
            if '대학성적' in pred_choise:
                new_data.append(st.number_input('대학생 때의 성적을 입력해주세요.', min_value=0, max_value=100, value=int(df['대학성적'].mean())))
            if '공부시간' in pred_choise:
                new_data.append(daily_studing_time[st.selectbox('하루 공부시간을 정해주세요.', daily_studing_time.keys())])
            if '희망연봉' in pred_choise:
                new_data.append(st.number_input('원하는 연봉($)을 정해주세요.', min_value=0, value=int(df['희망연봉'].median())))
            if '학위만족도' in pred_choise:
                new_data.append(label_dict[st.select_slider('학력이 마음에 드시나요?', ['Yes', 'No'], value=df['학위만족도'].value_counts().first_valid_index())])
            if '학위기반 취업고려(%)' in pred_choise:
                new_data.append(st.number_input('학위에 따라 경력을 추구하려는 의지(%)를 입력해주세요.', min_value=0, max_value=100, value=int(df['학위기반 취업고려(%)'].mean())))
            if '미디어 이용시간' in pred_choise:
                new_data.append(social_medai_video[st.selectbox('소셜미디어와 영상시청에 하루에 할당하는 시간을 정해주세요.', social_medai_video.keys())])
            if '왕복통학시간' in pred_choise:
                new_data.append(Travelling_Time[st.selectbox('등교시간(왕복)을 정해주세요.', Travelling_Time.keys())])
            if '스트레스 지수' in pred_choise:
                new_data.append(Stress_Level[st.selectbox('스트레스 레벨을 정해주세요.', Stress_Level.keys())])
            if '자산상황' in pred_choise:
                new_data.append(Financial_Status[st.selectbox('자금상태를 정해주세요.', Financial_Status.keys())])
            if '아르바이트 여부' in pred_choise:
                new_data.append(label_dict[st.select_slider('아르바이트를 하시나요?', ['Yes', 'No'], value=df['아르바이트 여부'].value_counts().first_valid_index())])



                # new_data.append(Financial_Status[st.selectbox('자금상태를 정해주세요.', Financial_Status.keys())]) # 이걸
                # new_data.append(X_df['자산상황'].values[df[df['자산상황'] == st.selectbox('자금상태를 정해주세요.', df[pred_choise].unique())]['자산상황'].index[0]])

                # 이렇게 바꿔도 될 듯.
                # 각 컬럼마다 설명을 적은 딕셔너리를 만들고,
                # 개요에 for문과 key()를 이용해 그 딕셔너리를 써서 설명하고,
                
                # 이 바로 위 코드의 for문에도 그 딕셔너리 설명값을 이용하면 코드가 확 줄어들 듯.


                # 셀렉트 슬라이더로 정할 값 위로 보내고,
                # 그 밑에 일반값들 정하도록.


                # csv를 재가공하기.
                # 컬럼 순서 변경.


                # 가능하면 수치화한 데이터들을
                # sorted 했을 때 값이 순서대로 이쁘게 나오도록
                # 값도 변경해주기.
                # ex: "1. ' ~ 0.5'" 같이 변경하기.


                # label_value_dict를 정의해서
                # 성별이면 [여, 남]
                # 그 외의 라벨인코딩값은 [No, Yes]로.
                # 기본값 가능하면 [1]을 통해서 남자와 yes를 기본값으로.


                # X값 다 선택하면 예측 전에 한번 확인할 수 있도록 목록을 작성하고,
                # 예측을 버튼을 눌러 하도록 하기.




            if '학과' in pred_choise:
                sel_department = st.selectbox('학과를 정해주세요.', onehot_dict['학과'])
                if sel_department == 'B.com Accounting and Finance':
                    new_data.extend([1, 0, 0])
                elif sel_department == 'B.com ISM':
                    new_data.extend([0, 1, 0])
                elif sel_department == 'BCA':
                    new_data.extend([0, 0, 1])
                elif sel_department == 'Commerce':
                    new_data.extend([0, 0, 0])
                X_choise.extend(onehot_dict['학과'][:-1])

            if '취미' in pred_choise:
                sel_hobbies = st.selectbox('취미를 정해주세요.', onehot_dict['취미'])
                if sel_hobbies == '영화':
                    new_data.extend([1, 0, 0])
                elif sel_hobbies == '책읽기':
                    new_data.extend([0, 1, 0])
                elif sel_hobbies == '운동':
                    new_data.extend([0, 0, 1])
                elif sel_hobbies == '비디오게임':
                    new_data.extend([0, 0, 0])
                X_choise.extend(onehot_dict['취미'][:-1])

            if '공부 시간대' in pred_choise:
                sel_prefer = st.selectbox('공부하는 시간대를 정해주세요.', onehot_dict['공부 시간대'])
                if sel_prefer == '아무때나':
                    new_data.extend([1, 0])
                elif sel_prefer == '아침':
                    new_data.extend([0, 1])
                elif sel_prefer == '저녁':
                    new_data.extend([0, 0])
                X_choise.extend(onehot_dict['공부 시간대'][:-1])

            st.subheader('예측할 데이터')
            y_choise = st.selectbox('예측할 데이터를 선택해주세요.', set(df.columns) - set(pred_choise) - {'학과', '취미', '공부 시간대'})

            X = X_df[X_choise]
            y = X_df[y_choise]


            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=600)

            from sklearn.linear_model import LinearRegression
            regressor = LinearRegression()
            regressor.fit(X_train.values, y_train.values)

            # 정확도 선언
            y_pred = regressor.predict(X_test)
            accuracy = round(abs((y_test - y_pred).mean()), 2)
            per_acc =round(abs(((y_test - y_pred).mean()) * 100), 1)

            new_data = np.array(new_data)
            new_data = new_data.reshape(1, len(new_data))
            new_data = pd.DataFrame(new_data)
            st.subheader('결과') # 소수점 2자리까지.
            # 예측한거
            answer_f = regressor.predict(new_data.values)[0]
            per_ans = round((answer_f * 100), 1)
            answer = round(answer_f, 2)

            if y_choise in label_list:
                label_values = ['No', 'Yes'] # 이거랑 남녀부분 위쪽에서 dict로 라벨인코더한 데이터들 값 정리할거.
                if y_choise == '성별':
                    label_values = ['여자', '남자']
                if answer > 1:
                    st.text(label_values[1])

                elif answer < 0:
                    st.text(label_values[0])

                else:
                    st.text(f'{per_ans - per_acc} ~ {per_ans + per_acc}% {label_values[round(answer_f)]}')


            # 수치화한 데이터들 예측할 때 맥스랑 민값 정하고,
            # 값의 의미를 알 수 있도록 표기하기.

            # elif y_choise in int_list:
            #     if answer > len():
            #         pass # 수정할거


            # 원핫인코딩한 값도 가능하면 도전해보기.
            # 아마 가능할 듯?
            else:
                st.text(f'{y_choise}: {answer} ± {accuracy}')
        else:
            st.text('선택된 값이 없습니다.')



    elif choise == menu[3]:
        pass

if __name__ == '__main__':
    main()