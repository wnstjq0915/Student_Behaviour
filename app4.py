import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# 시간있으면 가능한 리스트들 다 넘파이로 바꾸기.

def main():
    st.title('학생 행동 분석')

    df = pd.read_csv('data/Student_Behaviour_kor_sort.csv')
    menu = ['개요', '데이터 분석', '데이터 예측', '데이터 분류']
    choise = st.sidebar.selectbox('목록', menu)
    onehot_df = pd.read_csv('data/Student_Behaviour_kor3.csv')
    onehot_dict = {
    '학과' : ['B.com Accounting and Finance', 'B.com ISM', 'BCA', 'Commerce'],
    '취미' : ['영화', '책읽기', '운동', '비디오게임'],
    '공부 시간대' : ['아무때나', '아침', '저녁']
    }
    label_list = ['성별', '자격증', '학위만족도', '아르바이트 여부']
    int_list = ['공부시간', '미디어 이용시간', '왕복통학시간', '스트레스 지수', '자산상황']
    normal_list = ['신장(cm)', '몸무게(kg)', '10살 성적', '12살 성적', '대학성적', '희망연봉', '학위기반 취업고려(%)']
    col_explain = {
        '자격증' : '자격증 보유여부',
        '성별' : '성별',
        '학과' : '학과',
        '신장(cm)' : '키',
        '몸무게(kg)' : '몸무게',
        '10살 성적' : '10살때의 성적',
        '12살 성적' : '12살때의 성적',
        '대학성적' : '대학에서의 성적',
        '취미' : '취미',
        '공부시간' : '하루 공부시간',
        '공부 시간대' : '공부하는 시간대',
        '희망연봉' : '원하는 연봉($)',
        '학위만족도' : '학위에 대한 만족도',
        '학위기반 취업고려(%)' : '학위를 바탕으로 직업을 추구하려는 의지가 얼마나 되는지',
        '미디어 이용시간' : '소셜미디어와 영상시청에 하루에 할당하는 시간',
        '왕복통학시간' : '왕복 통학시간',
        '스트레스 지수' : '스트레스 지수',
        '자산상황' : '자금의 정도',
        '아르바이트 여부' : '아르바이트 활동을 하는지'
    }


    if choise == menu[0]:
        st.header('개요') # 데이터를 가져온 링크 적기.
        st.subheader('대학생들 정보를 분석한 사이트')
        st.dataframe(df)
        st.subheader('데이터 설명')
        for i in col_explain.keys():
            st.text(f'{i}: {col_explain[i]}')
        st.subheader('출처')
        st.text('kaggle Student Behavior')
        st.text('https://www.kaggle.com/datasets/gunapro/student-behavior?resource=download')






    elif choise == menu[1]:
        st.header('데이터 분석')
        st.dataframe(df)
        st.subheader('데이터의 갯수')
        select_count = st.selectbox('데이터를 선택해주세요.', df.columns)
        # 데이터 종류 제한하고, 간격 조정하기
        fig = plt.figure()
        sns.countplot(data=df, x = select_count)
        st.pyplot(fig)

        st.subheader('상관관계')
        df_corr = onehot_df.corr()
        for i in onehot_df.columns:
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







    elif choise == menu[2]:
        st.header('데이터 예측')
        st.text('예측할 데이터와 입력 받을 데이터를 정하면 해당 값을 출력합니다.')

        st.subheader('예측할 데이터')
        y_choise = st.selectbox('예측할 데이터를 선택해주세요.', df.columns)
        st.text(set(df.columns) - set(y_choise))

        st.subheader('입력할 데이터')
        if st.checkbox('전부 선택하기'):
            pred_choise = st.multiselect('입력받을 데이터를 정해주세요.', df.drop(y_choise, axis=1).columns, default=df.drop(y_choise, axis=1).columns.values)
        else:
            pred_choise = st.multiselect('입력받을 데이터를 정해주세요.', set(df.columns) - set(y_choise))
        if pred_choise:
            new_data = []
            X_choise = []

            for i in pred_choise:
                if i in normal_list:
                    new_data.append(st.number_input(f'{add_postposition(col_explain[i])} 입력해주세요.', min_value=0, value=int(df[i].mean())))
                elif i in int_list:
                    new_data.append(sorted(df[i].unique()).index(st.selectbox(f'{add_postposition(col_explain[i])} 정해주세요.', sorted(df[i].unique()))))
                elif i in label_list:
                    new_data.append(label_def(i).index(st.select_slider(f'{add_postposition(col_explain[i])} 정해주세요.', label_def(i), value=df[i].value_counts().first_valid_index())))
                else:
                    select_onehot = onehot_dict[i].index(st.selectbox(f'{add_postposition(i)} 정해주세요.', onehot_dict[i]))
                    extend_list = [0 for j in range(len(onehot_dict[i]))]
                    extend_list[select_onehot] = 1
                    new_data.extend(extend_list)
                if i in onehot_dict.keys():
                    for k in onehot_dict[i]:
                        X_choise.append(k)
                else:
                    X_choise.append(i)

            X = onehot_df[X_choise]
            

            if st.button('예측결과 보기'):
                st.subheader('결과')
                if y_choise in onehot_dict.keys():
                    ans_dict = {'ans' : [], 'acc' : []}
                    for i in onehot_dict[y_choise]:
                        j, k = reg(onehot_df, X, new_data, i)
                        ans_dict['ans'].append(j)
                        ans_dict['acc'].append(k)
                    for i in range(len(onehot_dict[y_choise])):
                        st.text(f'{onehot_dict[y_choise][i]}: {round(ans_dict["ans"][i], 3) * 100}% ± {ans_dict["acc"][i] * 100}%')

                elif y_choise in label_list:
                    ans, acc = reg(onehot_df, X, new_data, y_choise)
                    label_values = label_def(y_choise)
                    val = label_values[1] if ans > 0.5 else label_values[0]
                    ans_list = [round(1 - abs(ans - acc - label_values.index(val)) / (abs(ans - acc) + abs(ans - acc - 1)), 3) * 100, round(1 - abs(ans + acc - label_values.index(val)) / (abs(ans + acc) + abs(ans + acc - 1)), 3) * 100]
                    min_per, max_per = min(ans_list), max(ans_list)
                    over_under = [['이하', '이상'], ['이상', '이하']]
                    if ans - acc > 1 or ans + acc < 0:
                        st.text(val)
                    elif ans + acc > 1 and ans - acc < 1:
                        st.text(f'{val}가 {min_per}% {over_under[label_values.index(val)][0]}')
                    elif ans - acc < 0 and ans + acc > 0:
                        st.text(f'{val}가 {max_per}% {over_under[label_values.index(val)][1]}')
                    else:
                        st.text(f'{min_per} ~ {max_per}% {val}')

                elif y_choise in int_list:
                    import math
                    ans, acc = reg(onehot_df, X, new_data, y_choise)
                    if y_choise in int_list[:-2]:
                        if ans > df[y_choise].nunique() - 1:
                            ans = df[y_choise].nunique() - 1
                        elif ans < 0:
                            ans = 0
                        ans = math.ceil(ans)
                        st.text(sorted(df[y_choise].unique())[ans][4:-1] + '시간')
                    else:
                        answer1, answer2 = ans - acc, ans + acc
                        if answer1 < 0:
                            answer1 = 0
                        elif answer2 > df[y_choise].nunique() - 1:
                            answer2 = df[y_choise].nunique() - 1
                        elif math.floor(answer1) == math.ceil(answer2):
                            st.text(sorted(df[y_choise].unique())[answer1][4:-1])
                        else:
                            st.text(sorted(df[y_choise].unique())[math.floor(answer1)][4:-1] + ' ~ ' + sorted(df[y_choise].unique())[math.ceil(answer1)][4:-1])
                else:
                    ans, acc = reg(onehot_df, X, new_data, y_choise)
                    st.text(f'{round(ans - acc, 2)} ~ {round(ans + acc, 2)}')

        else:
            st.text('선택된 값이 없습니다.')


def label_def(col):
    return ['여자', '남자'] if col == '성별' else ['No', 'Yes']

def reg(onehot_df, X, new_data, y_choise):
    y = onehot_df[y_choise]
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=600)
    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()
    regressor.fit(X_train.values, y_train.values)
    y_pred = regressor.predict(X_test)
    accuracy = round(abs((y_test - y_pred).mean()), 2)
    new_data = np.array(new_data)
    new_data = new_data.reshape(1, len(new_data))
    new_data = pd.DataFrame(new_data)
    answer = regressor.predict(new_data.values)[0]
    return [answer , accuracy]

def is_hangul(word):
    code = ord(word[-1])
    if 44032 <= code <= 55203:
        return True
    return False

def add_postposition(word):
    if not is_hangul(word):
        return word + '을(를)'
    return word + ('를' if (ord(word[-1]) - 44032) % 28 == 0 else '을')


if __name__ == '__main__':
    main()