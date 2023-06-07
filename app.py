import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
# 시간있으면 가능한 리스트들 다 넘파이로 바꾸기.

def main():
    st.title('대학생 행동 분석 앱')

    df = pd.read_csv('data/Student_Behaviour_kor_sort.csv')
    menu = ['개요', '데이터 분석', '데이터 예측']
    choise = st.sidebar.selectbox('목록', menu)
    onehot_df = pd.read_csv('data/Student_Behaviour_kor2.csv')
    onehot_dict = {
    '학과' : ['B.com Accounting and Finance', 'B.com ISM', 'BCA', 'Commerce'],
    '취미' : ['영화', '책읽기', '운동', '비디오게임'],
    '공부 시간대' : ['아무때나', '아침', '저녁']
    }
    label_list = ['성별', '자격증', '학위만족도', '아르바이트 여부']
    int_list = ['공부시간', '미디어 이용시간', '왕복통학시간', '스트레스 지수', '자산상황']
    normal_list = ['신장(cm)', '몸무게(kg)', '10살 성적', '12살 성적', '대학성적', '희망연봉', '학위기반 취업고려(%)']
    df1 = df[:][:]
    func1 = lambda x : x[4:-1]
    for i in int_list:
        df1[i] = df1[i].apply(func1)
    check = st.checkbox('데이터프레임 보기')
    if check:
        st.dataframe(df1)
        st.text('234 rows × 27 columns')
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
        if not check:
            st.image('data/img.jpg')
        st.subheader('개요')
        st.text('대학생들의 데이터를 분석하고, 데이터를 선택해 예측합니다.')
        st.subheader('목차')
        st.text('''
        - 데이터의 값 설명
        - 데이터 값 확인
        - 데이터 시각화 및 분석
        - 여러 데이터 예측하기
        ''')
        
        st.subheader('출처')
        st.text('kaggle Student Behavior')
        st.text('https://www.kaggle.com/datasets/gunapro/student-behavior?resource=download')


    elif choise == menu[1]:
        import platform
        platform.platform()
        # if platform.system() == 'Windows':
        plt.rcParams['font.family'] = 'Malgun Gothic'
        # elif platform.system() == 'Darwin':
        #     plt.rcParams['font.family'] = 'AppleGothic'
        # else:
        #     # import matplotlib.font_manager as fm
        #     # path = '/usr/share/fonts/NanumFont/NanumGothic.ttf'
        #     # fontprop = fm.FontProperties(fname=path)
        #     # plt.rcParams['font.family'] = fontprop
        #     plt.rcParams['font.family'] = 'NanumGothic'
        plt.rcParams['font.size'] = 15
        plt.rcParams['axes.unicode_minus'] = False

        calculator = {
            '값의 종류' : 'df1[i].unique()',
            '값의 종류 수' : 'df1[i].nunique()',
            '최댓값' : 'df1[i].max()',
            '최솟값' : 'df1[i].min()',
            '평균' : 'round(df1[i].mean(), 1)',
            '중앙값' : 'df1[i].median()'
        }
        st.header('데이터 분석')
        if check:
            st.subheader('데이터 설명')
            for i in col_explain.keys():
                st.text(f'{i}: {col_explain[i]}')

        st.subheader('데이터 값')
        select_calcul = st.radio('보고싶은 값을 선택해주세요.', calculator.keys())
        for i in df1.columns:
            if df1[i].dtype != object and select_calcul not in list(calculator.keys())[:2] or (df1[i].dtype == object and select_calcul in list(calculator.keys())[:2]):
                st.text(f'{i}의 {select_calcul}: {eval(calculator[select_calcul])}')

        st.subheader('데이터의 갯수')
        select_count = st.selectbox('데이터를 선택해주세요.', set(df1.columns) - set(normal_list[:-1]))
        fig = plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)

        ax = sns.countplot(data=df1[select_count].sort_values(ascending=False).to_frame(), x = select_count)

        if select_count.endswith('시간') or select_count == '학과':
            plt.xticks(fontsize=13)
            plt.xticks(rotation=17)
        ax.set(xlabel=None)
        plt.subplot(1, 2, 2)
        explode = [0.05] * df1[select_count].nunique()
        if select_count == normal_list[-1]:
            plt.pie(df1[select_count].sort_values().value_counts(sort=False).values, labels=df1[select_count].sort_values().unique(), explode=explode)
        else:
            plt.pie(df1[select_count].sort_values(ascending=False).value_counts(sort=False).values, labels=df1[select_count].sort_values(ascending=False).unique(), explode=explode, startangle=-78)
        plt.suptitle(select_count)
        st.pyplot(fig)
        st.dataframe(df[select_count].sort_values().value_counts(sort=False))

        st.subheader('산점도')
        plt_li = []
        plt_li.append(st.selectbox('x축을 선택해주세요.', set(normal_list[:-2])))
        plt_li.append(st.selectbox('y축을 선택해주세요.', set(normal_list[:-2]) - {plt_li[0]}))
        if st.checkbox('데이터를 분류하여 산점도를 표현하시겠습니까?'):
            plt_li.append(st.selectbox('분류할 기준을 선택해주세요.', set(label_list)))

        fig = plt.figure()
        if len(plt_li) == 3:
            plt.scatter(df[plt_li[0]], df[plt_li[1]], s=500, c=onehot_df[plt_li[2]], cmap='viridis', alpha=0.25)
            plt.colorbar(ticks=[], label=f'{sorted(df1[plt_li[2]].unique())[0]}             ~             {sorted(df1[plt_li[2]].unique())[1]}', shrink=0.5, orientation='horizontal')
        else:
            plt.scatter(df[plt_li[0]], df[plt_li[1]], s=500, cmap='viridis', alpha=0.4)
        plt.xlabel(plt_li[0])
        plt.ylabel(plt_li[1])
        st.pyplot(fig)

        st.subheader('상관관계')
        df1_corr = onehot_df.corr()
        fig = plt.figure()
        plt.title('간략한 상관관계(%)')
        sns.heatmap(data=df1.corr(numeric_only=True).loc[:, :] * 100, annot=True, vmin=-100, vmax=100, cmap='coolwarm', fmt='.1f', linewidths=1)
        st.pyplot(fig)
        for i in onehot_df.columns:
            df1_corr.loc[abs(df1_corr[i]) < 0.1, i] = np.NaN
        dict_key = None
        sel_corr = st.selectbox('자세한 상관관계를 볼 데이터를 선택해주세요.', df1.columns)
        if sel_corr in onehot_dict.keys():
            dict_key = sel_corr
            sel_corr = st.selectbox('값을 선택해주세요.', onehot_dict[sel_corr])
        df1_sel_corr = (df1_corr[sel_corr].dropna() * 100)
        if len(df1_sel_corr.keys()) == 1:
            st.text('관계 있는 데이터가 없습니다.')
        else:
            for i in df1_sel_corr.sort_values(ascending=False).keys()[1:]:
                if dict_key:
                    if i in onehot_dict[dict_key]:
                        continue
                if df1_sel_corr[i] > 0:
                    st.text(f"'{i}'값과 {int(df1_sel_corr[i])}% 비례관계")
                else:
                    st.text(f"'{i}'값과 {int(abs(df1_sel_corr[i]))}% 반비례관계")


    elif choise == menu[2]:
        st.header('데이터 예측')
        st.text('예측할 데이터와 입력 받을 데이터를 정하면 해당 값을 출력합니다.')

        st.subheader('예측할 데이터')
        y_choise = st.selectbox('예측할 데이터를 선택해주세요.', df.columns)

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
                st.text(y_choise)
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
                        st.text(sorted(df1[y_choise].unique())[ans] + '시간')
                    else:
                        answer1, answer2 = ans - acc, ans + acc
                        if answer1 < 0:
                            answer1 = 0
                        elif answer2 > df[y_choise].nunique() - 1:
                            answer2 = df[y_choise].nunique() - 1
                        elif math.floor(answer1) == math.ceil(answer2):
                            st.text(sorted(df1[y_choise].unique())[answer1])
                        else:
                            st.text(sorted(df1[y_choise].unique())[math.floor(answer1)] + ' ~ ' + sorted(df1[y_choise].unique())[math.ceil(answer1)])
                else:
                    ans, acc = reg(onehot_df, X, new_data, y_choise)
                    st.text(f'{round(ans - acc, 2)} ~ {round(ans + acc, 2)}')

        else:
            st.text('선택된 값이 없습니다.')


def label_def(col):
    return ['여자', '남자'] if col == '성별' else ['No', 'Yes']

def reg(onehot_df, X, new_data, y_choise):
    y = onehot_df[y_choise]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=600)
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