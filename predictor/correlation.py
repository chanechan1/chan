import pandas as pd

    #상관도 파악
def _data_correlator():

    #데이터 로드
    train_x = pd.read_csv('weather_actual.csv', parse_dates=True)
    train_y = pd.read_csv('gens.csv', parse_dates=True)
    test_x = pd.read_csv('weather_forecast.csv', parse_dates=True)
    test_y = pd.read_csv('gens.csv', parse_dates=True)

     #####################json to pandas(dataframe)#####################
    test_x = pd.DataFrame(test_x)
    test_y = pd.DataFrame(test_y)

    #############################데이터 전처리##################################
    train_x = train_x[train_x.columns[1:]]  # 날씨실측정보
    train_y = train_y[train_y.columns[1:]]  # 발전실측정보

    test_x = test_x[test_x.columns[2:]]  # 1에 대한 일기예보만 살리기위해 일단 없앰
    #test_x = test_x.iloc[0:11616]
    test_x = test_x.iloc[11616:]
    test_y=test_y[test_y.columns[1:]]
    test_x.reset_index(drop=True, inplace=True)
    # train_x와 train_y를 결합 (여기서는 간단히 index 기준으로 결합)
    combined_train = pd.concat([test_x, train_x], axis=1)

    # 상관관계 계산
    correlation_matrix = combined_train.corr()

    # 상관관계 출력
    print(correlation_matrix)
    #print(correlation_matrix.iloc[-1])
_data_correlator()
