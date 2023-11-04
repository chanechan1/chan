# import pandas as pd
#
# def _data_correlator():
#
#     # 데이터 로드
#     train_x = pd.read_csv('weather_actual.csv')
#     gens = pd.read_csv('gens.csv')
#     test_x = pd.read_csv('weather_forecast.csv')
#
#     # 발전량 데이터 준비 (gens.csv에서 'amount' 열을 사용)
#     gens = gens[['time', 'amount']]  # 'timestamp' 열은 시간대를 나타냅니다.
#
#     # train_x, test_x에 'timestamp' 열을 사용하여 gens 데이터와 결합
#     train_x = pd.merge(train_x, gens, on='time')
#     test_x = pd.merge(test_x, gens, on='time')
#
#     # 인덱스로 설정된 행들을 삭제하려면 해당 행들이 무엇인지 명확히 해야 합니다.
#     # 여기서는 먼저 timestamp를 기준으로 정렬한 후 해당 인덱스를 제거합니다.
#     train_x.sort_values('time', inplace=True)
#     train_x.reset_index(drop=True, inplace=True)
#
#     # 여기서는 제공된 인덱스를 바탕으로 특정 행을 제거합니다.
#     # 인덱스는 로드된 후의 DataFrame에서의 위치를 기준으로 해야 하며,
#     # 제거해야 할 인덱스가 정확히 어떤 것인지 데이터를 확인한 후에 입력해야 합니다.
#     train_x = train_x.drop(index=[11471, 11496, 10175, 10199], errors='ignore')  # 예시 인덱스
#
#     # 상관관계 계산
#     train_correlation = train_x.corr()
#     test_correlation = test_x.corr()
#
#     # 상관관계 출력
#     print("실측 날씨 데이터와 발전량의 상관관계:")
#     print(train_correlation)
#     print("\n날씨 예보 데이터와 발전량의 상관관계:")
#     print(test_correlation)
#     print('a')
# # 함수 실행
# _data_correlator()


import pandas as pd

def _data_correlator():
    # 데이터 로드
    train_x = pd.read_csv('weather_actual.csv')
    test_x = pd.read_csv('weather_forecast.csv')
    pred = pd.read_csv('pred.csv')

    # 결과를 저장할 빈 데이터프레임 초기화
    correlation_df = pd.DataFrame()

    # pred.csv를 model_id로 그룹화하고 각 그룹에 대해 반복
    for model_id, group in pred.groupby('model_id'):
        # 모델별 예측 발전량만 추출
        model_pred = group[['time', 'amount']]

        # train_x(실측 날씨 데이터)와 결합
        merged_train = pd.merge(train_x, model_pred, on='time')
        # test_x(날씨 예보 데이터)와 결합
        merged_test = pd.merge(test_x, model_pred, on='time')

        # 상관관계 계산
        train_correlation = merged_train.corr()['amount'].sort_values(ascending=False).drop('amount')
        test_correlation = merged_test.corr()['amount'].sort_values(ascending=False).drop('amount')

        # 상관관계를 데이터프레임에 추가
        correlation_df[f'Model_{model_id+1}_train'] = train_correlation
        correlation_df[f'Model_{model_id+1}_test'] = test_correlation

    # 모든 모델의 상관도 결과를 출력
    print(correlation_df)
    print('a')
# 함수 실행
_data_correlator()

