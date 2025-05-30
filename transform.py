import pandas as pd
import numpy as np

dtypes = {
    'Time': np.int32,
    'Opn':  np.float32,
    'Hgh':  np.float32,
    'Low':  np.float32,
    'Cls':  np.float32,
    'Vol':  np.float32,
    'NoT':  np.int32,
    'TBV':  np.float32,
}

# coin = ["BTCUSDT", "ETHUSDT"]
# purpose = ["gen15m", "online15m"]

# for c in coin:
#     for p in purpose:
#         fname = f'./datasets/{p}/{c}'
#         df = pd.read_csv(f"{fname}.csv", dtype = dtypes)

#         df['Ret'] = df['Cls'].pct_change().fillna(0).astype('float32')
        
#         df.to_csv(f"{fname}.csv", index = False)
#         df.to_parquet(f"{fname}.parquet", index=False, compression='snappy')

#         df = pd.read_parquet(f"{fname}.parquet")
#         print(df.dtypes)



# max min 확인

df = pd.read_csv(f"./datasets/gen15m/ETHUSDT.csv", dtype = dtypes)
# 컬럼별 min·max 요약
min_max = df.agg(['min', 'max'])


"""
i64 to i32
# 1) 명시적으로 astype 이용하기
df['Time'] = df['Time'].astype('int32')
df['NoT']  = df['NoT'].astype('int32')

# 2) 혹은 downcast 옵션 사용하기
df['Time'] = pd.to_numeric(df['Time'], downcast='integer')
df['NoT']  = pd.to_numeric(df['NoT'],  downcast='integer')

# 변환 확인
print(df.dtypes[['Time','NoT']])
"""

"""
f64 to f32
# Opn, Hgh, Low, Cls, Vol, TBV 컬럼을 float32로 변환
float_cols = ['Opn', 'Hgh', 'Low', 'Cls', 'Vol', 'TBV']
df[float_cols] = df[float_cols].astype('float32')

# 변환 결과 확인
print(df.dtypes[float_cols])
"""

# # CSV 로드
# fname = './datasets/online15m/BTCUSDT.csv'
# df = pd.read_csv(fname)

# # 2) Time: ms → sec, int32 다운캐스트
# df['Time'] = (df['Time'] // 1000).astype('int32')

# # 3) NoT: int64 → int32 다운캐스트
# df['NoT'] = df['NoT'].astype('int32')

# # 4) 나머지 컬럼: float64 → float32 다운캐스트
# float_cols = ['Opn', 'Hgh', 'Low', 'Cls', 'Vol', 'TBV']
# df[float_cols] = df[float_cols].astype('float32')

# # 5) 변환된 타입 확인 (옵션)
# print(df.dtypes[['Time', 'NoT'] + float_cols])

# # 6) 같은 파일에 덮어쓰기
# df.to_csv(fname, index=False)