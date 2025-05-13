import numpy as np
import pandas as pd
from mom_trans.data_prep import (
    calc_returns,
    calc_daily_vol,
    calc_vol_scaled_returns,
    MACDStrategy,
)
from settings.default import HALFLIFE_WINSORISE, VOL_THRESHOLD


def deep_momentum_strategy_features_crypto(df_asset: pd.DataFrame) -> pd.DataFrame:
    """
    15분 봉 Crypto 데이터용 모멘텀 전략 피처 생성기.
    Args:
        df_asset: index가 datetime, 컬럼에 ['Opn','Hgh','Low','Cls','Vol','NoT','TBV','Ret'] 보유
    Returns:
        피처가 추가된 DataFrame
    """
    df = df_asset.copy()

    # 4) 수익률·변동성·타깃 리턴
    #    - calc_returns는 pct_change 기반
    df["srs"] = df["Cls"]
    ewm = df["srs"].ewm(halflife=HALFLIFE_WINSORISE)
    means = ewm.mean()
    stds = ewm.std()
    df["srs"] = np.minimum(df["srs"], means + VOL_THRESHOLD * stds)
    df["srs"] = np.maximum(df["srs"], means - VOL_THRESHOLD * stds)


    df["MRt"] = calc_returns(df["srs"])
    df["MVo"] = calc_daily_vol(df["MRt"])
    df["RTg"] = (
        calc_vol_scaled_returns(df["MRt"], df["MVo"])
        .shift(-1)
    )

    # 5) 정규화된 여러 기간 리턴
    def norm_ret(n):
        return (
            calc_returns(df["srs"], n)
            / df["MVo"]
            / np.sqrt(n)
        )

    df["R15"]    = norm_ret(1)   # 1틱(15분) 기준
    df["R60"]   = norm_ret(4)   # 1시간 = 4틱
    df["R4h"]       = norm_ret(16)  # 4시간 = 16틱
    df["R1d"]    = norm_ret(96)  # 1일 = 96틱
    df["R1w"]   = norm_ret(672) # 1주 = 672틱

    # 6) MACD 스타일 트렌드 피처 (short, long in ticks)
    trend_windows = [(4,16), (16,64), (32,128)]  # 예: 1h vs 4h, 4h vs 16h ...
    for s, l in trend_windows:
        df[f"macd_{s}_{l}"] = MACDStrategy.calc_signal(df["srs"], s, l)

    # 7) 시간 인코딩
    idx = pd.to_datetime(df.index)
    df["minute"]      = idx.minute
    df["hour"]        = idx.hour
    df["day_of_week"] = idx.dayofweek
    df["month"]       = idx.month
    # sin/cos encoding
    df["sin_min"]  = np.sin(2*np.pi * df["minute"] / 60)
    df["cos_min"]  = np.cos(2*np.pi * df["minute"] / 60)
    df["sin_hour"] = np.sin(2*np.pi * df["hour"]   / 24)
    df["cos_hour"] = np.cos(2*np.pi * df["hour"]   / 24)

    # 8) 인덱스와 date 컬럼 정리
    df.index.name='Time'

    return df.dropna()
