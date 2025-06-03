"""Model Inputs"""
import numpy as np
import sklearn.preprocessing
import pandas as pd
import datetime as dt
import enum

from sklearn.preprocessing import MinMaxScaler

# Type definitions
class DataTypes(enum.IntEnum):
    REAL_VALUED = 0
    CATEGORICAL = 1
    DATE = 2

class InputTypes(enum.IntEnum):
    TARGET = 0
    OBSERVED_INPUT = 1
    KNOWN_INPUT = 2
    STATIC_INPUT = 3
    ID = 4   # Entity identifier
    TIME = 5 # Time index

# … get_single_col_by_input_type, extract_cols_from_data_type 그대로 …

class ModelFeatures:
    """Defines and formats data for the MomentumCp dataset."""
    def __init__(
        self,
        df: pd.DataFrame,
        total_time_steps: int,
        start_boundary: int = 2018,
        test_boundary: int = 2022,
        test_end: int = 2022,
        changepoint_lbws=None,
        train_valid_sliding=False,
        transform_real_inputs=False,
        train_valid_ratio=0.9,
        split_tickers_individually=True,
        add_ticker_as_static=False,
        time_features=False,
        lags=None,
        asset_class_dictionary=None,
        static_ticker_type_feature=False,
    ):
        # 1) 경계일자를 UTC-aware datetime으로 변환
        start_boundary = pd.to_datetime(dt.datetime(start_boundary, 1, 1)).tz_localize("UTC")
        test_boundary  = pd.to_datetime(dt.datetime(test_boundary, 1, 1)).tz_localize("UTC")
        test_end       = pd.to_datetime(dt.datetime(test_end, 1, 1)).tz_localize("UTC")

        # 2) 컬럼 정의: ID 컬럼을 "symbol"로 변경
        self._column_definition = [
            ("symbol", DataTypes.CATEGORICAL, InputTypes.ID),
            ("date",   DataTypes.DATE,        InputTypes.TIME),
            ("target_returns",       DataTypes.REAL_VALUED, InputTypes.TARGET),
            ("norm_daily_return",    DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
            ("norm_monthly_return",  DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
            ("norm_quarterly_return",DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
            ("norm_biannual_return", DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
            ("norm_annual_return",   DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
            ("macd_8_24",   DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
            ("macd_16_48",  DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
            ("macd_32_96",  DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        ]

        # 3) 결측치 제거 & 기간 필터링 (index가 datetime이어야 함)
        df = df.dropna()
        df = df[df.index >= start_boundary].copy()
        times = df.index  # UTC-aware DatetimeIndex

        # 4) 초기 속성 설정
        self.identifiers = None
        self._real_scalers = None
        self._cat_scalers = None
        self._target_scaler = None
        self._num_classes_per_cat_input = None
        self.total_time_steps = total_time_steps
        self.lags = lags

        # 5) CPD lookback window 컬럼 추가 (필요할 때)
        if changepoint_lbws:
            for lbw in changepoint_lbws:
                self._column_definition += [
                    (f"cp_score_{lbw}", DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
                    (f"cp_rl_{lbw}",    DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
                ]

        # 6) (옵션) time_features 처리 생략…

        # 7) static symbol 추가 (옵션)
        if add_ticker_as_static:
            self._column_definition.append(
                ("static_symbol", DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT)
            )
            df["static_symbol"] = df["symbol"].astype(str)
            if static_ticker_type_feature:
                df["static_symbol_type"] = df["symbol"].map(asset_class_dictionary)
                self._column_definition.append(
                    ("static_symbol_type", DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT)
                )

        self.transform_real_inputs = transform_real_inputs

        # 8) train/test split
        test       = df.loc[times >= test_boundary]
        if split_tickers_individually:
            trainvalid = df.loc[times < test_boundary]
            # symbol 단위로 그룹핑
            symbols = trainvalid["symbol"].unique().tolist()
            train, valid = [], []
            for sym in symbols:
                data_sym = trainvalid[trainvalid["symbol"] == sym]
                split_i = int(len(data_sym) * train_valid_ratio)
                train.append(data_sym.iloc[:split_i])
                valid.append(data_sym.iloc[split_i:])
            train = pd.concat(train)
            valid = pd.concat(valid)
            test  = test[test["symbol"].isin(symbols)]
        else:
            # 전체 시계열 기준 split
            dates = np.sort(df.index.unique())
            split_i = int(len(dates) * train_valid_ratio)
            train_dates = dates[:split_i]
            valid_dates = dates[split_i:]
            train = df[df.index.isin(train_dates)]
            valid = df[df.index.isin(valid_dates)]
            test  = test[test["symbol"].isin(valid["symbol"].unique())]

        # 9) buffer 포함 테스트 데이터 생성 (생략…)

        self.tickers = train["symbol"].unique().tolist()
        self.num_tickers = len(self.tickers)
        self.set_scalers(train)

        # 10) 데이터 변환 & 배칭
        self.train = self.transform_inputs(train)
        self.valid = self.transform_inputs(valid)
        self.test_fixed = self.transform_inputs(test)
        # sliding-window 테스트 배칭은 lags 옵션일 때 처리…

    # 이하 set_scalers, transform_inputs, get_column_definition 등은
    # "ticker"→"symbol"만 교체해 주시면 됩니다.
