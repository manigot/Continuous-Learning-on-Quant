import argparse
import datetime as dt
from typing import List

import pandas as pd

from data.pull_data import pull_binance_sample_data_ft
from settings.default import (
    BINANCE_SYMBOLS,
    CPD_BINANCE_OUTPUT_FOLDER,
    FEATURES_BINANCE_FILE_PATH,
)
from mom_trans.data_prep import ( 
    deep_momentum_strategy_features_crypto,
    include_changepoint_features,#TODO  변경 예정
)

def main(
    symbols: List[str],
    cpd_module_folder: str,
    lookback_window_length: int,
    output_file_path: str,
    extra_lbw: List[int],
):
    # 티커(또는 심볼)별 기본 모멘텀 피처 생성
    features = pd.concat(
        [
            deep_momentum_strategy_features(
                pull_binance_sample_data_ft(symbol)
            ).assign(
                symbol=symbol
            )
            for symbol in symbols
        ]
    )

    # CPD(Changepoint Detection) 적용 여부
    if lookback_window_length:
        # 메인 LBW CPD 피처 병합
        features_w_cpd = include_changepoint_features(
            features, cpd_module_folder, lookback_window_length
        )

        # 추가 LBW 리스트가 있으면 계속 병합
        if extra_lbw:
            for extra in extra_lbw:
                extra_data = pd.read_csv(
                    output_file_path.replace( # TODO file update
                        f"quandl_cpd_{lookback_window_length}lbw.csv",
                        f"quandl_cpd_{extra}lbw.csv",
                    ),
                    index_col=0,
                    parse_dates=True,
                ).reset_index()[ # TODO folder update
                    ["date", "ticker", f"cp_rl_{extra}", f"cp_score_{extra}"]
                ]
                extra_data["date"] = pd.to_datetime(extra_data["date"]) # TODO not date

                # 기존 피처에 추가 LBW CPD 피처 병합
                features_w_cpd = pd.merge(
                    features_w_cpd.set_index(["date", "ticker"]),
                    extra_data.set_index(["date", "ticker"]),
                    left_index=True,
                    right_index=True,
                ).reset_index()
                features_w_cpd.index = features_w_cpd["date"]
                features_w_cpd.index.name = "Date"
        else:
            features_w_cpd.index.name = "Date"
        # 최종 파일 저장
        features_w_cpd.to_csv(output_file_path)
    else:
        # CPD 없이 기본 피처 저장
        features.to_csv(output_file_path)


if __name__ == "__main__":

    def get_args():
        parser = argparse.ArgumentParser(
            description="Run changepoint detection module for Binance data"
        )
        parser.add_argument(
            "lookback_window_length",
            metavar="l",
            type=int,
            nargs="?",
            default=None,
            help="Changepoint lookback window length (e.g., 96, 480)."
        )
        parser.add_argument(
            "extra_lbw",
            metavar="-e",
            type=int,
            nargs="*",
            default=[],
            help="Additional lookback windows to merge CPD features."
        )

        args = parser.parse_known_args()[0]

        return (
            BINANCE_SYMBOLS,
            CPD_BINANCE_OUTPUT_FOLDER(args.lookback_window_length),
            args.lookback_window_length,
            FEATURES_BINANCE_FILE_PATH(args.lookback_window_length),
            args.extra_lbw,
        )

    main(*get_args())