import dask.dataframe as dd
import pandas as pd
import numpy as np
import time
from typing import Union
from _structures import (
    _single_partition_build_stdbars,
    _single_partition_build_imbbars
)

# TODO Check if running bar is empty before appending it
def build_bars(trades: Union[dd.DataFrame, pd.DataFrame],
               structure_type: str,
               bar_type: str,
               threshold: Union[int, float] = None,
               win_size_bv: int = None,
               win_size_T: int = None,
               init_expected_T = None,
               compute: bool = False):

    if structure_type not in ['std', 'imb', 'run']:
        raise ValueError(f'Invalid structure_type: {structure_type}.')


    if not ['time', 'price', 'volume'] == list(trades.columns):
        raise AttributeError(
            f'trades DataFrame must contain only the following columns in the given order:  time, price, volume.'
        )

    columns = ['time', 'open', 'high', 'low', 'close', 'volume']

    if structure_type == 'std':
        build_func = _single_partition_build_stdbars
        build_args = (bar_type, threshold)
    elif structure_type == 'imb':
        build_func = _single_partition_build_imbbars
        build_args = (bar_type, win_size_bv, win_size_T, init_expected_T)
    elif structure_type == 'run':
        return

    partition_build_state = dict()
    if isinstance(trades, dd.DataFrame):
        bars = []
        for part in trades.partitions:
            values = part.compute().to_numpy()
            partition_bars, partition_build_state = build_func(values, *build_args, partition_build_state)
            if partition_bars:
                bars.append(dd.from_array(np.array(partition_bars), columns=columns))

        if bars:
            bars = dd.concat([
                *bars,
                dd.from_pandas(
                    pd.DataFrame([partition_build_state['running_bar'].get_ohlcv()], columns=columns), npartitions=1
                )
            ])
        else:
            return pd.DataFrame([partition_build_state['running_bar'].get_ohlcv()], columns=columns)

        if compute:
            bars = bars.compute()

    if isinstance(trades, pd.DataFrame):
        values = trades.to_numpy()
        bars, partition_build_state = build_func(values, *build_args, partition_build_state)

        if bars:
            bars = pd.concat([
                bars, pd.DataFrame([partition_build_state['running_bar'].get_ohlcv()], columns=columns)
            ])
        else:
            return pd.DataFrame([partition_build_state['running_bar'].get_ohlcv()], columns=columns)

    return bars.reset_index(drop=True)




def build_stdbars(trades: Union[dd.DataFrame, pd.DataFrame],
                  bar_type: str,
                  threshold: Union[int, float] = None,
                  compute: bool = False):

    if bar_type not in ['time', 'tick', 'volume', 'dollar']:
        raise ValueError(f'Invalid bar_type: {bar_type}.')




def build_imbbars(trades: Union[dd.DataFrame, pd.DataFrame],
                  bar_type: str,
                  win_size_b: int,
                  win_size_T: int,
                  init_expected_T: int,
                  compute: bool = False):

    if bar_type not in ['tick', 'volume', 'dollar']:
        raise ValueError(f'Invalid bar_type: {bar_type}.')




def build_runbars(trades: Union[dd.DataFrame, pd.DataFrame],
                  bar_type: str,
                  win_size_b: int,
                  win_size_T: int,
                  init_expected_T: int,
                  compute: bool = False):

    if bar_type not in ['tick', 'volume', 'dollar']:
        raise ValueError(f'Invalid bar_type: {bar_type}.')




if __name__ == '__main__':

    path = 'D:/finml/data/binance/ETHUSDT/trades/*.?.parquet'
    ddf1 = dd.read_parquet(path)
    # path = 'C:/Users/grega/Desktop/trades/ETHUSDT-trades-2021-03-01.csv'
    # ddf1 = dd.read_csv(path, header=None, usecols=[1,2,4], names=['price', 'volume', 'time'])
    # ddf1 = ddf1[['time', 'price', 'volume']]
    # print(ddf1)

    start_time = time.time()
    res1 = build_bars(ddf1, 'imb', 'tick', win_size_bv=20, win_size_T=20, init_expected_T=1000, compute=True)
    print(res1)
    print(time.time() - start_time)
