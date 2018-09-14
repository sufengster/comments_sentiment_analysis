import multiprocessing
import pandas as pd
import numpy as np
from multiprocessing.dummy import Pool as ThreadPool


def _apply_df(args):
    df, func, kwargs = args
    return df.apply(func, **kwargs)


def apply_by_multiprocessing(df, func, **kwargs):
    workers = kwargs.pop('workers')
    pool = multiprocessing.dummy.Pool(processes=workers) #thread pool
    result = pool.map(_apply_df, [(d, func, kwargs) for d in np.array_split(df, workers)])
    pool.close()
    pool.join()
    return pd.concat(list(result))


def square(x):
    return x ** x


if __name__ == '__main__':
    df = pd.DataFrame({'a': range(10), 'b': range(10)})
    apply_by_multiprocessing(df, square, axis=1, workers=4)