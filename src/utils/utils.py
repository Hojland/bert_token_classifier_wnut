"""This module contains various helper functions mostly for Python lists.
"""
import logging
import math
import os
import re
import shutil
import sys
import time
import warnings
from datetime import datetime, timedelta
from typing import List

import numpy as np
import pandas as pd
import pytz


def invert_dict(d):
    """Inverts key-value pairs of dictionary.
    New values are lists, since original values can have duplicates.
    Args:
        d (dict): Original dictionary.
    Returns:
        dict: Inverse dictionary.
    """
    inv_d = {}
    for k, v in d.items():
        inv_d[v] = inv_d.get(v, [])
        inv_d[v].append(k)
    return inv_d


def without_keys(d, keys):
    return {x: d[x] for x in d if x not in keys}


def flatten_list(l):
    """Flattens a nested list that may or may not contain strings.
    Args:
        input_list (list): Nested list.
    Returns:
        list: Flattened list.
    """
    obj = []

    def recurse(ll):
        if isinstance(ll, list) or isinstance(ll, np.ndarray):
            for i, _ in enumerate(ll):
                recurse(ll[i])
        else:
            obj.append(ll)

    recurse(l)
    return obj


def get_size(obj):
    """Computes memory size of a Python object and returns it in the appropriate scale(MB,GB,...).
    Args:
        obj (Object): A Python object.
    Returns:
        str: The size of the object
    """
    size_bytes = sys.getsizeof(obj)
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])


def remove_dir(dir_path):
    """Removes folder directory recursively.
    Args:
        dir_path (str): The directory to be deleted.
    """
    try:
        shutil.rmtree(dir_path, ignore_errors=False)
        while os.path.isdir(dir_path):
            pass
        time.sleep(5)
    except Exception as e:
        warnings.warn(str(e))


def remove_contents_of_dir(dir_path):
    """Removes contencts folder directory recursively.
    Args:
        dir_path (str): The directory to be deleted.
    """
    # try:
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            warnings.warn("Failed to delete %s. Reason: %s" % (file_path, e))


class Recursionlimit(object):
    """Sets Python's recursion limit to `limit`."""

    def __init__(self, limit):
        self.limit = limit
        self.old_limit = sys.getrecursionlimit()

    def __enter__(self):
        sys.setrecursionlimit(self.limit)

    def __exit__(self, _type, value, tb):
        sys.setrecursionlimit(self.old_limit)


def get_logger(log_name: str = "/app/logs/hello.log"):
    """Creates new logger.
    Args:
        model_name (str):
            Folder name for the logger to be saved in.
            Accepted values: 'ncf', 'implicit_model'
        model_dir (str): Name of the logger file.
    Returns:
        logger: Logger object.
    """

    def copenhagen_time(*args):
        """Computes and returns local time in Copenhagen.
        Returns:
            time.struct_time: Time converted to CEST.
        """
        _ = args  # to explicitly remove warning
        utc_dt = pytz.utc.localize(datetime.utcnow()) + timedelta(minutes=5, seconds=30)
        local_timezone = pytz.timezone("Europe/Copenhagen")
        converted = utc_dt.astimezone(local_timezone)
        return converted.timetuple()

    logging.Formatter.converter = copenhagen_time
    logger = logging.getLogger()
    if logger.hasHandlers():
        logger.handlers.clear()

    # To files
    fh = logging.FileHandler(log_name)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)
    logger.setLevel(logging.INFO)

    # to std out
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def time_now(local_tz: pytz.timezone = None):
    if not local_tz:
        local_tz = pytz.timezone("Europe/Copenhagen")
    now = datetime.today().replace(tzinfo=pytz.utc).astimezone(tz=local_tz)
    return now


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if "log_time" in kw:
            name = kw.get("log_name", method.__name__.upper())
            kw["log_time"][name] = int((te - ts) * 1000)
        else:
            print("%r  %2.2f ms" % (method.__name__, (te - ts) * 1000))
        return result

    return timed


def flatten_dict(d, sep="_"):
    obj = {}

    def recurse(t, parent_key=""):
        if isinstance(t, list):
            for i, _ in enumerate(t):
                recurse(t[i], parent_key + sep + str(i) if parent_key else str(i))
        elif isinstance(t, dict):
            for k, v in t.items():
                recurse(v, parent_key + sep + k if parent_key else k)
        else:
            obj[parent_key] = t

    if isinstance(d, list):
        res_list = []
        for i, _ in enumerate(d):
            recurse(d[i])
            res_list.append(obj.copy())
        return res_list
    else:
        recurse(d)
    return obj


def multiple_replace(replace_dct: dict, text: str, **kwargs):
    # Create a regular expression  from the dictionary keys
    regex = re.compile("(%s)" % "|".join(map(re.escape, replace_dct.keys())), **kwargs)

    # For each match, look-up corresponding value in dictionary
    return regex.sub(lambda mo: replace_dct[mo.string[mo.start() : mo.end()]], text)


def split_list(lst: list, chunk_size: int):
    return [lst[offs : offs + chunk_size] for offs in range(0, len(lst), chunk_size)]


def merge_dataframelist(dfs: List[pd.DataFrame], **kwargs):
    """merges a list of dataframes using pd.merge
    Args:
        dfs (List[pandas.DataFrame]): A list of dataframes
        **kwargs: Several possible arguments to be passed onto pd.merge
    Returns:
        df (pd.DataFrame): A merged dataframe
    """
    for other_df in dfs:
        if not "df" in locals():
            df = other_df
            continue
        df = pd.merge(df, other_df, **kwargs)
    return df


def date_cat(dates, days: int = 14):
    bins_dt = pd.date_range(min(dates), max(dates) + timedelta(days=days), freq=f"{days}D")
    bins_str = bins_dt.astype(str).values

    labels = ["({}, {}]".format(bins_str[i - 1], bins_str[i]) for i in range(1, len(bins_str))]
    unified_dates = pd.cut(
        dates.astype(np.int64) // 10 ** 9,
        bins=bins_dt.astype(np.int64) // 10 ** 9,
        labels=labels,
        include_lowest=True,
    )
    return unified_dates


def mode(x):
    """
    Calculates a mode of a series from a group by operation
    """
    return np.nan if x.isnull().all() else x.value_counts().index[0]


def strfdelta(tdelta: timedelta, fmt):
    d = {"days": tdelta.days}
    d["hours"], rem = divmod(tdelta.seconds, 3600)
    d["minutes"], d["seconds"] = divmod(rem, 60)
    return fmt.format(**d)


def find_binary(X: pd.DataFrame, get_difference: bool = False):
    bin_cols = []
    for i in range(X.shape[1]):
        if len(np.unique(X.iloc[:, i])) == 2:
            bin_cols.append(X.columns[i])
        else:
            pass

    if get_difference:
        out = X.columns.difference(bin_cols)
    else:
        out = bin_cols
    return out
