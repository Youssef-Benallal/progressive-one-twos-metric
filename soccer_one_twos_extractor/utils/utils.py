import pandas as pd
import numpy as np
import glob
import os
from joblib import Parallel, delayed
from tqdm import tqdm
from soccer_one_twos_extractor.constants.constants import PITCH_LENGTH, PITCH_WIDTH
from soccer_one_twos_extractor.constants.qualifiers_mapping import (
    QUALIFIER_ID_TO_NAME,
    COLUMN_RENAMES,
)
from soccer_one_twos_extractor.constants.fields import F


def list_xml_filenames(folder, pattern="*.xml"):
    """
    Returns a sorted list of XML filenames in the given folder.
    """
    files = glob.glob(os.path.join(folder, pattern))
    filenames = [os.path.basename(f) for f in files]
    return sorted(filenames)


def normalize_coords(
    df: pd.DataFrame, length_m: float = PITCH_LENGTH, width_m: float = PITCH_WIDTH
) -> pd.DataFrame:
    """
    Scale x/y coordinates from 0-100 to meters, in-place.
    """
    for col in [F.X, f"pass_end_{F.X}"]:
        if col in df.columns:
            df[col] = df[col] * length_m / 100
    for col in [F.Y, f"pass_end_{F.Y}"]:
        if col in df.columns:
            df[col] = df[col] * width_m / 100
    return df


def fix_event_id(df: pd.DataFrame, sep: str = "-") -> pd.DataFrame:
    """
    Return a copy of df with an 'action_id' column built as:
    f"{event_id}{sep}{game_id}{sep}{player_id}{sep}{type_id}".
    """
    d = df.copy()
    req = [F.EVENT_ID, F.GAME_ID, F.TEAM_ID, F.PLAYER_ID, F.TYPE_ID]
    miss = [c for c in req if c not in d.columns]
    if miss:
        raise KeyError(f"Missing required columns for action_id: {miss}")
    d[F.ACTION_ID] = d[req].astype("string").fillna("NA").agg(sep.join, axis=1)
    return d


def extract_qualifier(q_list, qualifier_id):
    """
    Returns value for qualifier_id or np.nan if not found.
    """
    for q in q_list:
        if str(q.get(F.QUALIFIER_ID)) == str(qualifier_id):
            return float(q.get("value", np.nan))
    return np.nan


def explode_qualifiers(row):
    quals = pd.json_normalize(row[F.QUALIFIERS]).T.reset_index(drop=True)
    if quals.shape[0] < 2:
        return None
    quals.columns = list(quals.iloc[1])
    quals = quals.drop([0, 1], axis=0)
    quals = quals.rename(columns=QUALIFIER_ID_TO_NAME)
    quals.columns = quals.columns.str.lower().str.replace(" ", "_")
    quals = quals.rename(columns=COLUMN_RENAMES)
    quals[F.GAME_ID] = row[F.GAME_ID]
    quals[F.TEAM_ID] = row[F.TEAM_ID]
    return quals


def format_time(minutes, seconds, fallback="00:00"):
    if pd.notna(minutes) and pd.notna(seconds):
        return f"{int(minutes):02d}:{int(seconds):02d}"
    return fallback


def timestamp_to_minutes(time_str):
    if pd.isna(time_str) or not isinstance(time_str, str):
        return 0.0
    mins, secs = map(int, time_str.split(":"))
    return mins + secs / 60


def parallel_map(func, iterable, n_jobs, desc=None):
    """
    Maps func over iterable in parallel, with optional tqdm progress bar.
    """
    results = Parallel(n_jobs=n_jobs)(
        delayed(func)(x) for x in tqdm(iterable, desc=desc)
    )
    return results
