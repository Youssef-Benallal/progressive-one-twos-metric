import glob
import os
from typing import Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from soccer_one_twos_extractor.constants.constants import PITCH_LENGTH, PITCH_WIDTH
from soccer_one_twos_extractor.constants.fields import F
from soccer_one_twos_extractor.constants.qualifiers_mapping import (
    COLUMN_RENAMES,
    QUALIFIER_ID_TO_NAME,
)


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


def first_position(pos: Optional[str]) -> str:
    """
    Return the first (primary) position token from a comma-separated string.

    Parameters
    ----------
    pos : str or None
        A position label such as "CM,DM" or "LW". If None or not a string,
        returns an empty string.

    Returns
    -------
    str
        The first position token, stripped of whitespace. Empty string if input
        is None/non-string.

    Examples
    --------
    >>> first_position("CM,DM")
    'CM'
    >>> first_position("  LW ")
    'LW'
    >>> first_position(None)
    ''
    """
    if isinstance(pos, str):
        return pos.split(",")[0].strip()
    return ""


def get_pair_positions(one_twos_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a unique, undirected list of one–two player pairs with their positions.

    Given an event-level dataframe that contains both the initiating (A) and
    returning (B) player names/positions and the team name, this function
    returns a de-duplicated table of pairs, treating (A,B) and (B,A) as the
    same pair for a given team.

    Parameters
    ----------
    one_twos_df : pd.DataFrame
        Must include the columns:
        - "player_A_name", "player_B_name"
        - "team_name"
        - "player_A_position", "player_B_position"

    Returns
    -------
    pd.DataFrame
        Columns:
        - "player_A_name", "player_B_name", "team_name",
          "player_A_position", "player_B_position"
        Each row is a unique pair for that team (order-independent).

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     "player_A_name": ["A", "B"],
    ...     "player_B_name": ["B", "A"],
    ...     "team_name": ["Team", "Team"],
    ...     "player_A_position": ["CM,DM", "ST"],
    ...     "player_B_position": ["ST", "CM,DM"],
    ... })
    >>> out = get_pair_positions(df)
    >>> set(map(tuple, out[["player_A_name","player_B_name","team_name"]].values))
    {('A', 'B', 'Team')}
    """
    pos1 = one_twos_df[
        [
            "player_A_name",
            "player_B_name",
            "team_name",
            "player_A_position",
            "player_B_position",
        ]
    ].drop_duplicates()

    # Swap A/B to make the pair undirected, then unify column names
    pos2 = one_twos_df[
        [
            "player_B_name",
            "player_A_name",
            "team_name",
            "player_B_position",
            "player_A_position",
        ]
    ].drop_duplicates()
    pos2.columns = pos1.columns

    # Concatenate and drop duplicates so (A,B) and (B,A) are treated the same
    return pd.concat([pos1, pos2]).drop_duplicates(
        subset=["player_A_name", "player_B_name", "team_name"]
    )


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
