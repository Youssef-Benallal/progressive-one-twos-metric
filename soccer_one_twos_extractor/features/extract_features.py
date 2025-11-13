# features/extract_features.py

import numpy as np
import pandas as pd

from soccer_one_twos_extractor.constants.constants import PITCH_LENGTH, PITCH_WIDTH
from soccer_one_twos_extractor.constants.fields import F
from soccer_one_twos_extractor.constants.qualifiers_mapping import QUALIFIER_NAME_TO_ID
from soccer_one_twos_extractor.utils.utils import extract_qualifier


class FeaturesExtractor:
    def __init__(self):
        self.pass_end_x_id = QUALIFIER_NAME_TO_ID[F.PASS_END_X]
        self.pass_end_y_id = QUALIFIER_NAME_TO_ID[F.PASS_END_Y]
        self.length_id = QUALIFIER_NAME_TO_ID[F.PASS_LENGTH]

    def pass_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds a 'pass_receiver_player_id' column, which is the next player_id
        if the pass is complete and goes to a teammate, otherwise NaN.
        """
        df = df.copy()
        df = df.assign(
            pass_end_x=df[F.QUALIFIERS].apply(
                lambda q: extract_qualifier(q, self.pass_end_x_id)
            ),
            pass_end_y=df[F.QUALIFIERS].apply(
                lambda q: extract_qualifier(q, self.pass_end_y_id)
            ),
            length=df[F.QUALIFIERS].apply(
                lambda q: extract_qualifier(q, self.length_id)
            ),
        )
        next_player, next_team = (df[F.PLAYER_ID].shift(-1), df[F.TEAM_ID].shift(-1))
        is_valid = (
            (df[F.EVENT_NAME] == "Pass")
            & (df[F.OUTCOME].astype(int) == 1)
            & (df[F.TEAM_ID] == next_team)
            & (df[F.PLAYER_ID] != next_player)
        )
        df["pass_receiver_player_id"] = np.where(is_valid, next_player, np.nan)
        return df

    def progression_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add progression and carry distance features for each pass.
        Assumes df is sorted chronologically within each match.
        """
        d = df.copy()
        dn = d.shift(-1)  # Next event

        # Distance from start of pass to goal center
        start_distto_goal_center = np.hypot(
            PITCH_LENGTH - d[F.X], (PITCH_WIDTH / 2) - d[F.Y]
        )
        # Distance from end of pass to goal center
        end_distto_goal_center = np.hypot(
            PITCH_LENGTH - dn[f"pass_end_{F.X}"],
            (PITCH_WIDTH / 2) - dn[f"pass_end_{F.Y}"],
        )
        d["prog_goal_center"] = (
            start_distto_goal_center - end_distto_goal_center
        ) / start_distto_goal_center

        # Distance from start and end of pass to goal line
        start_distto_goal_line = PITCH_LENGTH - d[F.X]
        end_distto_goal_line = PITCH_LENGTH - dn[f"pass_end_{F.X}"]
        d["prog_goal_line"] = (
            start_distto_goal_line - end_distto_goal_line
        ) / start_distto_goal_line

        # Carry distance: from pass end to next event start
        carry_dx = dn[F.X] - d[f"pass_end_{F.X}"]
        carry_dy = dn[F.Y] - d[f"pass_end_{F.Y}"]
        d["receiver_carry_distance"] = np.hypot(carry_dx, carry_dy)

        return d

    def next_chance_creation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add next-event chance-creation context per team/period:
          - keypass_next, assist_next
          - secs_to_next_team_shot
          - shot_within_6s_after (binary)
        """
        df_ = df.copy()
        if F.ASSIST not in df_.columns:
            df_[F.ASSIST] = "0"

        df_[F.TIME] = df_[F.MINUTE] * 60 + df_[F.SECOND]
        df_ = df_.sort_values([F.GAME_ID, F.TEAM_ID, F.PERIOD_ID, F.TIME, F.ACTION_ID])

        g = df_.groupby([F.GAME_ID, F.TEAM_ID, F.PERIOD_ID], sort=False)

        df_["keypass_next"] = g[F.KEY_PASS].shift(-1).fillna("0")
        df_["assist_next"] = g[F.ASSIST].shift(-1).fillna("0")

        is_shot = pd.to_numeric(df_[F.TYPE_ID], errors="coerce").isin([13, 14, 15, 16])
        df_["_shot_time"] = np.where(is_shot, df_[F.TIME], np.nan)
        df_["_next_shot_t"] = g["_shot_time"].transform("bfill")

        df_["secs_to_next_team_shot"] = df_["_next_shot_t"] - df_[F.TIME]
        df_["shot_within_6s_after"] = (
            df_["secs_to_next_team_shot"].le(6).fillna(False).astype(int)
        )

        return df_.drop(columns=["_shot_time", "_next_shot_t"])
