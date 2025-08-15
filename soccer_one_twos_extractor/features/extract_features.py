# features/extract_features.py

import numpy as np
import pandas as pd
from soccer_one_twos_extractor.constants.constants import PITCH_LENGTH, PITCH_WIDTH
from soccer_one_twos_extractor.constants.qualifiers_mapping import QUALIFIER_NAME_TO_ID
from soccer_one_twos_extractor.utils.utils import extract_qualifier
from soccer_one_twos_extractor.constants.fields import F


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
