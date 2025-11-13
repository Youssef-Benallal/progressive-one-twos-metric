# metrics/association_metrics.py

import pandas as pd

from soccer_one_twos_extractor.constants.fields import F


class AssociationMetrics:
    """
    Detect progressive one–twos (A→B then B→A) from an event dataframe.

    Parameters
    ----------
    min_progression_ratio_pct : float
        Minimum progression (fraction) required to accept an exchange.
        Progression features are precomputed upstream.
    max_time_diff : float
        Maximum allowed time (in seconds) between pass A and pass B.
    max_player_b_ball_carry_distance : float
        Maximum allowed carry distance by player B before returning the pass.
    """

    def __init__(
        self,
        min_progression_ratio_pct: float,
        max_time_diff: float,
        max_player_b_ball_carry_distance: float,
    ):
        self.min_progression_ratio_pct = min_progression_ratio_pct
        self.max_time_diff = max_time_diff
        self.max_player_b_ball_carry_distance = max_player_b_ball_carry_distance

    def get_progressive_one_twos(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect progressive one–twos in a feature-enriched event DataFrame.

        Notes
        -----
        - Assumes `df` already contains:
            * pass-end coordinates (`pass_end_x`, `pass_end_y`)
            * `pass_receiver_player_id`
            * progression metrics: `prog_goal_center`, `prog_goal_line`
            * `receiver_carry_distance`
            * `action_id`, `keypass`
        - Progression threshold is applied using the precomputed columns.
        - The function pairs each pass A with the very next event (candidate B),
          then keeps rows that satisfy identity swap, same team/period, progression,
          time, and carry constraints.

        Returns
        -------
        pd.DataFrame
            One row per accepted A→B, B→A exchange, including:
            - A/B-suffixed coordinates/times/action ids/keypass
            - `KEY_PASS_next_B`: whether the event right after B is a key pass
        """
        # PREPROCESSING
        lookup = df.set_index(F.ACTION_ID)
        d, dn = self._prep_one_two_candidates(df)

        # PROG ONE TWOS DEFINITION
        prog_onetwos_mask = (
            # next passer is A's receiver (B)
            (dn[f"{F.PLAYER_ID}_A"] == d[f"{F.PLAYER_ID}_B"])
            &
            # next receiver is the original passer (A)
            (d[f"{F.PLAYER_ID}_A"] == dn[f"{F.PLAYER_ID}_B"])
            &
            # same team (no turnover)
            (dn[F.TEAM_ID] == d[F.TEAM_ID])
            &
            # same period of play
            (dn[F.PERIOD_ID] == d[F.PERIOD_ID])
            &
            # progression meets the threshold (goal center or goal line)
            (
                (d["prog_goal_center"] >= self.min_progression_ratio_pct)
                | (d["prog_goal_line"] >= self.min_progression_ratio_pct)
            )
            &
            # quick exchange (time constraint)
            (d[f"{F.TIME}_B"] - d[f"{F.TIME}_A"] <= self.max_time_diff)
            &
            # minimal ball carry of player B before return (one two closing)
            (d["receiver_carry_distance"] <= self.max_player_b_ball_carry_distance)
        )

        # # attach chance-creation context for the return pass (B)
        d[f"{F.KEY_PASS}_next_B"] = (
            d[f"{F.ACTION_ID}_B"].map(lookup["keypass_next"]).fillna("0")
        )
        d[f"{F.ASSIST}_next_B"] = (
            d[f"{F.ACTION_ID}_B"].map(lookup["assist_next"]).fillna("0")
        )
        d["shot_within_6s_after_B"] = (
            d[f"{F.ACTION_ID}_B"]
            .map(lookup["shot_within_6s_after"])
            .fillna(0)
            .astype(int)
        )
        d["secs_to_next_team_shot_B"] = d[f"{F.ACTION_ID}_B"].map(
            lookup["secs_to_next_team_shot"]
        )

        # Return only rows that matched the one–two definition.
        return d[prog_onetwos_mask].copy()

    @staticmethod
    def _prep_one_two_candidates(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Filter, time-sort, and build A/B fields for one-two detection.

        Includes:
        • Only successful open-play passes
        • Excludes set-piece or non-open-play passes (free kicks, corners, etc.)
        • Builds A/B (passer/receiver) pairing with time-ordered shift
        """
        # IDs to exclude (non–open-play passes)
        exclude_ids = {
            "5",
            "6",
            "107",
            "157",
            "3",
            "168",
        }  # FK, corner, throw-in, launch, headpass, flick-on

        # --- ensure key fields exist
        fields = [
            F.TIME,
            F.X,
            F.Y,
            F.ACTION_ID,
            F.KEY_PASS,
            F.ASSIST,
            f"pass_end_{F.X}",
            f"pass_end_{F.Y}",
        ]

        # --- base filtering: successful open-play passes only ---
        passes = df.loc[
            (df[F.EVENT_NAME].str.lower() == "pass")
            & (df[F.OUTCOME] == "1")
            & (df[f"pass_receiver_{F.PLAYER_ID}"].notna())
            & (
                ~df["qualifiers"].apply(
                    lambda q: any(
                        str(item.get("qualifier_id")) in exclude_ids
                        for item in (q or [])
                    )
                )
            )
        ].copy()

        # --- construct A/B pairing ---
        d = (
            passes.assign(**{F.ASSIST: passes.get(F.ASSIST, "0")})
            .assign(**{F.TIME: lambda x: x[F.MINUTE] * 60 + x[F.SECOND]})
            .sort_values([F.GAME_ID, F.PERIOD_ID, F.TIME])
            .rename(columns={c: f"{c}_A" for c in [F.PLAYER_ID, *fields]})
        )

        # next-pass receiver = player B
        d[f"{F.PLAYER_ID}_B"] = d.pop(f"pass_receiver_{F.PLAYER_ID}")

        # shifted version to get "next action"
        dn = d.shift(-1)

        # copy over corresponding B-fields from the next pass
        d[[f"{c}_B" for c in fields]] = dn[[f"{c}_A" for c in fields]].to_numpy()

        return d, dn
