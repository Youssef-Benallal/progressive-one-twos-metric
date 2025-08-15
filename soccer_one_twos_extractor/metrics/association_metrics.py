# metrics/association_metrics.py

import pandas as pd
from soccer_one_twos_extractor.constants.fields import F


class AssociationMetrics:
    """
    Detect progressive one–twos (A→B then B→A) from an event dataframe.

    Parameters
    ----------
    min_progression_ratio_pct : float, default 0.25
        Minimum progression (fraction) required to accept an exchange.
        Progression features are precomputed upstream.
    max_time_diff : float, default 5.0
        Maximum allowed time (in seconds) between pass A and pass B.
    max_player_b_ball_carry_distance : float, default 7.0
        Maximum allowed carry distance by player B before returning the pass.
    """

    def __init__(
        self,
        min_progression_ratio_pct=0.25,
        max_time_diff=5.0,
        max_player_b_ball_carry_distance=7.0,
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
        # Keep completed passes with a known receiver (same team, non-self).
        d = df[
            (df[F.EVENT_NAME] == "Pass")
            & (df[F.OUTCOME] == "1")
            & (df[f"pass_receiver_{F.PLAYER_ID}"].notna())
        ].copy()

        # Build absolute time (seconds) and sort within game/period.
        d[F.TIME] = d[F.MINUTE] * 60 + d[F.SECOND]
        d = d.sort_values([F.GAME_ID, F.PERIOD_ID, F.TIME])

        # Rename current row as "A" (opening pass).
        # Receiver becomes PLAYER_ID_B.
        d = d.rename(
            columns={
                F.PLAYER_ID: f"{F.PLAYER_ID}_A",
                F.ACTION_ID: f"{F.ACTION_ID}_A",
                F.TIME: f"{F.TIME}_A",
                F.X: f"{F.X}_A",
                F.Y: f"{F.Y}_A",
                F.KEY_PASS: f"{F.KEY_PASS}_A",
                f"pass_end_{F.X}": f"pass_end_{F.X}_A",
                f"pass_end_{F.Y}": f"pass_end_{F.Y}_A",
                f"pass_receiver_{F.PLAYER_ID}": f"{F.PLAYER_ID}_B",
            }
        )

        # Next row is the candidate returning pass "B".
        dn = d.shift(-1)

        # Copy next-row features into "B" fields.
        for col in [
            f"{F.X}_A",
            f"{F.Y}_A",
            f"{F.TIME}_A",
            f"{F.ACTION_ID}_A",
            f"{F.KEY_PASS}_A",
            f"pass_end_{F.X}_A",
            f"pass_end_{F.Y}_A",
        ]:
            d[col.replace("_A", "_B")] = dn[col]

        # Shortcuts to precomputed features and timing.
        prog_center = d["prog_goal_center"]
        prog_goal_line = d["prog_goal_line"]
        carry_dist = d["receiver_carry_distance"]
        time_diff = d[f"{F.TIME}_B"] - d[f"{F.TIME}_A"]

        # Keep only true one–twos:
        # next passer == A's receiver, next receiver == A's passer,
        # same team, same period, progressive, quick, and short carry.
        mask = (
            (dn[f"{F.PLAYER_ID}_A"] == d[f"{F.PLAYER_ID}_B"])
            & (d[f"{F.PLAYER_ID}_A"] == dn[f"{F.PLAYER_ID}_B"])
            & (dn[F.TEAM_ID] == d[F.TEAM_ID])
            & (dn[F.PERIOD_ID] == d[F.PERIOD_ID])
            & (
                (prog_center >= self.min_progression_ratio_pct)
                | (prog_goal_line >= self.min_progression_ratio_pct)
            )
            & (time_diff <= self.max_time_diff)
            & (carry_dist <= self.max_player_b_ball_carry_distance)
        )

        # Add next key pass flag
        t = df.sort_values(
            [F.GAME_ID, F.TEAM_ID, F.MINUTE, F.SECOND, F.PERIOD_ID]
        ).copy()
        t["kp_next"] = t.groupby([F.GAME_ID, F.TEAM_ID, F.PERIOD_ID])[F.KEY_PASS].shift(
            -1
        )

        d[f"{F.KEY_PASS}_next_B"] = (
            d[f"{F.ACTION_ID}_B"].map(t.set_index(F.ACTION_ID)["kp_next"]).fillna("0")
        )

        # Return only rows that matched the one–two definition.
        return d[mask].copy()
