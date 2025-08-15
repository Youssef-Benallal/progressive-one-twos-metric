import numpy as np
import pandas as pd


class PlayerPairsDataLoader:
    @staticmethod
    def get_duos_common_minutes(association_metric_df, players_data):
        """
        Compute total minutes two players (involved in one-twos) shared on the pitch.

        Parameters:
            association_metric_df: DataFrame with ['player_A_name', 'player_B_name'] for one-two events.
            players_df: DataFrame with player info per match, including on/off times.

        Returns:
            DataFrame with ['player_1', 'player_2', 'team_name', 'common_minutes'].
        """

        # Build unique, unordered pairs from one-two events
        one_two_pairs = {
            tuple(sorted([a, b]))
            for a, b in zip(
                association_metric_df["player_A_name"],
                association_metric_df["player_B_name"],
            )
        }

        # Convert player on/off times to seconds
        for col in ["player_on", "player_off"]:
            t = players_data[col].str.split(":", expand=True).astype(int)
            players_data[f"{col}_s"] = t[0] * 60 + t[1]

        # Merge players from the same match and team to find all potential pairs
        cm = players_data.merge(
            players_data, on=["game_id", "team_name"], suffixes=("_x", "_y")
        )

        # Keep only unique player combinations (no self-pairing, no duplicates)
        cm = cm[cm["player_id_x"] < cm["player_id_y"]]
        cm = cm.dropna(subset=["player_x", "player_y"])

        # Create a unique pair key (unordered)
        cm["pair"] = list(
            map(
                lambda a, b: tuple(sorted((str(a), str(b)))),
                cm["player_x"],
                cm["player_y"],
            )
        )

        # Calculate overlapping minutes on the pitch between player pairs
        cm["overlap"] = (
            np.minimum(cm["player_off_s_x"], cm["player_off_s_y"])
            - np.maximum(cm["player_on_s_x"], cm["player_on_s_y"])
        ).clip(lower=0) / 60

        # Filter to only include pairs involved in one-twos
        cm = cm[cm["pair"].isin(one_two_pairs)]

        # Aggregate total common minutes per pair and team
        result = (
            cm.groupby(["pair", "team_name"], as_index=False)["overlap"]
            .sum()
            .rename(columns={"overlap": "common_minutes"})
        )

        # Split pair tuple into two separate player columns
        result[["player_1", "player_2"]] = pd.DataFrame(
            result["pair"].tolist(), index=result.index
        )

        return result[["player_1", "player_2", "team_name", "common_minutes"]]
