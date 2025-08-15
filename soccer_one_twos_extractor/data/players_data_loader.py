import pandas as pd
import os
from soccer_one_twos_extractor.utils.utils import (
    explode_qualifiers,
    format_time,
    timestamp_to_minutes,
)
from soccer_one_twos_extractor.constants.fields import F


class PlayerDataLoader:
    def __init__(self, data_folder, external_players_file):
        self.data_folder = data_folder
        self.external_players_file = external_players_file

    def load_external_players_data(self):
        return pd.read_csv(
            os.path.join(self.data_folder, self.external_players_file),
            dtype={F.JERSEY_NUMBER: str},
        )

    def get_players_jersey_data(self, match_events):
        df = match_events.copy()
        team_setup = (
            df[df[F.EVENT_NAME] == "Team set up"]
            .apply(explode_qualifiers, axis=1)
            .dropna()
        )
        team_setup = pd.concat(team_setup.values, ignore_index=True)

        players_jersey_data = team_setup.assign(
            player_id=team_setup[F.PLAYER_ID].astype(str).str.split(r",\s*"),
            jersey_number=team_setup[F.JERSEY_NUMBER].astype(str).str.split(r",\s*"),
        ).explode([F.PLAYER_ID, F.JERSEY_NUMBER])

        players_jersey_data = df.merge(
            players_jersey_data, on=[F.GAME_ID, F.TEAM_ID, F.PLAYER_ID], how="left"
        )[
            [F.GAME_ID, F.MATCH_NAME, F.TEAM_ID, F.PLAYER_ID, F.JERSEY_NUMBER]
        ].drop_duplicates()

        return players_jersey_data.dropna(subset=[F.PLAYER_ID])

    def get_players_minutes_data(self, match_events):
        df_ = match_events.copy()
        players_on = (df_[df_[F.EVENT_NAME] == "Player on"])[
            [F.GAME_ID, F.TEAM_ID, F.PLAYER_ID, F.MINUTE, F.SECOND]
        ].rename(columns={F.MINUTE: f"{F.MINUTE}_on", F.SECOND: f"{F.SECOND}_on"})

        players_off = (df_[df_[F.EVENT_NAME] == "Player off"])[
            [F.GAME_ID, F.TEAM_ID, F.PLAYER_ID, F.MINUTE, F.SECOND]
        ].rename(columns={F.MINUTE: f"{F.MINUTE}_off", F.SECOND: f"{F.SECOND}_off"})

        df_ = df_.merge(
            players_on, on=[F.GAME_ID, F.TEAM_ID, F.PLAYER_ID], how="left"
        ).merge(players_off, on=[F.GAME_ID, F.TEAM_ID, F.PLAYER_ID], how="left")

        end_mask = df_[F.EVENT_NAME] == "End"
        end_min, end_sec = (
            df_.loc[end_mask, [F.MINUTE, F.SECOND]]
            .sort_values([F.MINUTE, F.SECOND])
            .iloc[-1]
            if end_mask.any()
            else (0, 0)
        )

        df_[["player_on", "player_off"]] = [
            [
                format_time(a, b, "00:00"),
                format_time(c, d, f"{int(end_min):02d}:{int(end_sec):02d}"),
            ]
            for a, b, c, d in zip(
                df_[f"{F.MINUTE}_on"],
                df_[f"{F.SECOND}_on"],
                df_[f"{F.MINUTE}_off"],
                df_[f"{F.SECOND}_off"],
            )
        ]
        df_["played_minutes"] = df_["player_off"].apply(timestamp_to_minutes) - df_[
            "player_on"
        ].apply(timestamp_to_minutes)

        players_minutes_data = df_[
            [
                F.GAME_ID,
                F.TEAM_ID,
                F.TEAM_NAME,
                F.PLAYER_ID,
                "player_on",
                "player_off",
                "played_minutes",
            ]
        ].drop_duplicates()

        return players_minutes_data.dropna(subset=[F.PLAYER_ID])

    def get_players_data(self, match_events):
        external_players_df = self.load_external_players_data()
        players_jersey_df = self.get_players_jersey_data(match_events)
        players_minutes_df = self.get_players_minutes_data(match_events)

        match_players_df = players_jersey_df.merge(
            players_minutes_df,
            on=[F.GAME_ID, F.TEAM_ID, F.PLAYER_ID],
            how="inner",
        )
        match_players_df = match_players_df.merge(
            external_players_df,
            on=[F.MATCH_NAME, F.JERSEY_NUMBER, F.TEAM_NAME],
            how="left",
        )

        return match_players_df
