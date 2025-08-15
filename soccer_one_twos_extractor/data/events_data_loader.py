import os
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from soccer_one_twos_extractor.constants.qualifiers_mapping import EVENT_TYPE_MAPPING_DF
from soccer_one_twos_extractor.constants.fields import F


class EventsDataLoader:
    def __init__(self, data_folder):
        self.data_folder = data_folder

    def load_f24_events(self, xml_filename):
        tree = ET.parse(os.path.join(self.data_folder, xml_filename))
        root = tree.getroot()
        game_info = root.find("Game")
        if game_info is None:
            return pd.DataFrame()
        meta = {
            k: game_info.get(k)
            for k in [
                F.ID,
                F.HOME_SCORE,
                F.AWAY_SCORE,
                F.HOME_TEAM_ID,
                F.HOME_TEAM_NAME,
                F.AWAY_TEAM_ID,
                F.AWAY_TEAM_NAME,
                F.COMPETITION_ID,
                F.COMPETITION_NAME,
                F.SEASON_ID,
            ]
        }
        events = []
        for game in root:
            for event in game:
                evt = pd.json_normalize(event.attrib)
                quals = [q.attrib for q in event]
                evt[F.QUALIFIERS] = [quals]
                events.append(evt)
        if not events:
            return pd.DataFrame()
        df = pd.concat(events, ignore_index=True)
        if F.TYPE_ID in df.columns:
            df = df.merge(EVENT_TYPE_MAPPING_DF, on=F.TYPE_ID, how="left")
        for key, val in meta.items():
            df[key] = val
        # Data types
        for col in [F.ID, F.EVENT_ID, F.TYPE_ID, F.PERIOD_ID, F.MINUTE, F.SECOND]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
        for col in [F.X, F.Y]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.rename(columns={F.ID: F.GAME_ID})

        # Additional columns
        df[F.MATCH_NAME] = (
            f"{df[F.HOME_TEAM_NAME].iloc[0]}-{df[F.AWAY_TEAM_NAME].iloc[0]}"
        )
        df[F.TEAM_NAME] = np.where(
            df[F.TEAM_ID] == df[F.HOME_TEAM_ID],
            df[F.HOME_TEAM_NAME],
            df[F.AWAY_TEAM_NAME],
        )
        return df
