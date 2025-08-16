import pandas as pd

from soccer_one_twos_extractor.data.events_data_loader import EventsDataLoader
from soccer_one_twos_extractor.data.players_data_loader import PlayerDataLoader
from soccer_one_twos_extractor.features.extract_features import FeaturesExtractor
from soccer_one_twos_extractor.metrics.association_metrics import AssociationMetrics
from soccer_one_twos_extractor.utils.utils import (
    fix_event_id,
    list_xml_filenames,
    normalize_coords,
    parallel_map,
)


class MetricPipeline:
    def __init__(self, data_folder, external_player_file, n_jobs=8):
        self.data_folder = data_folder
        self.external_players_file = external_player_file

        self.match_events_loader = EventsDataLoader(data_folder)
        self.match_players_loader = PlayerDataLoader(data_folder, external_player_file)
        self.features_extractor = FeaturesExtractor()
        self.association_metric_extractor = AssociationMetrics()

        self.n_jobs = n_jobs

    def process_match_file(self, file_path):
        match_events = (
            self.match_events_loader.load_f24_events(file_path)
            .pipe(fix_event_id)
            .pipe(self.features_extractor.pass_features)
            .pipe(normalize_coords)
            .pipe(self.features_extractor.progression_features)
        )
        match_players = self.match_players_loader.get_players_data(match_events)
        match_progressive_one_twos = (
            self.association_metric_extractor.get_progressive_one_twos(match_events)
        )
        return {
            "events": match_events,
            "players": match_players,
            "progressive_one_twos": match_progressive_one_twos,
        }

    def parallel_run(self):
        xml_file_names = list_xml_filenames(self.data_folder)

        results = parallel_map(
            self.process_match_file,
            xml_file_names,
            n_jobs=self.n_jobs,
            desc="Parsing matches",
        )
        events_df = pd.concat(
            [r["events"] for r in results],  # type: ignore
            ignore_index=True,  # type: ignore
        )
        players_df = pd.concat(
            [r["players"] for r in results],  # type: ignore
            ignore_index=True,  # type: ignore
        )
        progressive_one_twos_df = pd.concat(
            [r["progressive_one_twos"] for r in results],  # type: ignore
            ignore_index=True,  # type: ignore
        )

        return (events_df, players_df, progressive_one_twos_df)
