from __future__ import annotations

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
    """
    End-to-end pipeline to load Opta F24 matches, enrich events with features,
    and extract progressive one–twos + player tables.

    You can tune the one–two association detection via the `AssociationMetrics`
    parameters exposed here.

    Parameters
    ----------
    data_folder : str
        Folder containing the raw match XML files.
    external_player_file : str
        Path to an external CSV with player metadata (fbref, etc.).
    n_jobs : int, default 8
        Parallel workers for parsing multiple matches.
    min_progression_ratio_pct : float, default 0.25
        AssociationMetrics: minimum progression ratio (0..1) to accept the
        opening→closing exchange (goal-line or goal-center).
    max_time_diff : float, default 5.0
        AssociationMetrics: maximum time (seconds) between A pass and B pass.
    max_player_b_ball_carry_distance : float, default 7.0
        AssociationMetrics: maximum carry distance (meters) between A’s pass
        end and B’s next event start.
    association_metrics : Optional[AssociationMetrics], default None
        If provided, use this instance instead of constructing one from the
        parameters above.

    Notes
    -----
    - Assumes events are normalized to a 105×68 pitch by `normalize_coords`.
    - `parallel_run()` returns three concatenated DataFrames across all files:
      (events_df, players_df, progressive_one_twos_df).
    """

    def __init__(
        self,
        data_folder: str,
        external_player_file: str,
        n_jobs: int = 8,
        *,
        min_progression_ratio_pct: float = 0.25,
        max_time_diff: float = 5.0,
        max_player_b_ball_carry_distance: float = 7.0,
    ) -> None:
        self.data_folder = data_folder
        self.external_players_file = external_player_file
        self.n_jobs = n_jobs

        self.match_events_loader = EventsDataLoader(data_folder)
        self.match_players_loader = PlayerDataLoader(data_folder, external_player_file)
        self.features_extractor = FeaturesExtractor()

        # Allow injection or build from provided hyper-parameters
        self.association_metric_extractor = AssociationMetrics(
            min_progression_ratio_pct=min_progression_ratio_pct,
            max_time_diff=max_time_diff,
            max_player_b_ball_carry_distance=max_player_b_ball_carry_distance,
        )

    def process_match_file(self, file_path: str) -> dict[str, pd.DataFrame]:
        """
        Parse a single match file and compute:
          - enriched events,
          - player table,
          - progressive one–twos.

        Parameters
        ----------
        file_path : str
            Path to a single F24 XML file.

        Returns
        -------
        dict
            {
              "events": pd.DataFrame,
              "players": pd.DataFrame,
              "progressive_one_twos": pd.DataFrame
            }
        """
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

    def parallel_run(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Run the pipeline over all XML files in `data_folder` in parallel and
        concatenate results.

        Returns
        -------
        (events_df, players_df, progressive_one_twos_df) : tuple of DataFrames
        """
        xml_file_names = list_xml_filenames(self.data_folder)

        results = parallel_map(
            self.process_match_file,
            xml_file_names,
            n_jobs=self.n_jobs,
            desc="Parsing matches",
        )

        events_df = pd.concat(
            [r["events"] for r in results],
            ignore_index=True,  # type: ignore
        )
        players_df = pd.concat(
            [r["players"] for r in results],
            ignore_index=True,  # type: ignore
        )
        progressive_one_twos_df = pd.concat(
            [r["progressive_one_twos"] for r in results],  # type: ignore
            ignore_index=True,
        )

        return events_df, players_df, progressive_one_twos_df
