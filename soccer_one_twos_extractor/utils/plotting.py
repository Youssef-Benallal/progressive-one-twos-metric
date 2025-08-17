import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from mplsoccer import Pitch, VerticalPitch
from PIL import Image

from soccer_one_twos_extractor.constants.fields import F

from .utils import first_position, get_pair_positions


def plot_team_one_twos(one_twos_df: pd.DataFrame, team_name: str) -> None:
    """
    Plot all progressive one–twos made by a given team, match by match, on a
    105×68 pitch.

    For every match in `one_twos_df` where `team_name` appears (home or away),
    the function:
      • selects only the one–twos executed by `team_name` in that match,
      • draws the opening pass A→B (solid cyan), the return pass B→A (solid lime),
      • draws the receiver carry (A pass end → B pass start, dashed lime),
      • draws the overall exchange progression (A start → B pass end, dashed cyan),
      • annotates the minute (from `time_seconds_A`),
      • titles the figure with the full match scoreline.
    Matches with no rows for the team (after any upstream filtering)
    are skipped gracefully.

    Parameters
    ----------
    one_twos_df : pd.DataFrame
        Detected one–twos (typically your `progressive_one_twos_df_`), expected to
        contain at least:
          - Context: F.MATCH_NAME, F.TEAM_ID, F.HOME_TEAM_ID/NAME,
          F.AWAY_TEAM_ID/NAME, F.HOME_SCORE, F.AWAY_SCORE, "team_name"
          - Opening pass (A→B): f"{F.X}_A", f"{F.Y}_A", f"pass_end_{F.X}_A",
            f"pass_end_{F.Y}_A", f"{F.TIME}_A"
          - Return pass (B→A): f"{F.X}_B", f"{F.Y}_B", f"pass_end_{F.X}_B",
            f"pass_end_{F.Y}_B"
        Coordinates should already be normalized to a 105×68 pitch.
    team_name : str
        Team to visualize (exact string match against home/away team names).

    Returns
    -------
    None
        Displays one matplotlib figure per match. Requires `mplsoccer.Pitch`.

    Notes
    -----
    - This function assumes your global constant names come from `F` (fields enum)
      and that `Pitch` from `mplsoccer` is available in scope.
    - If you pre-filter `one_twos_df` (e.g., only key-pass cases), some matches may
      have zero one–twos for the team; those matches are skipped with a console note.
    """
    # matches where the team appears
    team_df = one_twos_df[
        (one_twos_df[F.HOME_TEAM_NAME] == team_name)
        | (one_twos_df[F.AWAY_TEAM_NAME] == team_name)
    ].copy()
    if team_df.empty:
        print(f"No one–twos found for team: {team_name}")
        return

    matches = team_df[F.MATCH_NAME].unique()

    for match in matches:
        match_df = team_df[team_df[F.MATCH_NAME] == match]

        # team_id for this match (home vs away)
        team_is_home = match_df[F.HOME_TEAM_NAME].iat[0] == team_name
        team_id = (
            match_df[F.HOME_TEAM_ID].iat[0]
            if team_is_home
            else match_df[F.AWAY_TEAM_ID].iat[0]
        )

        # only this team’s one–twos in this match
        match_team_df = match_df[match_df[F.TEAM_ID] == team_id]

        # --- guard: nothing to plot for this team in this match ---
        if match_team_df.empty:
            print(f"Skipping {match}: no {team_name} one–twos in current subset.")
            continue

        # draw pitch
        pitch = Pitch(
            pitch_type="custom",
            pitch_width=68,
            pitch_length=105,
            pitch_color="#17212d",
            line_color="white",
        )
        fig, ax = pitch.draw(figsize=(8, 8))  # type: ignore

        # plot arrows...
        for i, row in match_team_df.iterrows():
            pitch.arrows(
                row[f"{F.X}_A"],
                row[f"{F.Y}_A"],
                row[f"pass_end_{F.X}_A"],
                row[f"pass_end_{F.Y}_A"],
                width=1.5,
                headwidth=5,
                color="cyan",
                ax=ax,
                label="First pass" if i == match_team_df.index[0] else "",
            )
            pitch.arrows(
                row[f"{F.X}_B"],
                row[f"{F.Y}_B"],
                row[f"pass_end_{F.X}_B"],
                row[f"pass_end_{F.Y}_B"],
                width=1.5,
                headwidth=5,
                color="lime",
                ax=ax,
                label="Return pass" if i == match_team_df.index[0] else "",
            )
            ax.plot(  # type: ignore
                [row[f"pass_end_{F.X}_A"], row[f"{F.X}_B"]],
                [row[f"pass_end_{F.Y}_A"], row[f"{F.Y}_B"]],
                ls="--",
                color="lime",
                lw=1,
                alpha=0.7,
            )
            ax.plot(  # type: ignore
                [row[f"{F.X}_A"], row[f"pass_end_{F.X}_B"]],
                [row[f"{F.Y}_A"], row[f"pass_end_{F.Y}_B"]],
                ls="--",
                color="cyan",
                lw=1,
                alpha=0.7,
            )
            minute = int(row[f"{F.TIME}_A"] // 60)
            ax.text(  # type: ignore
                row[f"{F.X}_A"],
                row[f"{F.Y}_A"] + 1,
                f"{minute}′",
                fontsize=8,
                color="white",
                ha="center",
                va="bottom",
            )

        # legend (dedupe)
        handles, labels = ax.get_legend_handles_labels()  # type: ignore
        by_label = dict(zip(labels, handles))
        if by_label:
            ax.legend(  # type: ignore
                by_label.values(), by_label.keys(), loc="upper left", fontsize=10
            )  # type: ignore

        # title from full match slice
        home_team = match_df[F.HOME_TEAM_NAME].iat[0]
        away_team = match_df[F.AWAY_TEAM_NAME].iat[0]
        home_score = match_df[F.HOME_SCORE].iat[0]
        away_score = match_df[F.AWAY_SCORE].iat[0]
        ax.set_title(  # type: ignore
            f"{team_name} One–Twos: {home_team} {home_score}–{away_score} {away_team}",
            fontsize=14,
        )
        plt.tight_layout()
        plt.show()


def plot_season_avg_one_twos(
    one_twos_df: pd.DataFrame, logo_folder: Optional[str] = None
) -> None:
    """
    Plot per-team season averages of progressive one–twos
    (with per-team std as error bars).

    For each team, counts one–twos per match, reindexes so every team has
    exactly 38 match-rows (missing matches counted as zero), and then plots
    the average per game with the per-match standard deviation.

    Parameters
    ----------
    one_twos_df : pd.DataFrame
        Event-level one–twos DataFrame. Must contain at least:
        - F.GAME_ID (match identifier)
        - F.TEAM_NAME (team name for the event)
    logo_folder : Optional[str], default None
        Folder path that contains team logo PNGs named exactly as team names,
        e.g. "<logo_folder>/<team_name>.png". If provided and files exist,
        logos are drawn to the left of each bar.

    Returns
    -------
    None
        Displays a Matplotlib figure.

    Notes
    -----
    - Assumes a 38-match season when computing the average (total / 38).
      Change this constant if your competition has a different number of
      matches per team.
    - Expects coordinates and team/event fields to be already normalized
      by your pipeline; this function only aggregates and plots.

    Examples
    --------
    >>> plot_season_avg_one_twos(progressive_one_twos_df_, logo_folder="/path/to/logos")
    """

    df = one_twos_df.copy()

    # Count one–twos per (match, team)
    match_counts = (
        df.groupby([F.GAME_ID, F.TEAM_NAME]).size().reset_index(name="one_twos")
    )

    # Ensure every team has one row per game (fill missing with 0)
    teams = match_counts[F.TEAM_NAME].unique()
    games = match_counts[F.GAME_ID].unique()
    all_index = pd.MultiIndex.from_product(
        [games, teams],  # type: ignore
        names=[F.GAME_ID, F.TEAM_NAME],  # type: ignore
    )
    match_counts = (
        match_counts.set_index([F.GAME_ID, F.TEAM_NAME])
        .reindex(all_index, fill_value=0)
        .reset_index()
    )

    # Totals, std, and average per (assumed) 38 games
    stats = (
        match_counts.groupby(F.TEAM_NAME)["one_twos"]
        .agg(total="sum", std="std")
        .assign(avg_one_twos=lambda x: x["total"] / 38)
        .sort_values("avg_one_twos")
        .reset_index()
        .rename(columns={"std": "std_one_twos"})
    )

    # Plot (horizontal bar chart)
    y = range(len(stats))
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(
        y,
        stats["avg_one_twos"],
        xerr=stats["std_one_twos"],
        height=0.65,
    )
    ax.set_yticks([])
    ax.set_xlabel("Average Progressive One–Twos per Game")
    ax.spines["left"].set_visible(False)

    # Space on the left for logos + team names
    x_logo, x_text = -1.7, -1.3
    ax.set_xlim(left=-1.7)

    for i, team in enumerate(stats[F.TEAM_NAME]):
        y_coord = i
        if logo_folder:
            logo_path = os.path.join(logo_folder, f"{team}.png")
            if os.path.exists(logo_path):
                try:
                    img = Image.open(logo_path).resize((28, 28))
                    ab = AnnotationBbox(
                        OffsetImage(img),  # type: ignore
                        (x_logo, y_coord),
                        frameon=False,
                        box_alignment=(1, 0.5),
                    )
                    ax.add_artist(ab)
                except Exception:
                    # If a logo fails to load, just fall back to text label
                    pass
        ax.text(
            x_text, y_coord, team, va="center", ha="left", fontsize=10, color="#222"
        )

    # Drop negative tick labels (only show non-negative x)
    ax.set_xticks([t for t in ax.get_xticks() if t >= 0])

    plt.tight_layout()
    plt.show()


def plot_one_twos_heatmap(df: pd.DataFrame, teams_per_row: int = 4) -> None:
    """
    Plot per-team heatmaps (Opening vs Closing) for progressive one–twos.

    For each team, draws two vertical half-pitch KDE heatmaps side-by-side:
      - "Opening": density of the initiating pass start (x_A, y_A)
      - "Closing": density of the return pass end (pass_end_x_B, pass_end_y_B)

    Teams are laid out on a grid with `teams_per_row` columns.

    Parameters
    ----------
    df : pd.DataFrame
        One–twos dataframe containing at least:
        - F.TEAM_NAME
        - f"{F.X}_A", f"{F.Y}_A"                              (opening pass start)
        - f"pass_end_{F.X}_B", f"pass_end_{F.Y}_B"            (closing pass end)
        Coordinates are expected on a 105×68 pitch.
    teams_per_row : int, default 4
        Number of team panels per row in the figure.

    Returns
    -------
    None
        Displays a Matplotlib figure.

    Notes
    -----
    - Requires `mplsoccer.VerticalPitch` (install `mplsoccer`).
    - KDE rendering is handled by mplsoccer's built-in `kdeplot`.
    """
    teams = df[F.TEAM_NAME].dropna().unique()
    n_teams = len(teams)
    n_rows = int(np.ceil(n_teams / teams_per_row))

    # Figure and outer grid
    fig = plt.figure(figsize=(3.5 * teams_per_row * 2, 6 * n_rows))
    outer = fig.add_gridspec(
        n_rows,
        teams_per_row,
        wspace=0.2,  # space BETWEEN different teams
        hspace=0.05,  # space BETWEEN rows of teams
    )

    pitch = VerticalPitch(
        pitch_type="custom",
        pitch_width=68,
        pitch_length=105,
        pitch_color="white",
        line_color="black",
        line_zorder=3,
        shade_middle=True,
        shade_alpha=0.1,  # type: ignore
    )

    for idx, team in enumerate(teams):
        row = idx // teams_per_row
        col = idx % teams_per_row

        # Inner grid for this team: [Opening | Closing]
        inner = outer[row, col].subgridspec(1, 2, wspace=0.02, width_ratios=[1, 1])

        # Filter & coerce numeric
        team_df = df[df[F.TEAM_NAME] == team].copy()
        for c in [f"{F.X}_A", f"{F.Y}_A", f"pass_end_{F.X}_B", f"pass_end_{F.Y}_B"]:
            team_df[c] = pd.to_numeric(team_df[c], errors="coerce")
        team_df = team_df.dropna(
            subset=[f"{F.X}_A", f"{F.Y}_A", f"pass_end_{F.X}_B", f"pass_end_{F.Y}_B"]
        )

        # Opening heatmap
        ax_start = fig.add_subplot(inner[0, 0])
        pitch.draw(ax=ax_start)
        if not team_df.empty:
            pitch.kdeplot(
                x=team_df[f"{F.X}_A"],
                y=team_df[f"{F.Y}_A"],
                ax=ax_start,
                cmap="Blues",
                fill=True,
                thresh=0,
                levels=1000,
                zorder=1,
            )
        ax_start.set_title("Opening", fontsize=9)

        # Closing heatmap
        ax_end = fig.add_subplot(inner[0, 1])
        pitch.draw(ax=ax_end)
        if not team_df.empty:
            pitch.kdeplot(
                x=team_df[f"pass_end_{F.X}_B"],
                y=team_df[f"pass_end_{F.Y}_B"],
                ax=ax_end,
                cmap="Greens",
                fill=True,
                thresh=0,
                levels=1000,
                zorder=1,
            )
        ax_end.set_title("Closing", fontsize=9)

        # Team name centered above the pair
        left = ax_start.get_position().x0
        right = ax_end.get_position().x1
        top = ax_start.get_position().y1
        fig.text(
            (left + right) / 2,
            top + 0.01,
            str(team),
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.show()


def plot_one_twos_per_90(
    common_minutes_df: pd.DataFrame,
    one_twos_df: pd.DataFrame,
    min_minutes: int = 360,
    top_n: int = 20,
    logo_folder: Optional[str] = None,
) -> None:
    """
    Plot the top player duos by **one–twos per 90** (stacked A→B and B→A), sized by
    their **common minutes** on the pitch.

    What it shows
    -------------
    - One horizontal stacked bar per duo (team logo + names on the left).
    - Left segment = A→B one–twos per 90; right segment = B→A one–twos per 90.
    - Bar order = highest total per 90 (A→B + B→A).
    - Marker text at bar end = total one–twos (raw count).
    - Bar color intensity encodes common minutes (darker = more minutes).

    Parameters
    ----------
    common_minutes_df : pd.DataFrame
        Must contain, at minimum:
          - "player_1", "player_2" : player display names (same naming as `one_twos_df`)
          - "team_name"            : team of the pair
          - "common_minutes"       : minutes both players were on the pitch together
    one_twos_df : pd.DataFrame
        Event-level one–twos with columns:
          - "player_A_name", "player_B_name", "team_name"
        (Used to count A→B and B→A one–twos.)
    min_minutes : int, default 360
        Minimum **common minutes** for a duo to be included.
    top_n : int, default 20
        Plot only the top N duos by total one–twos per 90.
    logo_folder : Optional[str], default None
        If provided, will look for "<logo_folder>/<team_name>.png"
        to draw left of labels.

    Returns
    -------
    None
        Displays a Matplotlib figure.

    Notes
    -----
    - Relies on helper functions `get_pair_positions` and `first_position` to label
      players with their primary positions.
    - Assumes names in `common_minutes_df` and `one_twos_df` match exactly.

    Examples
    --------
    >>> plot_one_twos_per_90(common_minutes_df, progressive_one_twos_df_,
    ...                      min_minutes=360, top_n=20, logo_folder="logos/")
    """
    # Eligible duos = those that actually formed a one–two (undirected)
    pairs = {
        tuple(sorted([a, b]))
        for a, b in zip(one_twos_df["player_A_name"], one_twos_df["player_B_name"])
    }

    # Filter duos by minutes and membership in the one–two pairs set
    df = common_minutes_df[
        (common_minutes_df["common_minutes"] >= min_minutes)
        & (
            common_minutes_df.apply(
                lambda r: (r["player_1"], r["player_2"]) in pairs, axis=1
            )
        )
    ].copy()

    # Count A→B and B→A one–twos
    a2b = (
        one_twos_df.groupby(["player_A_name", "player_B_name", "team_name"])
        .size()
        .reset_index(name="A2B")
    )
    b2a = (
        one_twos_df.groupby(["player_B_name", "player_A_name", "team_name"])
        .size()
        .reset_index(name="B2A")
        .rename(
            columns={"player_B_name": "player_A_name", "player_A_name": "player_B_name"}
        )
    )

    # Merge counts into the filtered pairs
    m = (
        df.merge(
            a2b,
            left_on=["player_1", "player_2", "team_name"],
            right_on=["player_A_name", "player_B_name", "team_name"],
            how="left",
        )
        .merge(
            b2a,
            left_on=["player_1", "player_2", "team_name"],
            right_on=["player_A_name", "player_B_name", "team_name"],
            how="left",
        )
        .fillna(0)
    )

    # Per-90 rates
    m["A2B_p90"] = m["A2B"] / m["common_minutes"] * 90.0
    m["B2A_p90"] = m["B2A"] / m["common_minutes"] * 90.0
    m["total_p90"] = m["A2B_p90"] + m["B2A_p90"]
    m = m.sort_values("total_p90", ascending=False).head(top_n)

    # Attach primary positions for labeling
    pos_all = get_pair_positions(one_twos_df)
    m = m.merge(
        pos_all,
        left_on=["player_1", "player_2", "team_name"],
        right_on=["player_A_name", "player_B_name", "team_name"],
        how="left",
    )

    # Color intensity by common minutes
    mins = m["common_minutes"]
    norm = (mins - mins.min()) / (mins.max() - mins.min() + 1e-9)
    blues = [plt.cm.Blues(0.5 + 0.5 * n) for n in norm][::-1]  # type: ignore
    greens = [plt.cm.Greens(0.5 + 0.5 * n) for n in norm][::-1]  # type: ignore

    # Plot from top (best) at the bottom of the chart
    m = m.iloc[::-1].reset_index(drop=True)

    y = np.arange(len(m))
    fig, ax = plt.subplots(figsize=(15, max(7, len(m) * 0.43)))
    ax.barh(y, m["A2B_p90"], color=blues, label="player_1 opens")
    ax.barh(y, m["B2A_p90"], color=greens, left=m["A2B_p90"], label="player_2 opens")
    ax.set_yticks([])
    ax.set_xlabel("Progressive One–twos per 90 minutes")
    # a bit more space between title and subtitle
    ax.set_title(
        f"Progressive One–twos Duos per 90 mins (Min. {min_minutes} common mins)",
        pad=42,  # ↑ move title up from the axes
    )

    sub_y = 1.015  # ↓ lower than 1.03 → more gap to the title
    ax.text(
        0.49,
        sub_y,
        "Left Player Opens",
        color="tab:blue",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        weight="bold",
    )
    ax.text(
        0.51,
        sub_y,
        "Right Player Opens",
        color="tab:green",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        weight="bold",
    )

    # keep headroom so nothing gets clipped
    fig.tight_layout(rect=[0, 0, 1, 0.95])  # type: ignore

    # Left-side labels: team logo + "player_1 (pos) - player_2 (pos)"
    for i, (team, a, b, pos_a, pos_b, _cmin) in enumerate(
        zip(
            m["team_name"],
            m["player_1"],
            m["player_2"],
            m.get("player_A_position", pd.Series(index=m.index, dtype=object)),
            m.get("player_B_position", pd.Series(index=m.index, dtype=object)),
            m["common_minutes"],
        )
    ):
        pa = first_position(pos_a) if pd.notna(pos_a) else ""
        pb = first_position(pos_b) if pd.notna(pos_b) else ""
        a_name = f"{a}{f' ({pa})' if pa else ''}"
        b_name = f"{b}{f' ({pb})' if pb else ''}"
        ax.text(
            -0.8,
            i,
            f"{a_name} - {b_name}",
            va="center",
            ha="left",
            fontsize=9,
            color="black",
            clip_on=False,
        )

        if logo_folder:
            logo_path = os.path.join(logo_folder, f"{team}.png")
            if os.path.exists(logo_path):
                try:
                    img = Image.open(logo_path).resize((22, 22))
                    ab = AnnotationBbox(
                        OffsetImage(img),  # type: ignore
                        (-0.9, i),
                        frameon=False,
                        box_alignment=(1, 0.5),
                    )
                    ax.add_artist(ab)
                except Exception:
                    pass

    # Total one–twos (raw count) to the right of each bar
    for i, total in enumerate((m["A2B"] + m["B2A"]).astype(int)):
        ax.text(
            m.loc[i, "total_p90"] + 0.2,
            i,
            f"{total} one-twos",
            va="center",
            ha="left",
            fontsize=8,
            color="gray",
        )

    ax.set_xlim(left=-1.3)
    ax.set_xticks([tick for tick in ax.get_xticks() if tick >= 0])

    plt.tight_layout(rect=[0, 0, 1, 0.88])  # type: ignore
    plt.show()


def pos_bucket(pos_str: Optional[str]) -> str:
    """
    Map a raw position string to one of five buckets:
    'Fullback', 'Winger', 'Central midfield', 'Forward', or 'Other'.

    Rules
    -----
    - Fullback: RB, LB, RWB, LWB, WB
    - Winger:   LW, RW, WF
    - Central midfield: DM, CDM, CM, RCM, LCM, AM, CAM
    - Forward:  ST, CF, FW
    - Other:    everything else (incl. GK, CB, etc.)

    Parameters
    ----------
    pos_str : Optional[str]
        Position string (e.g., "CM,DM" or "RB"). Comma-separated tokens allowed.

    Returns
    -------
    str
        One of {"Fullback", "Winger", "Central midfield", "Forward", "Other"}.

    Examples
    --------
    >>> pos_bucket("RB")
    'Fullback'
    >>> pos_bucket("CM,DM")
    'Central midfield'
    >>> pos_bucket("ST")
    'Forward'
    """
    if pd.isna(pos_str):
        return "Other"
    toks = [
        t.strip().upper().replace("-", "") for t in str(pos_str).split(",") if t.strip()
    ]
    for t in toks:
        if t in {"RB", "LB", "RWB", "LWB", "WB"}:
            return "Fullback"
        if t in {"LW", "RW", "WF"}:
            return "Winger"
        if t in {"DM", "CDM", "CM", "RCM", "LCM", "AM", "CAM"}:
            return "Central midfield"
        if t in {"ST", "CF", "FW"}:
            return "Forward"
    return "Other"


def plot_onetwos_scatter_player(
    prog_df: pd.DataFrame,
    players_df: pd.DataFrame,
    min_minutes: int = 900,
    label_thresh: float = 0.6,
) -> pd.DataFrame:
    """
    Scatter of one–twos opened P90 (x) vs closed P90 (y), one dot per player.

    - Filters players with total minutes < `min_minutes`.
    - Sizes points by minutes played (across games).
    - Colors by 5-role bucket from `pos_bucket`: Fullback / Winger /
      Central midfield / Forward / Other.
    - Draws median lines (x and y).
    - Labels only players with opened_p90 >= `label_thresh` OR
      closed_p90 >= `label_thresh` (last-name tag).

    Required columns
    ----------------
    prog_df:
      - "player_id_A", "player_id_B"  (to count opened/closed one–twos)
    players_df:
      - "player_id", "played_minutes", "player", "pos"

    Parameters
    ----------
    prog_df : pd.DataFrame
        Progressive one–twos dataframe (event-level).
    players_df : pd.DataFrame
        Player participation dataframe (per match), including minutes and position.
    min_minutes : int, default 900
        Minimum total minutes for a player to be included.
    label_thresh : float, default 0.6
        Threshold for annotating points (opened_p90 or closed_p90).

    Returns
    -------
    pd.DataFrame
        Table used for plotting:
        ["player_id","player","bucket","played_minutes",
         "opened","closed","opened_p90","closed_p90"].
    """
    # --- aggregate minutes + canonical name + bucket per player ---
    mins = (
        players_df.assign(bucket=players_df["pos"].map(pos_bucket))
        .groupby("player_id", as_index=False)
        .agg(
            played_minutes=("played_minutes", "sum"),
            player=(
                "player",
                lambda s: s.dropna().mode().iat[0] if s.dropna().size else "Unknown",
            ),
            bucket=(
                "bucket",
                lambda s: s.dropna().mode().iat[0] if s.dropna().size else "Other",
            ),
        )
    )

    # --- counts of one–twos opened/closed per player ---
    opened = (
        prog_df.groupby("player_id_A")
        .size()
        .rename("opened")
        .reset_index()
        .rename(columns={"player_id_A": "player_id"})
    )
    closed = (
        prog_df.groupby("player_id_B")
        .size()
        .rename("closed")
        .reset_index()
        .rename(columns={"player_id_B": "player_id"})
    )

    df = (
        mins.merge(opened, on="player_id", how="left")
        .merge(closed, on="player_id", how="left")
        .fillna({"opened": 0, "closed": 0})
    )

    # --- filter and compute per90 ---
    df = df[df["played_minutes"] >= min_minutes].copy().reset_index(drop=True)
    df["opened_p90"] = 90.0 * df["opened"] / df["played_minutes"]
    df["closed_p90"] = 90.0 * df["closed"] / df["played_minutes"]

    # --- sizes aligned to index (Series to avoid indexing errors) ---
    m = df["played_minutes"]
    sizes = 40 + 260 * (m - m.min()) / (m.max() - m.min() + 1e-9)

    # --- palette for the 5 buckets ---
    palette = {
        "Fullback": "#1f77b4",
        "Winger": "#e377c2",
        "Central midfield": "#ff7f0e",
        "Forward": "#d62728",
        "Other": "#7f7f7f",
    }

    # --- plot ---
    fig, ax = plt.subplots(figsize=(12, 9))
    for bucket, grp in df.groupby("bucket"):
        ax.scatter(
            grp["opened_p90"],
            grp["closed_p90"],
            s=sizes.loc[grp.index],
            c=palette.get(bucket, "#7f7f7f"),  # type: ignore
            alpha=0.85,
            edgecolors="none",
            label=bucket,
        )

    # median lines
    mx, my = df["opened_p90"].median(), df["closed_p90"].median()
    ax.axvline(mx, ls="--", lw=1)
    ax.axhline(my, ls="--", lw=1)

    # label only standouts (last-name tag)
    standout = df[
        (df["opened_p90"] >= label_thresh) | (df["closed_p90"] >= label_thresh)
    ]
    if not standout.empty:
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        dx, dy = 0.01 * (x1 - x0), 0.01 * (y1 - y0)
        last_names = standout["player"].astype(str).str.split().str[-1]
        for x, y, name in zip(
            standout["opened_p90"], standout["closed_p90"], last_names
        ):
            ax.text(x + dx, y + dy, name, fontsize=9)

    ax.legend(title="Role", frameon=False, loc="upper left", ncol=1)
    ax.set_xlabel("One–twos Opened per 90 min")
    ax.set_ylabel("One–twos Closed per 90 min")
    ax.grid(True, ls="--", lw=0.5, alpha=0.5)
    ax.set_title(
        "Progressive One–twos: Opened (Initial pass) vs Closed (Return pass)"
        f"(Minimum {min_minutes} mins played)"
    )
    plt.tight_layout()
    plt.show()

    return df[
        [
            "player_id",
            "player",
            "bucket",
            "played_minutes",
            "opened",
            "closed",
            "opened_p90",
            "closed_p90",
        ]
    ]


def plot_onetwos_chance_creation(
    df: pd.DataFrame,
    x_thresh: float = 70.0,
    logo_folder: Optional[str] = None,
) -> pd.DataFrame:
    """
    Plot per-team **final-third one–twos chance creation** as a single stacked bar:
      [  % B is key pass  |  % next event is key pass  |  grey remainder to 100% ].

    What it does
    ------------
    1) Filters one–twos with `x_A >= x_thresh` (final third).
    2) Per team, computes:
         - `pct_kpB`   = share where the closing pass B is a key pass
         - `pct_kpNext`= share where the event immediately after B is a key pass
    3) Draws one stacked horizontal bar per team (two blue segments + grey to 100%),
       sorted by (`pct_kpB` + `pct_kpNext`) ascending.
    4) Optionally places team logos and names on the left margin.

    Parameters
    ----------
    df : pd.DataFrame
        One–twos dataframe with at least:
        - "team_name" or "team_id" (team identifier)
        - "x_A" (opening pass start X, same scale as `x_thresh`)
        - "keypass_B" or "KEY_PASS_B" (flag for B being a key pass)
        - "keypass_next_B" or "KEY_PASS_next_B" (flag for next event being a key pass)
    x_thresh : float, default 70.0
        Final-third X threshold in the same coordinate system as `x_A`.
    logo_folder : Optional[str], default None
        If provided, attempts to load "<logo_folder>/<team_name>.png" for each bar.

    Returns
    -------
    pd.DataFrame
        Table used for plotting with columns:
        [team, n_ft, n_kpB, n_kpN, pct_kpB, pct_kpNext, sort_key]

    Notes
    -----
    - Key-pass flags are interpreted robustly: {1, "1", True} → True.
    """
    team_col = "team_name" if "team_name" in df.columns else "team_id"
    kp_b_col = "keypass_B" if "keypass_B" in df.columns else "KEY_PASS_B"
    kp_n_col = "keypass_next_B" if "keypass_next_B" in df.columns else "KEY_PASS_next_B"
    bar_height, label_decimals, x_max_pct = 0.60, 0, 100

    # -- compute per-team rates (fractions 0..1) --
    ft = df.loc[df["x_A"] >= x_thresh].copy()

    def _as_bool(s: pd.Series) -> pd.Series:
        """Treat 1/'1'/True (any case/whitespace) as True; else False."""
        return s.fillna(0).astype(str).str.strip().str.lower().isin({"1", "true"})

    ft["kpB"] = _as_bool(ft[kp_b_col]).astype(int)
    ft["kpN"] = _as_bool(ft[kp_n_col]).astype(int)

    stats = ft.groupby(team_col, as_index=False).agg(
        n_ft=("x_A", "size"), n_kpB=("kpB", "sum"), n_kpN=("kpN", "sum")
    )
    stats["pct_kpB"] = np.where(stats["n_ft"] > 0, stats["n_kpB"] / stats["n_ft"], 0.0)
    stats["pct_kpNext"] = np.where(
        stats["n_ft"] > 0, stats["n_kpN"] / stats["n_ft"], 0.0
    )
    stats["sort_key"] = stats["pct_kpB"] + stats["pct_kpNext"]
    stats = stats.sort_values("sort_key", ascending=True).reset_index(drop=True)

    # -- stacked widths (clip to 100%) --
    w_b = stats["pct_kpB"].to_numpy(float)
    w_n_true = stats["pct_kpNext"].to_numpy(float)
    w_n = np.minimum(w_n_true, 1.0 - w_b)
    w_g = 1.0 - (w_b + w_n)
    y = np.arange(len(stats))

    # -- plot --
    fig_h = max(4.5, 0.45 * len(stats) + 1.5)
    fig, ax = plt.subplots(figsize=(12, fig_h))
    ax.set_xlim(-0.22, x_max_pct / 100.0)  # left margin for logos/text
    margin = bar_height * 1.8
    ax.set_ylim(-margin, len(stats) - 1 + margin)
    ax.set_yticks([])

    col_b, col_n, col_g = "#1f77b4", "#6baed6", "#e0e0e0"
    ax.barh(
        y,
        w_b,
        left=0.0,
        height=bar_height,
        color=col_b,
        edgecolor="none",
        label="One two led to key pass",
    )
    ax.barh(
        y,
        w_n,
        left=w_b,
        height=bar_height,
        color=col_n,
        edgecolor="none",
        label="One two followed by key pass",
    )
    ax.barh(y, w_g, left=w_b + w_n, height=bar_height, color=col_g, edgecolor="none")

    # labels on blue segments only (0–100%)
    for i, (b, n_true) in enumerate(zip(w_b, w_n_true)):
        if b > 0.02:
            ax.text(
                min(b, ax.get_xlim()[1]) - 0.005,
                i,
                f"{b*100:.{label_decimals}f}%",
                va="center",
                ha="right",
                color="white",
                fontsize=9,
            )
        if n_true > 0.02:
            right_edge = w_b[i] + min(n_true, 1 - w_b[i])
            ax.text(
                min(right_edge, ax.get_xlim()[1]) - 0.005,
                i,
                f"{n_true*100:.{label_decimals}f}%",
                va="center",
                ha="right",
                color="white",
                fontsize=9,
            )

    # logos + names at left
    x_logo, x_text = -0.18, -0.16
    for i, name in enumerate(stats[team_col].astype(str)):
        if logo_folder:
            path = os.path.join(logo_folder, f"{name}.png")
            if os.path.exists(path):
                try:
                    img = Image.open(path)
                    ab = AnnotationBbox(
                        OffsetImage(img, zoom=0.08),  # type: ignore
                        (x_logo, i),
                        frameon=False,
                        box_alignment=(1, 0.5),
                    )
                    ax.add_artist(ab)
                except Exception:
                    pass
        ax.text(x_text, i, name, va="center", ha="left", fontsize=10, color="#222")

    # x-axis 0–100%
    xticks = np.linspace(0, x_max_pct, 5)
    ax.set_xticks(xticks / 100.0)
    ax.set_xticklabels([f"{int(t)}%" for t in xticks])

    for s in ("left", "right", "top"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="lower right", frameon=False)
    ax.set_title("Final-third one–twos chance creation rate", pad=12)
    plt.tight_layout()
    plt.show()

    return stats[
        [team_col, "n_ft", "n_kpB", "n_kpN", "pct_kpB", "pct_kpNext", "sort_key"]
    ]
