"""
src/data/collect.py
-------------------
Chargement et nettoyage des données pour le World Cup 2026 Predictor.

Sources :
  - data/raw/results.csv      : 49 287 matchs internationaux (1872 → WC 2026)
  - data/raw/goalscorers.csv  : 47 601 buteurs historiques
  - data/raw/shootouts.csv    : résultats des tirs au but
  - data/raw/former_names.csv : anciens noms de pays
"""

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger

RAW_DIR = Path("data/raw")


# ============================================================
# NOMS NORMALISÉS — harmonisation des noms de pays
# ============================================================

# Certains pays ont changé de nom au fil du temps.
# On normalise tout vers le nom actuel pour éviter les doublons.
COUNTRY_ALIASES = {
    "West Germany":          "Germany",
    "East Germany":          "Germany",
    "Soviet Union":          "Russia",
    "Yugoslavia":            "Serbia",
    "Czechoslovakia":        "Czech Republic",
    "Zaire":                 "DR Congo",
    "Republic of Ireland":   "Ireland",
    "China PR":              "China",
    "Korea Republic":        "South Korea",
    "Korea DPR":             "North Korea",
    "USA":                   "United States",
    "Türkiye":               "Turkey",
}

def normalize_team_name(name: str) -> str:
    """Normalise le nom d'une équipe vers son nom actuel."""
    return COUNTRY_ALIASES.get(name, name)


# ============================================================
# CHARGEMENT DES DONNÉES
# ============================================================

def load_results(min_year: int = 1990) -> pd.DataFrame:
    """
    Charge l'historique des matchs internationaux.

    On filtre à partir de 1990 par défaut car les équipes
    nationales évoluent beaucoup sur le long terme — les
    résultats des années 60 sont peu prédictifs pour 2026.

    Args:
        min_year : année minimum à inclure (défaut 1990)

    Returns:
        DataFrame avec tous les matchs joués (scores non nuls)
    """
    logger.info("Chargement des résultats historiques...")
    df = pd.read_csv(RAW_DIR / "results.csv", parse_dates=["date"])

    # Normalisation des noms
    df["home_team"] = df["home_team"].apply(normalize_team_name)
    df["away_team"] = df["away_team"].apply(normalize_team_name)

    # Filtre : matchs joués uniquement (score connu)
    df_played = df[df["home_score"].notna()].copy()

    # Filtre temporel
    df_played = df_played[df_played["date"].dt.year >= min_year]

    logger.success(f"✅ {len(df_played)} matchs chargés ({min_year} → aujourd'hui)")
    return df_played


def load_wc2026_fixtures() -> pd.DataFrame:
    """
    Charge le calendrier des matchs WC 2026 (scores vides).

    Ces matchs sont déjà dans results.csv avec home_score = NaN.
    On les extrait directement.

    Returns:
        DataFrame avec tous les matchs WC 2026 à prédire.
    """
    logger.info("Chargement du calendrier WC 2026...")
    df = pd.read_csv(RAW_DIR / "results.csv", parse_dates=["date"])

    df["home_team"] = df["home_team"].apply(normalize_team_name)
    df["away_team"] = df["away_team"].apply(normalize_team_name)

    # Matchs WC 2026 = FIFA World Cup 2026 avec score vide
    wc2026 = df[
        (df["tournament"] == "FIFA World Cup") &
        (df["date"].dt.year == 2026) &
        (df["home_score"].isna())
    ].copy().reset_index(drop=True)

    logger.success(f"✅ {len(wc2026)} matchs WC 2026 trouvés")
    return wc2026


def load_goalscorers(min_year: int = 1990) -> pd.DataFrame:
    """
    Charge l'historique des buteurs.

    Args:
        min_year : année minimum

    Returns:
        DataFrame avec tous les buts marqués.
    """
    logger.info("Chargement des buteurs historiques...")
    df = pd.read_csv(RAW_DIR / "goalscorers.csv", parse_dates=["date"])

    df["home_team"] = df["home_team"].apply(normalize_team_name)
    df["away_team"] = df["away_team"].apply(normalize_team_name)
    df["team"]      = df["team"].apply(normalize_team_name)

    df = df[df["date"].dt.year >= min_year]

    # Exclure les buts contre son camp pour les stats individuelles
    df_goals = df[df["own_goal"] == False].copy()

    logger.success(f"✅ {len(df_goals)} buts chargés ({min_year} → aujourd'hui)")
    return df_goals


def get_team_stats(results: pd.DataFrame, team: str) -> dict:
    """
    Calcule les statistiques de base d'une équipe sur ses derniers matchs.

    Args:
        results : DataFrame des résultats historiques
        team    : nom de l'équipe

    Returns:
        dict avec buts marqués, encaissés, victoires, etc.
    """
    home = results[results["home_team"] == team]
    away = results[results["away_team"] == team]

    if len(home) + len(away) == 0:
        return {}

    goals_scored   = home["home_score"].sum() + away["away_score"].sum()
    goals_conceded = home["away_score"].sum() + away["home_score"].sum()
    total_matches  = len(home) + len(away)

    wins = (
        ((home["home_score"] > home["away_score"]).sum()) +
        ((away["away_score"] > away["home_score"]).sum())
    )

    return {
        "team":            team,
        "matches":         total_matches,
        "goals_scored":    goals_scored,
        "goals_conceded":  goals_conceded,
        "avg_scored":      round(goals_scored / total_matches, 2),
        "avg_conceded":    round(goals_conceded / total_matches, 2),
        "win_rate":        round(wins / total_matches, 2),
    }


def get_top_scorers(goalscorers: pd.DataFrame, team: str, top_n: int = 5) -> pd.DataFrame:
    """
    Retourne les meilleurs buteurs d'une équipe sur la période.

    Args:
        goalscorers : DataFrame des buteurs
        team        : nom de l'équipe
        top_n       : nombre de joueurs à retourner

    Returns:
        DataFrame avec les top buteurs.
    """
    team_goals = goalscorers[goalscorers["team"] == team]

    if team_goals.empty:
        return pd.DataFrame()

    top = (
        team_goals.groupby("scorer")
        .agg(
            buts=("scorer", "count"),
            penaltys=("penalty", "sum"),
        )
        .reset_index()
        .sort_values("buts", ascending=False)
        .head(top_n)
    )
    top["buts_hors_pk"] = top["buts"] - top["penaltys"]
    return top


# ============================================================
# POINT D'ENTRÉE — test rapide
# ============================================================

if __name__ == "__main__":
    print("=" * 55)
    print("  World Cup 2026 Predictor — Chargement données")
    print("=" * 55)

    # Résultats historiques
    results = load_results(min_year=1990)
    print(f"\n📊 Matchs historiques : {len(results)}")
    print(f"   Période : {results['date'].min().year} → {results['date'].max().year}")
    print(f"   Tournois : {results['tournament'].nunique()} compétitions différentes")

    # Calendrier WC 2026
    fixtures = load_wc2026_fixtures()
    print(f"\n🏆 Matchs WC 2026 à prédire : {len(fixtures)}")
    print(f"   Premier match : {fixtures['date'].min().strftime('%d %B %Y')}")
    print(f"   Finale : {fixtures['date'].max().strftime('%d %B %Y')}")
    print(f"\n   Exemples :")
    for _, row in fixtures.head(5).iterrows():
        print(f"   {row['date'].strftime('%d/%m')} — {row['home_team']} vs {row['away_team']}")

    # Buteurs
    goalscorers = load_goalscorers(min_year=1990)
    print(f"\n⚽ Buteurs historiques : {len(goalscorers)}")

    # Stats d'une équipe exemple
    print(f"\n📈 Stats France (depuis 1990) :")
    stats = get_team_stats(results, "France")
    for k, v in stats.items():
        print(f"   {k:20} : {v}")

    # Top buteurs France
    print(f"\n🥇 Top 5 buteurs France (depuis 1990) :")
    top = get_top_scorers(goalscorers, "France")
    print(top.to_string(index=False))
