"""
src/models/scorer_predictor.py
--------------------------------
Prédiction des buteurs par match et sur l'ensemble du tournoi.

APPROCHE :
  1. Pour chaque équipe, on calcule la part de buts de chaque joueur
     (ratio historique depuis 2020 — période la plus représentative)
  2. On prédit le nombre de buts attendus d'une équipe dans un match
     via le modèle de Poisson
  3. On distribue ces buts entre les joueurs selon leurs ratios
  4. Sur l'ensemble du tournoi, on somme les buts attendus match par match

POINT D'ATTENTION :
  Le dataset goalscorers.csv est incomplet sur certaines périodes.
  Les ratios sont calculés sur les données disponibles (2020-2026).
  Certains joueurs récents peuvent être sous-représentés.
  On corrige ça avec des ajustements manuels pour les stars connues.

JOUEURS RETIRÉS / INDISPONIBLES :
  Benzema, Di María, Giroud, Sterling, Neymar retirés ou peu probables.
  On les exclut avec EXCLUDED_PLAYERS.
"""

import pandas as pd
import numpy as np
from loguru import logger
from pathlib import Path

# ============================================================
# JOUEURS EXCLUS — retraités ou très probablement absents
# ============================================================
EXCLUDED_PLAYERS = {
    "Karim Benzema",       # retraité international
    "Angel Di María",      # retraité international
    "Ángel Di María",      # variante
    "Olivier Giroud",      # retraité international
    "Raheem Sterling",     # probable absence sélection
    "Neymar",              # blessures chroniques, incertain
    "Marcus Rashford",     # forme en berne, incertain
    "Álvaro Morata",       # retraité international Euro 2024
    "Memphis Depay",       # retraité international
    "Ivan Perišić",        # retraité international
    "Edin Džeko",          # retraité international
    "Islam Slimani",       # retraité international
    "Gareth Bale",         # retraité
    "Robert Lewandowski",  # Pologne non qualifiée WC 2026
    "Marko Arnautović",    # retraité international autrichien
    "Gareth Bale",         # retraité
    "Robert Lewandowski",  # Pologne non qualifiée
    "Romelu Lukaku",       # Belgique non qualifiée WC 2026
    "Mohamed Salah",       # Égypte non qualifiée (phase de groupes seulement)
}

# ============================================================
# AJUSTEMENTS MANUELS — stars dont les données sont incomplètes
# Ces multiplicateurs corrigent les sous-estimations du dataset
# ============================================================
PLAYER_BOOST = {
    # France
    "Kylian Mbappé":      1.10,  # capitaine, forme excellente
    "Antoine Griezmann":  1.05,
    # Espagne
    "Ferran Torres":      0.95,  # souvent remplaçant
    "Álvaro Morata":      0.90,  # retraité ? à vérifier
    "Mikel Oyarzabal":    1.05,
    "Dani Olmo":          1.10,  # en grande forme
    "Lamine Yamal":       1.20,  # nouvelle star, sous-représenté dataset
    "Nico Williams":      1.15,  # idem
    # Argentine
    "Lionel Messi":       1.00,  # dataset assez complet
    "Lautaro Martínez":   1.05,
    "Julián Álvarez":     1.10,  # en grande forme
    # Angleterre
    "Harry Kane":         1.05,
    "Bukayo Saka":        1.10,
    "Phil Foden":         1.10,  # sous-représenté
    "Cole Palmer":        1.15,  # nouvelle star
    # Portugal
    "Cristiano Ronaldo":  0.85,  # 40 ans, moins décisif
    "Bruno Fernandes":    1.05,
    "Gonçalo Ramos":      1.15,  # en grande forme
    "Rafael Leão":        1.10,
    # Brésil
    "Vinícius Júnior":    1.20,  # Ballon d'Or niveau, sous-représenté
    "Raphinha":           1.10,
    "Rodrygo":            1.10,
    # Allemagne
    "Kai Havertz":        1.10,
    "Florian Wirtz":      1.20,  # nouvelle star, peu de matchs dataset
    "Jamal Musiala":      1.15,
    # Pays-Bas
    "Memphis Depay":      0.90,
    "Cody Gakpo":         1.15,
    "Xavi Simons":        1.10,
}

# ============================================================
# CLASSE PRINCIPALE
# ============================================================

class ScorerPredictor:
    """
    Prédit les buteurs probables par match et sur le tournoi.

    Utilise les données historiques (2020-2026) pour calculer
    la part de buts de chaque joueur dans son équipe,
    puis distribue les buts attendus (modèle Poisson) entre les joueurs.
    """

    def __init__(self, min_year: int = 2020):
        """
        Args:
            min_year : année minimum pour les stats de buteurs
                       2020 = données récentes et représentatives
        """
        self.min_year      = min_year
        self.team_scorers_ = {}   # {team: DataFrame avec ratios par joueur}
        self.fitted_       = False

    def fit(self, goalscorers: pd.DataFrame) -> "ScorerPredictor":
        """
        Calcule les ratios de buts par joueur pour chaque équipe.

        Args:
            goalscorers : DataFrame retourné par load_goalscorers()

        Returns:
            self
        """
        logger.info(f"Calcul des ratios buteurs (depuis {self.min_year})...")

        # Filtre temporel et exclusion des CSC
        df = goalscorers[
            (goalscorers["date"].dt.year >= self.min_year) &
            (goalscorers["own_goal"] == False)
        ].copy()

        # Calcul par équipe
        teams = df["team"].unique()
        for team in teams:
            team_df = df[df["team"] == team]

            # Agrégation par joueur
            player_stats = (
                team_df.groupby("scorer")
                .agg(
                    goals      = ("scorer", "count"),
                    penalties  = ("penalty", "sum"),
                )
                .reset_index()
            )
            player_stats["goals_non_pk"] = player_stats["goals"] - player_stats["penalties"]

            # Exclure les joueurs retirés
            player_stats = player_stats[
                ~player_stats["scorer"].isin(EXCLUDED_PLAYERS)
            ]

            if player_stats.empty:
                continue

            # Appliquer les boosts manuels
            player_stats["boost"] = player_stats["scorer"].map(PLAYER_BOOST).fillna(1.0)
            player_stats["goals_adj"] = player_stats["goals"] * player_stats["boost"]

            # Calcul du ratio (part des buts de l'équipe)
            total_adj = player_stats["goals_adj"].sum()
            if total_adj > 0:
                player_stats["ratio"] = player_stats["goals_adj"] / total_adj
            else:
                player_stats["ratio"] = 1.0 / len(player_stats)

            # Tri par ratio décroissant
            player_stats = player_stats.sort_values("goals_adj", ascending=False)

            self.team_scorers_[team] = player_stats

        self.fitted_ = True
        logger.success(f"✅ Ratios calculés pour {len(self.team_scorers_)} équipes")
        return self

    def predict_match_scorers(
        self,
        home: str,
        away: str,
        expected_home: float,
        expected_away: float,
        top_n: int = 5,
    ) -> dict:
        """
        Prédit les buteurs probables pour un match.

        Pour chaque équipe :
          buts_attendus_joueur = buts_attendus_équipe × ratio_joueur

        Args:
            home          : équipe domicile
            away          : équipe extérieur
            expected_home : lambda Poisson pour l'équipe domicile (du modèle)
            expected_away : lambda Poisson pour l'équipe extérieur
            top_n         : nombre de buteurs à retourner par équipe

        Returns:
            dict avec top buteurs pour chaque équipe
        """
        if not self.fitted_:
            raise RuntimeError("Lance fit() d'abord.")

        result = {}
        for team, expected_goals in [(home, expected_home), (away, expected_away)]:
            if team not in self.team_scorers_:
                # Pas de données pour cette équipe
                result[team] = pd.DataFrame(
                    {"scorer": ["Données insuffisantes"], "prob_goal": [expected_goals]}
                )
                continue

            scorers = self.team_scorers_[team].copy()

            # Probabilité de marquer au moins 1 but dans ce match
            # P(marquer) = 1 - P(0 but) = 1 - e^(-lambda_joueur)
            scorers["expected_goals"] = expected_goals * scorers["ratio"]
            scorers["prob_goal"]      = 1 - np.exp(-scorers["expected_goals"])
            scorers["prob_goal_pct"]  = (scorers["prob_goal"] * 100).round(1)

            result[team] = scorers.head(top_n)[
                ["scorer", "goals", "expected_goals", "prob_goal_pct"]
            ].rename(columns={
                "goals":          "Buts historiques",
                "expected_goals": "Buts attendus",
                "prob_goal_pct":  "% chance de marquer",
            })

        return result

    def predict_tournament_scorers(
        self,
        fixtures: pd.DataFrame,
        match_predictions: pd.DataFrame,
        top_n: int = 15,
    ) -> pd.DataFrame:
        """
        Prédit les meilleurs buteurs sur l'ensemble du tournoi
        (phase de groupes uniquement — on ne connaît pas encore la phase finale).

        Pour chaque joueur :
          total_buts_attendus = Σ (buts_attendus_équipe × ratio_joueur)
          sur tous les matchs de groupe de son équipe

        Args:
            fixtures          : DataFrame des matchs WC 2026 (load_wc2026_fixtures)
            match_predictions : DataFrame des prédictions (predict_all_fixtures)
            top_n             : nombre de joueurs dans le classement

        Returns:
            DataFrame des meilleurs buteurs attendus sur la phase de groupes
        """
        if not self.fitted_:
            raise RuntimeError("Lance fit() d'abord.")

        logger.info("Calcul des buteurs attendus sur la phase de groupes...")

        # Fusionner fixtures et prédictions
        df = match_predictions.copy()

        # Accumuler les buts attendus par joueur
        player_goals = {}  # {(team, scorer): expected_goals}

        for _, row in df.iterrows():
            home      = row["home"]
            away      = row["away"]
            exp_home  = row["buts_home"]
            exp_away  = row["buts_away"]

            for team, exp_goals in [(home, exp_home), (away, exp_away)]:
                if team not in self.team_scorers_:
                    continue

                scorers = self.team_scorers_[team]
                for _, player in scorers.iterrows():
                    key = (team, player["scorer"])
                    contrib = exp_goals * player["ratio"]
                    player_goals[key] = player_goals.get(key, 0) + contrib

        # Convertir en DataFrame
        rows = []
        for (team, scorer), exp_goals in player_goals.items():
            # Récupérer les stats historiques
            team_df = self.team_scorers_.get(team, pd.DataFrame())
            if not team_df.empty:
                player_row = team_df[team_df["scorer"] == scorer]
                hist_goals = player_row["goals"].values[0] if len(player_row) else 0
                penalties  = player_row["penalties"].values[0] if len(player_row) else 0
            else:
                hist_goals = 0
                penalties  = 0

            rows.append({
                "Joueur":              scorer,
                "Équipe":              team,
                "Buts attendus (WC)":  round(exp_goals, 2),
                "Buts historiques":    int(hist_goals),
                "dont penalties":      int(penalties),
            })

        df_result = (
            pd.DataFrame(rows)
            .sort_values("Buts attendus (WC)", ascending=False)
            .reset_index(drop=True)
        )
        df_result.index += 1  # classement à partir de 1

        logger.success(f"✅ Classement buteurs calculé — {len(df_result)} joueurs")
        return df_result.head(top_n)

    def get_team_top_scorers(self, team: str, top_n: int = 8) -> pd.DataFrame:
        """
        Retourne les meilleurs buteurs d'une équipe avec leurs stats.

        Args:
            team  : nom de l'équipe
            top_n : nombre de joueurs

        Returns:
            DataFrame avec stats des buteurs
        """
        if team not in self.team_scorers_:
            return pd.DataFrame({"Info": [f"Pas de données pour {team}"]})

        df = self.team_scorers_[team].head(top_n).copy().reset_index(drop=True)
        df["% buts de l'équipe"] = (df["ratio"] * 100).round(1)
        return df[["scorer", "goals", "penalties", "% buts de l'équipe"]].rename(columns={
            "scorer":    "Joueur",
            "goals":     "Buts (depuis 2020)",
            "penalties": "dont pénaltys",
        }).reset_index(drop=True)


# ============================================================
# POINT D'ENTRÉE
# ============================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.data.collect import load_results, load_goalscorers, load_wc2026_fixtures
    from src.models.match_predictor import PoissonPredictor

    print("=" * 65)
    print("  World Cup 2026 — Prédiction des buteurs")
    print("=" * 65)

    # Chargement
    results     = load_results(min_year=2018)
    goalscorers = load_goalscorers(min_year=2020)
    fixtures    = load_wc2026_fixtures()

    # Modèle de match
    match_model = PoissonPredictor(min_year=2018, decay_rate=0.005, n_iter=30)
    match_model.fit(results)

    # Modèle buteurs
    scorer_model = ScorerPredictor(min_year=2020)
    scorer_model.fit(goalscorers)

    # Prédictions des matchs
    all_preds = match_model.predict_all_fixtures(fixtures)

    # ── Buteurs attendus sur la phase de groupes ──────────────
    print("\n🥇 TOP 20 BUTEURS ATTENDUS — Phase de groupes WC 2026\n")
    top_scorers = scorer_model.predict_tournament_scorers(
        fixtures, all_preds, top_n=20
    )
    print(top_scorers.to_string())

    # ── Buteurs par match — exemples ──────────────────────────
    print("\n\n⚽ BUTEURS PROBABLES — Matchs emblématiques\n")
    test_matches = [
        ("France",    "Senegal"),
        ("Argentina", "Algeria"),
        ("Spain",     "Cape Verde"),
        ("England",   "Croatia"),
    ]

    for home, away in test_matches:
        pred = match_model.predict_score(home, away)
        scorers = scorer_model.predict_match_scorers(
            home, away,
            pred["expected_home"],
            pred["expected_away"],
            top_n=3,
        )
        print(f"  {home} {pred['rounded_score']} {away}")
        print(f"  (buts attendus : {pred['expected_home']:.2f} - {pred['expected_away']:.2f})")
        print(f"  {home} :")
        if home in scorers and not scorers[home].empty:
            for _, r in scorers[home].iterrows():
                print(f"    → {r['scorer']:25} {r['% chance de marquer']:.1f}%")
        print(f"  {away} :")
        if away in scorers and not scorers[away].empty:
            for _, r in scorers[away].iterrows():
                print(f"    → {r['scorer']:25} {r['% chance de marquer']:.1f}%")
        print()

    # ── Meilleurs buteurs par équipe ──────────────────────────
    print("\n🏴 TOP BUTEURS PAR ÉQUIPE FAVORITE\n")
    for team in ["France", "Spain", "Argentina", "England", "Portugal", "Brazil"]:
        print(f"  {team} :")
        df_t = scorer_model.get_team_top_scorers(team, top_n=5)
        print(df_t.to_string(index=False))
        print()
