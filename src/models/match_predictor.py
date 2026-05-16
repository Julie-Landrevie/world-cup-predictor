"""
src/models/match_predictor.py
------------------------------
Modèle de Poisson pour prédire les scores des matchs WC 2026.

VERSION RAPIDE : calcul direct des forces d'attaque/défense
par itération (méthode de Dixon-Coles simplifiée).
Tourne en ~2 secondes au lieu de plusieurs minutes.

POINT D'ATTENTION :
  Le fichier goalscorers.csv est incomplet sur certaines périodes.
  Les stats d'équipe (results.csv) sont fiables mais les prédictions
  de buteurs individuels ont une marge d'incertitude plus grande.
"""

import pandas as pd
import numpy as np
from scipy.stats import poisson
from pathlib import Path
from loguru import logger

# Poids des compétitions
TOURNAMENT_WEIGHTS = {
    "FIFA World Cup":                       3.0,
    "FIFA World Cup qualification":         2.0,
    "UEFA Euro":                            2.5,
    "UEFA Euro qualification":              1.5,
    "Copa América":                         2.5,
    "African Cup of Nations":               2.0,
    "African Cup of Nations qualification": 1.5,
    "AFC Asian Cup":                        2.0,
    "AFC Asian Cup qualification":          1.5,
    "UEFA Nations League":                  1.8,
    "CONCACAF Nations League":              1.5,
    "Gold Cup":                             1.5,
    "Friendly":                             0.3,
}
DEFAULT_WEIGHT = 1.0


class PoissonPredictor:
    """
    Modèle de Poisson rapide — calcul direct des forces par itération.

    Utilisation :
        model = PoissonPredictor()
        model.fit(results_df)
        pred = model.predict_score("France", "Brazil")
    """

    def __init__(self, min_year: int = 2010, decay_rate: float = 0.003, n_iter: int = 50):
        """
        Args:
            min_year   : année minimum des matchs utilisés
            decay_rate : taux de décroissance temporelle
            n_iter     : nombre d'itérations pour converger (50 suffit)
        """
        self.min_year   = min_year
        self.decay_rate = decay_rate
        self.n_iter     = n_iter
        self.teams_     = []
        self.att_       = {}
        self.def_       = {}
        self.mu_        = 0.0
        self.home_adv_  = 1.1  # avantage domicile typique en football international
        self.fitted_    = False

    def _prepare(self, results: pd.DataFrame) -> pd.DataFrame:
        """Filtre et pondère les matchs."""
        df = results.copy()
        df = df[df["date"].dt.year >= self.min_year]
        df = df[df["home_score"].notna() & df["away_score"].notna()]

        # Poids compétition
        df["tournament_weight"] = df["tournament"].map(TOURNAMENT_WEIGHTS).fillna(DEFAULT_WEIGHT)

        # Poids temporel : décroissance exponentielle
        max_date = df["date"].max()
        days_ago = (max_date - df["date"]).dt.days
        df["time_weight"] = np.exp(-self.decay_rate * days_ago)

        # Poids final
        df["weight"] = df["tournament_weight"] * df["time_weight"]

        # Garder uniquement équipes avec au moins 5 matchs
        counts = (
            df.groupby("home_team").size()
            .add(df.groupby("away_team").size(), fill_value=0)
        )
        valid = counts[counts >= 5].index
        df = df[df["home_team"].isin(valid) & df["away_team"].isin(valid)]

        return df

    def fit(self, results: pd.DataFrame) -> "PoissonPredictor":
        """
        Entraîne le modèle par itérations successives.

        Algorithme :
          1. Initialiser att=1, def=1 pour toutes les équipes
          2. Mettre à jour att de chaque équipe = buts_marqués / buts_attendus
          3. Mettre à jour def de chaque équipe = buts_encaissés / buts_attendus
          4. Normaliser pour que la moyenne = 1
          5. Répéter jusqu'à convergence

        C'est la méthode des moindres carrés alternés, beaucoup plus
        rapide que l'optimisation globale.
        """
        logger.info("Entraînement du modèle de Poisson (méthode itérative)...")
        df = self._prepare(results)

        self.teams_ = sorted(set(df["home_team"].tolist() + df["away_team"].tolist()))
        logger.info(f"  {len(df)} matchs, {len(self.teams_)} équipes")

        # Moyenne globale de buts
        total_w = df["weight"].sum()
        self.mu_ = (
            (df["home_score"] * df["weight"]).sum() +
            (df["away_score"] * df["weight"]).sum()
        ) / (2 * total_w)

        # Initialisation
        att = {t: 1.0 for t in self.teams_}
        deff = {t: 1.0 for t in self.teams_}

        # Itérations
        for iteration in range(self.n_iter):
            att_new  = {}
            deff_new = {}

            for team in self.teams_:
                # Matchs à domicile
                home_mask = df["home_team"] == team
                # Matchs à l'extérieur
                away_mask = df["away_team"] == team

                # Buts marqués pondérés
                goals_scored = (
                    (df.loc[home_mask, "home_score"] * df.loc[home_mask, "weight"]).sum() +
                    (df.loc[away_mask, "away_score"] * df.loc[away_mask, "weight"]).sum()
                )

                # Buts attendus marqués (selon paramètres actuels)
                expected_scored = 0.0
                for _, row in df[home_mask].iterrows():
                    expected_scored += row["weight"] * self.mu_ * att[team] * deff[row["away_team"]] * self.home_adv_
                for _, row in df[away_mask].iterrows():
                    expected_scored += row["weight"] * self.mu_ * att[team] * deff[row["home_team"]]

                # Mise à jour attaque
                if expected_scored > 0:
                    att_new[team] = att[team] * (goals_scored / expected_scored)
                else:
                    att_new[team] = att[team]

                # Buts encaissés pondérés
                goals_conceded = (
                    (df.loc[home_mask, "away_score"] * df.loc[home_mask, "weight"]).sum() +
                    (df.loc[away_mask, "home_score"] * df.loc[away_mask, "weight"]).sum()
                )

                # Buts attendus encaissés
                expected_conceded = 0.0
                for _, row in df[home_mask].iterrows():
                    expected_conceded += row["weight"] * self.mu_ * att[row["away_team"]] * deff[team]
                for _, row in df[away_mask].iterrows():
                    expected_conceded += row["weight"] * self.mu_ * att[row["home_team"]] * deff[team] * self.home_adv_

                # Mise à jour défense
                if expected_conceded > 0:
                    deff_new[team] = deff[team] * (goals_conceded / expected_conceded)
                else:
                    deff_new[team] = deff[team]

            # Normalisation
            att_mean  = np.mean(list(att_new.values()))
            deff_mean = np.mean(list(deff_new.values()))
            att  = {t: v / att_mean  for t, v in att_new.items()}
            deff = {t: v / deff_mean for t, v in deff_new.items()}

            if iteration % 10 == 0:
                logger.debug(f"  Itération {iteration}/{self.n_iter}")

        self.att_    = att
        self.def_    = deff
        self.fitted_ = True
        logger.success(f"✅ Modèle entraîné — mu={self.mu_:.3f}, home_adv={self.home_adv_:.2f}")
        return self

    def predict_score(self, home: str, away: str, neutral: bool = True, max_goals: int = 8) -> dict:
        """
        Prédit la distribution des scores pour un match.

        Args:
            home      : équipe 1
            away      : équipe 2
            neutral   : terrain neutre (True pour WC 2026)
            max_goals : buts max considérés par équipe

        Returns:
            dict avec score prédit, probabilités, buts attendus
        """
        if not self.fitted_:
            raise RuntimeError("Lance fit() d'abord.")

        att_h = self.att_.get(home, 1.0)
        def_h = self.def_.get(home, 1.0)
        att_a = self.att_.get(away, 1.0)
        def_a = self.def_.get(away, 1.0)

        home_factor = 1.0 if neutral else self.home_adv_

        lam_h = self.mu_ * att_h * def_a * home_factor
        lam_a = self.mu_ * att_a * def_h

        # Matrice de probabilités
        score_matrix = np.zeros((max_goals + 1, max_goals + 1))
        for h in range(max_goals + 1):
            for a in range(max_goals + 1):
                score_matrix[h, a] = poisson.pmf(h, lam_h) * poisson.pmf(a, lam_a)
        score_matrix /= score_matrix.sum()

        idx         = np.unravel_index(score_matrix.argmax(), score_matrix.shape)
        most_likely = f"{idx[0]}-{idx[1]}"
        prob_home   = float(np.sum(np.tril(score_matrix, -1)))
        prob_draw   = float(np.sum(np.diag(score_matrix)))
        prob_away   = float(np.sum(np.triu(score_matrix, 1)))

        return {
            "home":          home,
            "away":          away,
            "expected_home": round(lam_h, 2),
            "expected_away": round(lam_a, 2),
            "most_likely":   most_likely,
            "prob_home_win": round(prob_home * 100, 1),
            "prob_draw":     round(prob_draw * 100, 1),
            "prob_away_win": round(prob_away * 100, 1),
            "score_matrix":  score_matrix,
        }

    def predict_all_fixtures(self, fixtures: pd.DataFrame) -> pd.DataFrame:
        """Prédit tous les matchs WC 2026."""
        logger.info(f"Prédiction de {len(fixtures)} matchs...")
        rows = []
        for _, row in fixtures.iterrows():
            pred = self.predict_score(row["home_team"], row["away_team"], neutral=True)
            rows.append({
                "date":         row["date"].strftime("%d/%m/%Y"),
                "home":         row["home_team"],
                "away":         row["away_team"],
                "score_prédit": pred["most_likely"],
                "buts_home":    pred["expected_home"],
                "buts_away":    pred["expected_away"],
                "% victoire":   pred["prob_home_win"],
                "% nul":        pred["prob_draw"],
                "% défaite":    pred["prob_away_win"],
            })
        return pd.DataFrame(rows)

    def team_strength(self, team: str) -> dict:
        """Force d'une équipe."""
        return {
            "team":    team,
            "attaque": round(self.att_.get(team, 1.0), 3),
            "defense": round(self.def_.get(team, 1.0), 3),
        }


# ============================================================
# POINT D'ENTRÉE
# ============================================================

if __name__ == "__main__":
    from src.data.collect import load_results, load_wc2026_fixtures

    print("=" * 60)
    print("  World Cup 2026 — Modèle de Poisson")
    print("=" * 60)

    results  = load_results(min_year=2010)
    fixtures = load_wc2026_fixtures()

    model = PoissonPredictor(min_year=2010, decay_rate=0.003, n_iter=50)
    model.fit(results)

    print("\n🏆 PRÉDICTIONS — matchs emblématiques\n")
    test_matches = [
        ("France",        "Argentina"),
        ("Brazil",        "Germany"),
        ("Spain",         "England"),
        ("United States", "Paraguay"),
        ("Mexico",        "South Africa"),
    ]
    for home, away in test_matches:
        if home in model.teams_ and away in model.teams_:
            pred = model.predict_score(home, away)
            print(f"  {home:22} vs {away:22}")
            print(f"    Score prédit  : {pred['most_likely']}")
            print(f"    Buts attendus : {pred['expected_home']:.2f} — {pred['expected_away']:.2f}")
            print(f"    Victoire {pred['prob_home_win']}% | Nul {pred['prob_draw']}% | Défaite {pred['prob_away_win']}%")
            print()

    print("💪 FORCES DES ÉQUIPES FAVORITES\n")
    for team in ["France", "Brazil", "Argentina", "Spain", "England", "Germany", "Portugal"]:
        if team in model.teams_:
            s = model.team_strength(team)
            print(f"  {s['team']:15} attaque={s['attaque']}  défense={s['defense']}")

    print("\n📋 TOUTES LES PRÉDICTIONS WC 2026\n")
    all_preds = model.predict_all_fixtures(fixtures)
    print(all_preds.to_string(index=False))
