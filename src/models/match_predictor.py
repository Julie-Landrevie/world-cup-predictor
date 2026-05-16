"""
src/models/match_predictor.py
------------------------------
Modèle de Poisson amélioré — v3

3 AMÉLIORATIONS PAR RAPPORT À LA V2 :

1. DONNÉES RÉCENTES RENFORCÉES
   min_year=2018, decay_rate=0.005
   Les matchs de 2018-2026 comptent beaucoup plus.
   Les matchs d'avant 2018 sont exclus.

2. QUALIFICATIONS WC SÉPARÉES
   Les qualifications WC reçoivent un poids 3.0 (au lieu de 2.0)
   car elles sont les plus représentatives du niveau réel en vue du tournoi.

3. CLASSEMENT FIFA INTÉGRÉ
   Le classement FIFA avril 2026 est encodé manuellement
   (FIFA bloque le scraping automatique).
   Il sert de facteur de correction sur les forces d'attaque/défense :
   une équipe bien classée mais avec peu de matchs récents
   dans le dataset est "boostée" vers son vrai niveau.

POINT D'ATTENTION (inchangé) :
   goalscorers.csv incomplet sur certaines périodes.
   Les prédictions de buteurs individuels ont une marge d'incertitude.
"""

import pandas as pd
import numpy as np
from scipy.stats import poisson
from pathlib import Path
from loguru import logger

# ============================================================
# CLASSEMENT FIFA — Avril 2026
# Source : fifa.com/fifa-world-ranking
# Encodé manuellement car FIFA bloque le scraping
# ============================================================

FIFA_RANKING = {
    # Top 20
    "Argentina":       1,
    "France":          2,
    "Spain":           3,
    "England":         4,
    "Brazil":          5,
    "Portugal":        6,
    "Netherlands":     7,
    "Belgium":         8,
    "Germany":         9,
    "Uruguay":        10,
    "Colombia":       11,
    "Italy":          12,
    "Croatia":        13,
    "Morocco":        14,
    "United States":  15,
    "Mexico":         16,
    "Senegal":        17,
    "Denmark":        18,
    "Switzerland":    19,
    "Japan":          20,
    # 21-48
    "South Korea":    21,
    "Ecuador":        22,
    "Austria":        23,
    "Ukraine":        24,
    "Turkey":         25,
    "Australia":      26,
    "Hungary":        27,
    "Norway":         28,
    "Czech Republic": 29,
    "Poland":         30,
    "Serbia":         31,
    "Sweden":         32,
    "Canada":         33,
    "Algeria":        34,
    "Ivory Coast":    35,
    "Ghana":          36,
    "Tunisia":        37,
    "Saudi Arabia":   38,
    "Egypt":          39,
    "South Africa":   40,
    "Nigeria":        41,
    "Cameroon":       42,
    "Paraguay":       43,
    "Iran":           44,
    "DR Congo":       45,
    "Panama":         46,
    "Scotland":       47,
    "Bolivia":        48,
    # Reste des équipes qualifiées WC 2026
    "Qatar":          60,
    "Bosnia and Herzegovina": 55,
    "Slovakia":       50,
    "New Zealand":    95,
    "Haiti":         105,
    "Jamaica":        65,
    "Honduras":       70,
    "Cuba":          140,
    "Jordan":         75,
    "Uzbekistan":     72,
    "Iraq":           68,
    "Indonesia":      90,
    "Thailand":      115,
    "Cape Verde":     80,
    "Curaçao":       85,
    "Croatia":        13,
}

# Nombre total d'équipes FIFA (pour normalisation)
FIFA_TOTAL_TEAMS = 210

def get_fifa_score(team: str) -> float:
    """
    Convertit le classement FIFA en score entre 0.5 et 1.5.
    Rang 1 → 1.5, rang 100+ → 0.5.
    Équipes inconnues → 1.0 (neutre).
    """
    rank = FIFA_RANKING.get(team, None)
    if rank is None:
        return 1.0
    # Normalisation linéaire inverse : meilleur rang = score plus élevé
    score = 1.5 - (rank - 1) / (FIFA_TOTAL_TEAMS - 1)
    return round(max(0.5, min(1.5, score)), 3)


# ============================================================
# POIDS DES COMPÉTITIONS — v3
# Qualification WC séparée avec poids plus élevé
# ============================================================

TOURNAMENT_WEIGHTS = {
    # Compétitions majeures — poids maximum
    "FIFA World Cup":                       4.0,
    "UEFA Euro":                            3.0,
    "Copa América":                         3.0,
    "African Cup of Nations":               2.5,
    "AFC Asian Cup":                        2.5,
    "Gold Cup":                             2.0,

    # Qualifications WC — poids élevé (représentatif du niveau réel)
    "FIFA World Cup qualification":         3.0,  # ← augmenté de 2.0 à 3.0

    # Autres qualifications
    "UEFA Euro qualification":              1.8,
    "African Cup of Nations qualification": 1.8,
    "AFC Asian Cup qualification":          1.8,
    "CONCACAF Nations League":              1.8,

    # Nations League et tournois régionaux
    "UEFA Nations League":                  2.0,  # ← augmenté car compétitif

    # Amicaux — poids faible
    "Friendly":                             0.2,  # ← réduit de 0.3 à 0.2
}

DEFAULT_WEIGHT = 1.2


class PoissonPredictor:
    """
    Modèle de Poisson amélioré v3.

    Nouveautés :
      - Données post-2018 uniquement
      - Qualifications WC pondérées à 3.0
      - Classement FIFA intégré comme facteur de correction
    """

    def __init__(self, min_year: int = 2018, decay_rate: float = 0.005, n_iter: int = 50):
        """
        Args:
            min_year   : 2018 par défaut (ère moderne post-Russie)
            decay_rate : 0.005 = les matchs de 2021 comptent ~50% moins que 2026
            n_iter     : itérations pour convergence
        """
        self.min_year   = min_year
        self.decay_rate = decay_rate
        self.n_iter     = n_iter
        self.teams_     = []
        self.att_       = {}
        self.def_       = {}
        self.mu_        = 0.0
        self.home_adv_  = 1.1
        self.fitted_    = False

    def _prepare(self, results: pd.DataFrame) -> pd.DataFrame:
        """Filtre et pondère les matchs."""
        df = results.copy()
        df = df[df["date"].dt.year >= self.min_year]
        df = df[df["home_score"].notna() & df["away_score"].notna()]

        # Poids compétition
        df["tournament_weight"] = (
            df["tournament"].map(TOURNAMENT_WEIGHTS).fillna(DEFAULT_WEIGHT)
        )

        # Poids temporel — décroissance plus forte (0.005)
        max_date = df["date"].max()
        days_ago = (max_date - df["date"]).dt.days
        df["time_weight"] = np.exp(-self.decay_rate * days_ago)

        # Poids FIFA — bonus pour matchs entre grandes équipes
        df["fifa_weight"] = df.apply(
            lambda r: (get_fifa_score(r["home_team"]) + get_fifa_score(r["away_team"])) / 2,
            axis=1
        )

        # Poids final = compétition × temps × fifa
        df["weight"] = df["tournament_weight"] * df["time_weight"] * df["fifa_weight"]

        # Équipes avec au moins 5 matchs
        counts = (
            df.groupby("home_team").size()
            .add(df.groupby("away_team").size(), fill_value=0)
        )
        valid = counts[counts >= 5].index
        df = df[df["home_team"].isin(valid) & df["away_team"].isin(valid)]

        return df

    def fit(self, results: pd.DataFrame) -> "PoissonPredictor":
        """Entraîne le modèle par itérations."""
        logger.info("Entraînement du modèle de Poisson v3...")
        df = self._prepare(results)

        self.teams_ = sorted(set(df["home_team"].tolist() + df["away_team"].tolist()))
        logger.info(f"  {len(df)} matchs, {len(self.teams_)} équipes")

        # Moyenne globale de buts pondérée
        total_w  = df["weight"].sum()
        self.mu_ = (
            (df["home_score"] * df["weight"]).sum() +
            (df["away_score"] * df["weight"]).sum()
        ) / (2 * total_w)

        # Initialisation — on part du score FIFA comme point de départ
        att  = {t: get_fifa_score(t) for t in self.teams_}
        deff = {t: 2.0 - get_fifa_score(t) for t in self.teams_}  # inverse : meilleur = moins encaisse

        # Itérations
        for iteration in range(self.n_iter):
            att_new  = {}
            deff_new = {}

            for team in self.teams_:
                home_mask = df["home_team"] == team
                away_mask = df["away_team"] == team

                # Buts marqués pondérés
                goals_scored = (
                    (df.loc[home_mask, "home_score"] * df.loc[home_mask, "weight"]).sum() +
                    (df.loc[away_mask, "away_score"] * df.loc[away_mask, "weight"]).sum()
                )

                # Buts attendus marqués
                expected_scored = 0.0
                for _, row in df[home_mask].iterrows():
                    expected_scored += row["weight"] * self.mu_ * att[team] * deff[row["away_team"]] * self.home_adv_
                for _, row in df[away_mask].iterrows():
                    expected_scored += row["weight"] * self.mu_ * att[team] * deff[row["home_team"]]

                att_new[team] = att[team] * (goals_scored / expected_scored) if expected_scored > 0 else att[team]

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

                deff_new[team] = deff[team] * (goals_conceded / expected_conceded) if expected_conceded > 0 else deff[team]

            # Normalisation
            att_mean  = np.mean(list(att_new.values()))
            deff_mean = np.mean(list(deff_new.values()))
            att  = {t: v / att_mean  for t, v in att_new.items()}
            deff = {t: v / deff_mean for t, v in deff_new.items()}

            # Correction FIFA à chaque itération — on tire doucement vers le ranking
            fifa_weight = max(0.0, 0.3 - iteration * 0.006)  # diminue progressivement
            for team in self.teams_:
                fifa_s = get_fifa_score(team)
                att[team]  = (1 - fifa_weight) * att[team]  + fifa_weight * fifa_s
                deff[team] = (1 - fifa_weight) * deff[team] + fifa_weight * (2.0 - fifa_s)

            if iteration % 10 == 0:
                logger.debug(f"  Itération {iteration}/{self.n_iter}")

        self.att_    = att
        self.def_    = deff
        self.fitted_ = True
        logger.success(f"✅ Modèle v3 entraîné — mu={self.mu_:.3f}")
        return self

    def predict_score(self, home: str, away: str, neutral: bool = True, max_goals: int = 8) -> dict:
        """Prédit la distribution des scores pour un match."""
        if not self.fitted_:
            raise RuntimeError("Lance fit() d'abord.")

        att_h = self.att_.get(home, get_fifa_score(home))
        def_h = self.def_.get(home, 2.0 - get_fifa_score(home))
        att_a = self.att_.get(away, get_fifa_score(away))
        def_a = self.def_.get(away, 2.0 - get_fifa_score(away))

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

        # Score arrondi (plus lisible que le mode de Poisson)
        rounded_score = f"{round(lam_h)}-{round(lam_a)}"

        prob_home = float(np.sum(np.tril(score_matrix, -1)))
        prob_draw = float(np.sum(np.diag(score_matrix)))
        prob_away = float(np.sum(np.triu(score_matrix, 1)))

        return {
            "home":           home,
            "away":           away,
            "expected_home":  round(lam_h, 2),
            "expected_away":  round(lam_a, 2),
            "most_likely":    most_likely,
            "rounded_score":  rounded_score,
            "prob_home_win":  round(prob_home * 100, 1),
            "prob_draw":      round(prob_draw * 100, 1),
            "prob_away_win":  round(prob_away * 100, 1),
            "score_matrix":   score_matrix,
        }

    def predict_all_fixtures(self, fixtures: pd.DataFrame) -> pd.DataFrame:
        """Prédit tous les matchs WC 2026."""
        logger.info(f"Prédiction de {len(fixtures)} matchs...")
        rows = []
        for _, row in fixtures.iterrows():
            pred = self.predict_score(row["home_team"], row["away_team"], neutral=True)
            rows.append({
                "date":           row["date"].strftime("%d/%m/%Y"),
                "home":           row["home_team"],
                "away":           row["away_team"],
                "score_prédit":   pred["rounded_score"],
                "buts_home":      pred["expected_home"],
                "buts_away":      pred["expected_away"],
                "% victoire":     pred["prob_home_win"],
                "% nul":          pred["prob_draw"],
                "% défaite":      pred["prob_away_win"],
            })
        return pd.DataFrame(rows)

    def team_strength(self, team: str) -> dict:
        """Force d'une équipe + classement FIFA."""
        return {
            "team":        team,
            "fifa_rank":   FIFA_RANKING.get(team, "?"),
            "attaque":     round(self.att_.get(team, 1.0), 3),
            "defense":     round(self.def_.get(team, 1.0), 3),
            "fifa_score":  get_fifa_score(team),
        }


# ============================================================
# POINT D'ENTRÉE
# ============================================================

if __name__ == "__main__":
    from src.data.collect import load_results, load_wc2026_fixtures

    print("=" * 65)
    print("  World Cup 2026 — Modèle de Poisson v3")
    print("  (données 2018+, qualif WC 3.0, classement FIFA)")
    print("=" * 65)

    results  = load_results(min_year=2018)
    fixtures = load_wc2026_fixtures()

    model = PoissonPredictor(min_year=2018, decay_rate=0.005, n_iter=50)
    model.fit(results)

    print("\n🏆 PRÉDICTIONS — matchs emblématiques\n")
    test_matches = [
        ("France",        "Argentina"),
        ("Brazil",        "Germany"),
        ("Spain",         "England"),
        ("United States", "Paraguay"),
        ("Mexico",        "South Africa"),
        ("France",        "Norway"),
    ]
    for home, away in test_matches:
        if home in model.teams_ and away in model.teams_:
            pred = model.predict_score(home, away)
            print(f"  {home:22} vs {away:22}")
            print(f"    Score prédit  : {pred['rounded_score']} (buts attendus : {pred['expected_home']:.2f}-{pred['expected_away']:.2f})")
            print(f"    Victoire {pred['prob_home_win']}% | Nul {pred['prob_draw']}% | Défaite {pred['prob_away_win']}%")
            print()

    print("💪 FORCES DES ÉQUIPES FAVORITES\n")
    favorites = ["Argentina", "France", "Spain", "England", "Brazil", "Germany", "Portugal"]
    rows = []
    for team in favorites:
        if team in model.teams_:
            rows.append(model.team_strength(team))
    df_str = pd.DataFrame(rows).sort_values("attaque", ascending=False)
    print(df_str.to_string(index=False))

    print("\n📋 TOUTES LES PRÉDICTIONS WC 2026\n")
    all_preds = model.predict_all_fixtures(fixtures)
    print(all_preds.to_string(index=False))
