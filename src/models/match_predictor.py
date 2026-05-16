"""
src/models/match_predictor.py
------------------------------
Modèle de Poisson amélioré — v4

AMÉLIORATIONS PAR RAPPORT À LA V3 :

1. COEFFICIENTS DE CONFÉDÉRATION
   UEFA et CONMEBOL boostés car historiquement les seules à remporter la WC.
   CAF, AFC, CONCACAF, OFC avec des coefficients réduits.

2. EURO VALORISÉ VS AUTRES TOURNOIS CONTINENTAUX
   UEFA Euro : poids 3.5 (vs CAN 2.0, Gold Cup 1.8)
   Reflète le niveau objectivement plus élevé de l'Euro.

3. COTES BOOKMAKERS INTÉGRÉES
   Probabilités agrégées depuis Polymarket + Kalshi (mai 2026)
   ~1 milliard de dollars de volume — signal très fiable.
   Utilisées comme correction finale sur les forces d'équipe.

4. FACTEUR D'EXPÉRIENCE WC
   Bonus pour équipes ayant atteint les QF ou + en 2018/2022.

5. PLAFONNEMENT AMÉLIORÉ
   att max 2.2, def min 0.2 pour éviter les valeurs extrêmes.

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

# ============================================================
# COTES BOOKMAKERS — Polymarket + Kalshi agrégés (mai 2026)
# Source : defirate.com/prediction-markets/world-cup-odds
# ~1 milliard $ de volume — signal très fiable
# Probabilités implicites (après correction marge bookmaker)
# ============================================================

BOOKMAKER_PROBS = {
    "France":        0.174,  # +475 — favori marché
    "Spain":         0.165,  # +505
    "England":       0.113,  # +782
    "Argentina":     0.105,  # ~+850
    "Brazil":        0.095,  # +800
    "Germany":       0.065,  # ~+1200
    "Portugal":      0.055,  # ~+1400
    "Netherlands":   0.040,
    "Morocco":       0.020,  # 50-1
    "United States": 0.016,  # 60-1 (hôte)
    "Belgium":       0.015,
    "Colombia":      0.014,
    "Switzerland":   0.012,  # 80-1
    "Croatia":       0.012,  # 80-1
    "Norway":        0.010,  # 30-1 (après allongement)
    "Mexico":        0.009,  # 75-1
    "Japan":         0.008,
    "South Korea":   0.007,
    "Australia":     0.006,
    "Turkey":        0.006,  # 100-1
    "Uruguay":       0.008,
    "Ecuador":       0.006,  # 90-1
    "Senegal":       0.005,
    "Sweden":        0.005,
    "Austria":       0.004,
    "Canada":        0.004,
    "Ghana":         0.003,
    "Ivory Coast":   0.003,
    "Tunisia":       0.002,
    "Saudi Arabia":  0.002,
    "South Africa":  0.002,
    "Scotland":      0.002,
    "Czech Republic":0.003,
    "Bosnia and Herzegovina": 0.002,
    "Paraguay":      0.002,
    "Algeria":       0.002,
    "Iran":          0.001,
    "Egypt":         0.001,
    "DR Congo":      0.001,
    "New Zealand":   0.001,
    "Jordan":        0.0005,
    "Iraq":          0.0005,
    "Uzbekistan":    0.001,
    "Qatar":         0.001,
    "Haiti":         0.0002,
    "Panama":        0.0005,
    "Cape Verde":    0.001,
    "Curaçao":       0.0002,
}

# ============================================================
# COEFFICIENTS DE CONFÉDÉRATION
# Basés sur l'historique des vainqueurs de la WC
# UEFA: 12 victoires, CONMEBOL: 9, autres: 0
# ============================================================

CONFEDERATION = {
    # UEFA — Europe
    "France": "UEFA", "Spain": "UEFA", "England": "UEFA",
    "Germany": "UEFA", "Portugal": "UEFA", "Netherlands": "UEFA",
    "Belgium": "UEFA", "Croatia": "UEFA", "Switzerland": "UEFA",
    "Norway": "UEFA", "Sweden": "UEFA", "Denmark": "UEFA",
    "Austria": "UEFA", "Czech Republic": "UEFA", "Scotland": "UEFA",
    "Turkey": "UEFA", "Bosnia and Herzegovina": "UEFA",
    "Hungary": "UEFA", "Ukraine": "UEFA", "Serbia": "UEFA",
    "Poland": "UEFA", "Slovakia": "UEFA", "Albania": "UEFA",
    "Georgia": "UEFA", "Slovenia": "UEFA", "Romania": "UEFA",

    # CONMEBOL — Amérique du Sud
    "Argentina": "CONMEBOL", "Brazil": "CONMEBOL",
    "Colombia": "CONMEBOL", "Uruguay": "CONMEBOL",
    "Ecuador": "CONMEBOL", "Paraguay": "CONMEBOL",
    "Bolivia": "CONMEBOL", "Chile": "CONMEBOL",
    "Peru": "CONMEBOL", "Venezuela": "CONMEBOL",

    # CONCACAF — Amérique du Nord/Centrale
    "United States": "CONCACAF", "Mexico": "CONCACAF",
    "Canada": "CONCACAF", "Panama": "CONCACAF",
    "Costa Rica": "CONCACAF", "Jamaica": "CONCACAF",
    "Honduras": "CONCACAF", "Haiti": "CONCACAF",
    "Curaçao": "CONCACAF", "Cuba": "CONCACAF",

    # CAF — Afrique
    "Morocco": "CAF", "Senegal": "CAF", "Ivory Coast": "CAF",
    "Ghana": "CAF", "Tunisia": "CAF", "Egypt": "CAF",
    "South Africa": "CAF", "DR Congo": "CAF", "Algeria": "CAF",
    "Cameroon": "CAF", "Nigeria": "CAF", "Cape Verde": "CAF",
    "Mali": "CAF",

    # AFC — Asie
    "Japan": "AFC", "South Korea": "AFC", "Australia": "AFC",
    "Saudi Arabia": "AFC", "Iran": "AFC", "Jordan": "AFC",
    "Iraq": "AFC", "Uzbekistan": "AFC", "Qatar": "AFC",
    "Indonesia": "AFC", "Thailand": "AFC",

    # OFC — Océanie
    "New Zealand": "OFC",
}

CONFEDERATION_BOOST = {
    "UEFA":     1.12,   # 12 victoires WC — très largement dominant
    "CONMEBOL": 1.09,   # 9 victoires WC — Argentine/Brésil/Uruguay
    "CONCACAF": 0.95,   # compétitifs mais jamais vainqueurs
    "CAF":      0.91,   # jamais vainqueur WC malgré des performances récentes
    "AFC":      0.90,   # jamais vainqueur WC
    "OFC":      0.86,   # niveau structurellement plus faible
}

# ============================================================
# EXPÉRIENCE WC — Bonus pour équipes récemment performantes
# Quarts de finale ou mieux en 2018 ET/OU 2022
# ============================================================

# ============================================================
# EXPÉRIENCE WC — Système de points sur 2018 ET 2022
# Points : Champion=10, Final=7, SF=5, QF=3, R16=1, Groupes=0
# Bonus = 1.0 + (points_2018 + points_2022) / 100
# → Seules les équipes performantes SUR LES DEUX tournois
#   obtiennent un bonus significatif
# ============================================================

WC_PERFORMANCE = {
    # (points_2018, points_2022)
    "France":        (7, 3),   # Final 2018, QF 2022      → +10%
    "Argentina":     (1, 10),  # R16 2018, Champion 2022  → +11%
    "Croatia":       (7, 5),   # Final 2018, SF 2022      → +12%
    "England":       (5, 3),   # SF 2018, QF 2022         → +8%
    "Belgium":       (5, 0),   # 3e 2018, groupes 2022    → +5%
    "Brazil":        (3, 3),   # QF 2018, QF 2022         → +6%
    "Uruguay":       (3, 0),   # QF 2018, groupes 2022    → +3%
    "Sweden":        (3, 0),   # QF 2018, non qualifiée   → +3%
    "Netherlands":   (0, 3),   # absente 2018, QF 2022    → +3%
    "Portugal":      (0, 3),   # R16 2018, QF 2022        → +4%
    "Spain":         (1, 3),   # R16 2018, QF 2022        → +4%
    "Switzerland":   (1, 3),   # R16 2018, QF 2022        → +4%
    "Morocco":       (0, 5),   # groupes 2018, SF 2022    → +5% (une seule WC)
    "Japan":         (1, 1),   # R16 2018, R16 2022       → +2%
    "South Korea":   (1, 1),   # R16 2018, R16 2022       → +2%
    "Australia":     (0, 1),   # absente 2018, R16 2022   → +1%
    "Senegal":       (0, 1),   # absente 2018, R16 2022   → +1%
    "United States": (0, 1),   # absente 2018, R16 2022   → +1%
    "Germany":       (0, 0),   # groupes 2018, groupes 2022 → +0%
    "Denmark":       (0, 1),   # R16 2018, R16 2022       → +2%
    "Poland":        (0, 1),   # groupes 2018, R16 2022   → +1%
    "Ecuador":       (0, 0),
    "Mexico":        (1, 0),   # R16 2018, groupes 2022   → +1%
    "Colombia":      (0, 0),
    "Norway":        (0, 0),   # non qualifiée les 2      → +0%
    "Austria":       (0, 0),
    "Turkey":        (0, 0),
    "Canada":        (0, 0),   # première qualification depuis 1986
    "Iran":          (0, 0),
    "Tunisia":       (0, 0),
    "Saudi Arabia":  (0, 0),
}

def get_wc_experience_bonus(team: str) -> float:
    """
    Calcule le bonus d'expérience WC sur 2018 + 2022.
    Favorise les équipes régulièrement performantes sur les deux tournois.
    """
    pts = WC_PERFORMANCE.get(team, (0, 0))
    total = pts[0] + pts[1]
    return 1.0 + total / 100.0

WC_EXPERIENCE_BONUS = {team: get_wc_experience_bonus(team) 
                        for team in WC_PERFORMANCE}

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
    # Compétitions majeures
    "FIFA World Cup":                       4.0,
    "UEFA Euro":                            3.5,  # ← v4: augmenté car niveau > autres tournois continentaux
    "Copa América":                         3.0,
    "African Cup of Nations":               2.0,  # ← v4: réduit (niveau < Euro)
    "AFC Asian Cup":                        2.0,  # ← v4: réduit
    "Gold Cup":                             1.8,  # ← v4: réduit

    # Qualifications WC — très représentatif
    "FIFA World Cup qualification":         3.0,

    # Qualifications continentales
    "UEFA Euro qualification":              2.0,  # ← v4: augmenté (niveau élevé)
    "African Cup of Nations qualification": 1.5,
    "AFC Asian Cup qualification":          1.5,
    "CONCACAF Nations League":              1.5,

    # Nations League
    "UEFA Nations League":                  2.2,  # ← v4: augmenté (très compétitif)

    # Amicaux
    "Friendly":                             0.2,
}

DEFAULT_WEIGHT = 1.0


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

        # Initialisation — combinaison FIFA + bookmakers + expérience WC
        att  = {}
        deff = {}
        for t in self.teams_:
            fifa_s    = get_fifa_score(t)
            book_prob = BOOKMAKER_PROBS.get(t, 0.005)
            # Conversion proba bookmaker en force relative (0.5 à 1.5)
            book_s    = 0.5 + min(book_prob / 0.18, 1.0)  # normalise sur France (17.4%)
            # Coefficient de confédération
            conf      = CONFEDERATION.get(t, "other")
            conf_b    = CONFEDERATION_BOOST.get(conf, 1.0)
            # Expérience WC
            exp_b     = get_wc_experience_bonus(t)
            # Force initiale = moyenne pondérée FIFA + bookmakers
            base      = 0.25 * fifa_s + 0.75 * book_s  # bookmakers encore plus valorisés
            att[t]    = base * conf_b * exp_b
            deff[t]   = (2.0 - base) * conf_b

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

            # Correction progressive vers FIFA + bookmakers
            corr_weight = max(0.0, 0.25 - iteration * 0.005)
            for team in self.teams_:
                fifa_s    = get_fifa_score(team)
                book_prob = BOOKMAKER_PROBS.get(team, 0.005)
                book_s    = 0.5 + min(book_prob / 0.18, 1.0)
                conf_b    = CONFEDERATION_BOOST.get(CONFEDERATION.get(team, "other"), 1.0)
                target    = (0.25 * fifa_s + 0.75 * book_s) * conf_b  # bookmakers très valorisés
                att[team]  = (1 - corr_weight) * att[team]  + corr_weight * target
                deff[team] = (1 - corr_weight) * deff[team] + corr_weight * (2.0 - target)

            # Plafonnement
            for team in self.teams_:
                att[team]  = min(max(att[team],  0.3), 2.2)
                deff[team] = min(max(deff[team], 0.2), 1.8)

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
    print("  World Cup 2026 — Modèle de Poisson v4")
    print("  (UEFA boosté, bookmakers, confédérations, expérience WC)")
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
