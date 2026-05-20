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
# ============================================================
# JOUEURS EXCLUS — absents des listes officielles WC 2026
# Sources : listes officielles publiées mai 2026
# ============================================================
EXCLUDED_PLAYERS = {
    # ── Retraités internationaux ──────────────────────────────
    "Karim Benzema",       # retraité
    "Antoine Griezmann",   # retraité international (annonce mai 2026)
    "Angel Di María",      # retraité international
    "Ángel Di María",      # variante accent
    "Olivier Giroud",      # retraité international
    "Álvaro Morata",       # retraité international Euro 2024
    "Memphis Depay",       # retraité international
    "Ivan Perišić",        # retraité international
    "Edin Džeko",          # retraité international
    "Islam Slimani",       # retraité international
    "Gareth Bale",         # retraité
    "Marko Arnautović",    # retraité international
    "Sergio Agüero",       # retraité
    "Luis Suárez",         # retraité international
    "Giorgio Chiellini",   # retraité
    "Thomas Müller",       # retraité international
    "Manuel Neuer",        # retraité international

    # ── Non qualifiés / pays absents ──────────────────────────
    "Robert Lewandowski",  # Pologne non qualifiée
    "Mohamed Salah",       # Égypte — groupe G, présent mais peu de buts attendus
    "Sadio Mané",          # Sénégal présent — garder !

    # ── Blessés / non convoqués ───────────────────────────────
    "Raheem Sterling",     # non convoqué Angleterre
    "Marcus Rashford",     # non convoqué Angleterre (Tuchel)
    # Neymar convoqué par le Brésil (18 mai 2026) — NE PAS EXCLURE
    "Hugo Ekitike",        # blessé tendon Achille, remplacé par Mateta
    "Rodrygo",             # absent liste Brésil — blessé (surprise Ancelotti)
    "Estêvão",             # absent liste Brésil
    "João Pedro",          # absent liste Brésil (Chelsea)
    "Thiago Silva",        # non retenu Brésil
    "Diogo Jota",          # décédé le 3 juillet 2025 — RIP 🙏

    # ── Doublons dataset (variantes orthographiques) ──────────
    "Julián Alvarez",      # doublon de "Julián Álvarez" (sans accent)
    "Julian Alvarez",      # variante sans accent
}

# ── Joueurs présents malgré doutes (à conserver) ─────────────
# Sadio Mané : convoqué Sénégal ✅
# Cristiano Ronaldo : convoqué Portugal ✅
# Romelu Lukaku : convoqué Belgique ✅
# Harry Kane : convoqué Angleterre ✅

# ============================================================
# LISTES OFFICIELLES — joueurs autorisés par équipe
# Si une équipe est listée ici, SEULS ces joueurs apparaissent.
# Les équipes non listées utilisent le filtrage EXCLUDED_PLAYERS.
# Sources : listes officielles annoncées avant le 18 mai 2026
# ============================================================

OFFICIAL_SQUADS = {
    # ── France — liste officielle Deschamps (14 mai 2026) ────
    "France": {
        # Gardiens
        "Mike Maignan", "Brice Samba", "Robin Risser",
        # Défenseurs
        "Lucas Digne", "Malo Gusto", "Lucas Hernandez", "Théo Hernandez",
        "Ibrahima Konaté", "Jules Koundé", "Maxence Lacroix",
        "William Saliba", "Dayot Upamecano",
        # Milieux
        "N'Golo Kanté", "Aurélien Tchouaméni", "Warren Zaïre-Emery",
        "Manu Koné", "Adrien Rabiot",
        # Attaquants
        "Maghnes Akliouche", "Rayan Cherki", "Ousmane Dembélé",
        "Désiré Doué", "Bradley Barcola", "Jean-Philippe Mateta",
        "Kylian Mbappé", "Michael Olise", "Marcus Thuram",
    },

    # ── Brésil — liste officielle Ancelotti (18 mai 2026) ───
    # Rodrygo absent (blessé), Neymar de retour, Estêvão absent
    "Brazil": {
        # Gardiens
        "Alisson", "Ederson", "Weverton",
        # Défenseurs
        "Marquinhos", "Gabriel", "Bremer", "Ibañez",
        "Leo Pereira", "Wesley", "Danilo", "Alex Sandro", "Douglas Santos",
        # Milieux
        "Casemiro", "Bruno Guimarães", "Fabinho", "Lucas Paquetá",
        # Attaquants
        "Vinícius Júnior", "Raphinha", "Neymar", "Matheus Cunha",
        "Luiz Henrique", "Igor Thiago", "Endrick", "Martinelli",
        "Rayan", "Richarlison",
    },

    # ── Croatie — liste officielle (18 mai 2026) ─────────────
    "Croatia": {
        "Dominik Livaković", "Ivica Ivušić", "Lovre Kalinić",
        "Josip Šutalo", "Josip Stanišić", "Martin Erlić", "Dario Spikić",
        "Borna Sosa", "Joško Gvardiol", "Ivan Perišić",
        "Luka Modrić", "Mateo Kovačić", "Marcelo Brozović",
        "Mario Pašalić", "Lovro Majer", "Luka Sučić",
        "Andrej Kramarić", "Marko Pjaca", "Bruno Petković",
        "Ivan Perišić", "Petar Sucić",
    },

    # ── Autriche — liste officielle (18 mai 2026) ────────────
    "Austria": {
        "Patrick Pentz", "Daniel Bachmann", "Tobias Lawal",
        "Stefan Posch", "Maximilian Wöber", "Kevin Danso",
        "Phillipp Mwene", "Christoph Baumgartner",
        "Konrad Laimer", "Nicolas Seiwald", "Florian Kainz",
        "Marcel Sabitzer", "Patrick Wimmer", "Romano Schmid",
        "Michael Gregoritsch", "Marko Arnautović",
        "Andreas Weimann", "Guido Burgstaller",
    },
    # Autres équipes à ajouter au fur et à mesure des annonces officielles
    # Portugal (19 mai), Espagne (25 mai), Angleterre (22 mai), Allemagne (21 mai)
}

# ============================================================
# AJUSTEMENTS MANUELS — stars dont les données sont incomplètes
# Ces multiplicateurs corrigent les sous-estimations du dataset
# ============================================================
# ============================================================
# AJUSTEMENTS MANUELS — basés sur les listes officielles WC 2026
# et la forme récente des joueurs (mai 2026)
# ============================================================
PLAYER_BOOST = {
    # ── France (liste Deschamps 14 mai 2026) ─────────────────
    "Kylian Mbappé":        1.15,  # capitaine, stars des Bleus
    "Marcus Thuram":        1.10,  # titulaire indiscutable, prolifique à l'Inter
    "Ousmane Dembélé":      1.05,  # bon début de saison PSG
    "Bradley Barcola":      1.10,  # nouvelle star, sous-représenté dataset
    "Adrien Rabiot":        0.45,  # milieu défensif — rarement buteur
    "Aurélien Tchouaméni":  0.40,  # milieu défensif
    "Manu Koné":            0.40,  # milieu défensif
    "Michael Olise":        1.10,  # en grande forme Bayern
    "Rayan Cherki":         1.05,  # nouvelle star, peu de capes
    "Maghnes Akliouche":    1.05,  # surprenant mais présent
    "Désiré Doué":          1.00,
    "Jean-Philippe Mateta": 1.05,  # remplace Ekitike blessé

    # ── Espagne (liste attendue ~25 mai) ─────────────────────
    "Ferran Torres":        0.90,  # souvent remplaçant
    "Mikel Oyarzabal":      1.05,
    "Dani Olmo":            1.15,  # en très grande forme
    "Lamine Yamal":         1.25,  # star mondiale, très sous-représenté dataset
    "Nico Williams":        1.15,  # idem, peu de données
    "Álvaro Morata":        0.00,  # retraité — sera filtré par EXCLUDED
    "Mikel Merino":         1.05,

    # ── Argentine (liste préliminaire 55 joueurs, 11 mai) ────
    "Lionel Messi":         1.05,  # 6e Coupe du monde
    "Lautaro Martínez":     1.10,  # grand buteur Serie A
    "Julián Álvarez":       1.15,  # en très grande forme Atlético
    "Alejandro Garnacho":   1.10,  # nouvelle star, peu de données

    # ── Angleterre (liste Tuchel attendue 22 mai) ────────────
    "Harry Kane":           1.05,  # capitaine et buteur numéro 1
    "Bukayo Saka":          1.10,  # titulaire indiscutable
    "Phil Foden":           1.10,  # sous-représenté dataset
    "Cole Palmer":          1.20,  # révélation saison, peu de capes
    "Jude Bellingham":      1.15,  # stars mais peu de buts internationaux

    # ── Portugal (liste attendue 19 mai) ─────────────────────
    "Cristiano Ronaldo":    0.80,  # 41 ans, rôle réduit
    "Bruno Fernandes":      1.10,  # capitaine, très prolifique
    "Gonçalo Ramos":        1.20,  # meilleur buteur équipe nationale récent
    "Rafael Leão":          1.10,
    "Pedro Neto":           1.05,

    # ── Brésil (liste attendue 18 mai) ───────────────────────
    "Vinícius Júnior":      1.25,  # meilleur joueur monde, sous-représenté
    "Neymar":              0.75,  # retour après 2.5 ans absence — incertitude forme
    "Martinelli":          1.10,  # en grande forme Arsenal
    "Matheus Cunha":       1.05,  # en grande forme Manchester United
    "Raphinha":             1.10,  # capitaine, en grande forme Barça
    "Rodrygo":              1.10,
    "Richarlison":          1.00,
    "Endrick":              1.10,  # nouvelle star, peu de capes

    # ── Allemagne ─────────────────────────────────────────────
    "Kai Havertz":          1.10,
    "Florian Wirtz":        1.25,  # meilleur joueur Bundesliga, peu de données
    "Jamal Musiala":        1.15,  # star allemande
    "Leroy Sané":           1.00,

    # ── Pays-Bas ──────────────────────────────────────────────
    "Cody Gakpo":           1.15,
    "Xavi Simons":          1.10,
    "Donyell Malen":        1.05,

    # ── Norvège ───────────────────────────────────────────────
    "Erling Haaland":       1.10,  # meilleur buteur monde
    "Alexander Sørloth":    1.05,

    # ── Autres favoris ────────────────────────────────────────
    "Pedri":                1.10,  # Espagne, milieu créateur
    "Rodri":                1.05,  # Espagne, milieu défensif
    "Romelu Lukaku":        1.05,  # Belgique, toujours présent
    "Luis Díaz":            1.10,  # Colombia, très en forme Liverpool
    "Federico Valverde":    1.05,  # Argentine, milieu
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

            # Filtrage selon liste officielle si disponible, sinon exclusions générales
            if team in OFFICIAL_SQUADS:
                # On garde UNIQUEMENT les joueurs de la liste officielle
                player_stats = player_stats[
                    player_stats["scorer"].isin(OFFICIAL_SQUADS[team])
                ]
            else:
                # On exclut les joueurs confirmés absents
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
                "Buts attendus WC":    round(exp_goals, 2),
                "% meilleur buteur":   round(min(1 - np.exp(-exp_goals), 0.999) * 100, 1),
            })

        df_result = (
            pd.DataFrame(rows)
            .sort_values("Buts attendus WC", ascending=False)
            .reset_index(drop=True)
        )
        df_result.index += 1

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
    cols = [c for c in ["Joueur","Équipe","Buts attendus WC","% meilleur buteur"] if c in top_scorers.columns]
    print(top_scorers[cols].to_string())

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
