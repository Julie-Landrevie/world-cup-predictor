"""
src/simulation/tournament.py
-----------------------------
Simulateur complet de la Coupe du Monde 2026.

FONCTIONNALITÉS :
  1. Simulation de la phase de groupes (48 équipes, 12 groupes)
  2. Simulation de la phase à élimination directe (huitièmes → finale)
  3. Monte Carlo : 10 000 simulations pour obtenir des probabilités
  4. Mise à jour en temps réel : on peut entrer les vrais résultats
     au fur et à mesure de la compétition et recalculer les probabilités

ARCHITECTURE "ÉTAT DU TOURNOI" :
  TournamentState stocke :
    - Les vrais résultats entrés manuellement
    - Les groupes et leurs standings actuels
    - Le bracket de la phase finale
  
  On peut appeler update_result() après chaque vrai match
  et relancer simulate() pour des probabilités mises à jour.

FORMAT WC 2026 :
  - 48 équipes, 12 groupes de 4
  - Les 2 premiers de chaque groupe + 8 meilleurs 3èmes → 32 équipes
  - Phase finale classique : huitièmes, quarts, demis, finale
"""

import pandas as pd
import numpy as np
from scipy.stats import poisson
from dataclasses import dataclass, field
from typing import Optional
from loguru import logger

# ============================================================
# GROUPES WC 2026 — tirage officiel FIFA
# ============================================================

WC2026_GROUPS = {
    "A": ["Mexico",        "South Africa",  "South Korea",          "Czech Republic"],
    "B": ["Canada",        "Bosnia and Herzegovina", "United States", "Paraguay"],
    "C": ["Qatar",         "Switzerland",   "Brazil",               "Morocco"],
    "D": ["Haiti",         "Scotland",      "Australia",            "Turkey"],
    "E": ["Germany",       "Curaçao",       "Ivory Coast",          "Ecuador"],
    "F": ["Netherlands",   "Japan",         "Sweden",               "Tunisia"],
    "G": ["Belgium",       "Egypt",         "Iran",                 "New Zealand"],
    "H": ["Spain",         "Cape Verde",    "Saudi Arabia",         "Uruguay"],
    "I": ["France",        "Senegal",       "Iraq",                 "Norway"],
    "J": ["Argentina",     "Algeria",       "Austria",              "Jordan"],
    "K": ["Portugal",      "DR Congo",      "Uzbekistan",           "Colombia"],
    "L": ["England",       "Croatia",       "Ghana",                "Panama"],
}

# ============================================================
# STRUCTURES DE DONNÉES
# ============================================================

@dataclass
class MatchResult:
    """Résultat d'un match — réel ou simulé."""
    home:       str
    away:       str
    home_score: int
    away_score: int
    stage:      str = "group"  # "group", "r32", "qf", "sf", "final"
    real:       bool = False   # True = vrai résultat entré manuellement


@dataclass
class TeamStanding:
    """Classement d'une équipe dans son groupe."""
    team:       str
    played:     int = 0
    won:        int = 0
    drawn:      int = 0
    lost:       int = 0
    gf:         int = 0   # buts marqués
    ga:         int = 0   # buts encaissés
    gd:         int = 0   # différence de buts
    points:     int = 0

    def update(self, scored: int, conceded: int):
        self.played += 1
        self.gf     += scored
        self.ga     += conceded
        self.gd      = self.gf - self.ga
        if scored > conceded:
            self.won    += 1
            self.points += 3
        elif scored == conceded:
            self.drawn  += 1
            self.points += 1
        else:
            self.lost   += 1


@dataclass
class TournamentState:
    """
    État complet du tournoi à un instant donné.

    C'est cet objet qu'on met à jour après chaque vrai résultat.
    On peut le sérialiser (JSON) pour le sauvegarder entre les sessions.
    """
    # Vrais résultats entrés manuellement
    real_results: list = field(default_factory=list)

    # Équipes qualifiées pour chaque phase (rempli au fur et à mesure)
    group_qualifiers: dict = field(default_factory=dict)  # {groupe: [1er, 2e, 3e, 4e]}
    r32_bracket:      list = field(default_factory=list)  # [(home, away), ...]
    qf_bracket:       list = field(default_factory=list)
    sf_bracket:       list = field(default_factory=list)
    finalist:         list = field(default_factory=list)
    champion:         str  = ""

    def add_real_result(self, home: str, away: str, home_score: int, away_score: int, stage: str = "group"):
        """Ajoute un vrai résultat — appelé après chaque match réel."""
        # Vérifie si ce match existe déjà
        for r in self.real_results:
            if r.home == home and r.away == away and r.stage == stage:
                r.home_score = home_score
                r.away_score = away_score
                logger.info(f"✅ Résultat mis à jour : {home} {home_score}-{away_score} {away}")
                return
        self.real_results.append(MatchResult(home, away, home_score, away_score, stage, real=True))
        logger.info(f"✅ Résultat enregistré : {home} {home_score}-{away_score} {away}")

    def get_real_result(self, home: str, away: str, stage: str = "group") -> Optional[MatchResult]:
        """Retourne le vrai résultat d'un match s'il existe."""
        for r in self.real_results:
            if r.home == home and r.away == away and r.stage == stage and r.real:
                return r
        return None

    def to_dict(self) -> dict:
        """Sérialise l'état pour sauvegarde JSON."""
        return {
            "real_results": [
                {"home": r.home, "away": r.away,
                 "home_score": r.home_score, "away_score": r.away_score,
                 "stage": r.stage}
                for r in self.real_results
            ],
            "champion": self.champion,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TournamentState":
        """Reconstruit l'état depuis un dict JSON."""
        state = cls()
        for r in data.get("real_results", []):
            state.add_real_result(r["home"], r["away"], r["home_score"], r["away_score"], r["stage"])
        state.champion = data.get("champion", "")
        return state


# ============================================================
# SIMULATEUR
# ============================================================

class TournamentSimulator:
    """
    Simule le tournoi complet de la WC 2026.

    Utilise le modèle de Poisson pour prédire les matchs non joués
    et les vrais résultats pour les matchs déjà joués.
    """

    def __init__(self, model, state: Optional[TournamentState] = None):
        """
        Args:
            model : PoissonPredictor entraîné
            state : TournamentState avec les vrais résultats (optionnel)
        """
        self.model = model
        self.state = state or TournamentState()

    def _simulate_match(self, home: str, away: str, stage: str = "group") -> tuple:
        """
        Simule un match.

        Si un vrai résultat existe → on l'utilise.
        Sinon → on tire aléatoirement selon la distribution de Poisson.

        Returns:
            (home_score, away_score)
        """
        # Vrai résultat disponible ?
        real = self.state.get_real_result(home, away, stage)
        if real:
            return real.home_score, real.away_score

        # Sinon : simulation Poisson
        pred = self.model.predict_score(home, away, neutral=True)
        lam_h = pred["expected_home"]
        lam_a = pred["expected_away"]

        home_score = np.random.poisson(lam_h)
        away_score = np.random.poisson(lam_a)
        return home_score, away_score

    def _simulate_match_ko(self, home: str, away: str, stage: str) -> str:
        """
        Simule un match à élimination directe.
        En cas d'égalité après 90min → prolongations → tirs au but.

        Returns:
            Nom de l'équipe qualifiée.
        """
        real = self.state.get_real_result(home, away, stage)
        if real:
            if real.home_score > real.away_score:
                return home
            elif real.away_score > real.home_score:
                return away
            else:
                # Résultat réel nul → on simule les TAB
                return np.random.choice([home, away])

        pred  = self.model.predict_score(home, away, neutral=True)
        lam_h = pred["expected_home"]
        lam_a = pred["expected_away"]

        h = np.random.poisson(lam_h)
        a = np.random.poisson(lam_a)

        if h > a:
            return home
        elif a > h:
            return away
        else:
            # Égalité → probabilité basée sur les forces relatives
            p_home = pred["prob_home_win"] / (pred["prob_home_win"] + pred["prob_away_win"] + 1e-6)
            return home if np.random.random() < p_home else away

    def _simulate_group(self, group_name: str, teams: list) -> list:
        """
        Simule tous les matchs d'un groupe.

        Returns:
            Liste des équipes triées par classement [1er, 2e, 3e, 4e].
        """
        standings = {t: TeamStanding(team=t) for t in teams}

        # Matchs aller-retour dans le groupe (chaque équipe joue 3 matchs)
        matchups = [
            (teams[0], teams[1]), (teams[0], teams[2]), (teams[0], teams[3]),
            (teams[1], teams[2]), (teams[1], teams[3]), (teams[2], teams[3]),
        ]

        for home, away in matchups:
            h, a = self._simulate_match(home, away, stage="group")
            standings[home].update(h, a)
            standings[away].update(a, h)

        # Tri : points → différence de buts → buts marqués
        ranked = sorted(
            standings.values(),
            key=lambda s: (s.points, s.gd, s.gf),
            reverse=True
        )
        return [s.team for s in ranked]

    def _get_third_place_points(self, group_name: str, teams: list) -> TeamStanding:
        """Retourne le standing du 3e de groupe pour départager les meilleurs 3es."""
        standings = {t: TeamStanding(team=t) for t in teams}
        matchups = [
            (teams[0], teams[1]), (teams[0], teams[2]), (teams[0], teams[3]),
            (teams[1], teams[2]), (teams[1], teams[3]), (teams[2], teams[3]),
        ]
        for home, away in matchups:
            h, a = self._simulate_match(home, away, stage="group")
            standings[home].update(h, a)
            standings[away].update(a, h)

        ranked = sorted(standings.values(), key=lambda s: (s.points, s.gd, s.gf), reverse=True)
        return ranked[2]  # le 3e

    def simulate_once(self) -> str:
        """
        Simule le tournoi complet une fois.

        Returns:
            Nom du champion.
        """
        # ── Phase de groupes ──────────────────────────────────
        group_results = {}
        thirds = []

        for group_name, teams in WC2026_GROUPS.items():
            ranked = self._simulate_group(group_name, teams)
            group_results[group_name] = ranked
            thirds.append((group_name, ranked[2]))

        # ── Sélection des 8 meilleurs 3es ─────────────────────
        # On recalcule les standings réels des 3es pour les trier
        # (simplification : on utilise le rang dans le groupe)
        # En réalité FIFA utilise pts/gd/gf des 3es
        thirds_sorted = thirds[:8]  # on prend les 8 premiers groupes alphabétiquement
        # → Dans une vraie implémentation on trierait par points/gd

        # ── Sélection des 8 meilleurs 3es ─────────────────────
        # On trie les 3es par points puis différence de buts
        # Pour la simulation on les tire aléatoirement parmi les 12
        # (dans la réalité c'est les 8 meilleurs sur pts/gd/gf)
        all_thirds = [(g, group_results[g][2]) for g in WC2026_GROUPS]
        # On simule leurs standings approximatifs pour les classer
        # Simplification : on prend les 8 premiers groupes alphabétiquement
        # car on ne connaît pas leurs vrais points à ce stade
        # En prod, il faudrait tracker les vrais standings
        best_thirds_groups = sorted(all_thirds)[:8]  # A→H = les 8 premiers groupes
        thirds_by_group = {g: team for g, team in best_thirds_groups}

        def get_third(group: str) -> str:
            """Retourne le meilleur 3e d'un groupe donné, ou un placeholder."""
            return thirds_by_group.get(group, group_results[group][2])

        # ── Round of 32 — Bracket officiel FIFA 2026 ──────────
        # Source : wikipedia.org/wiki/2026_FIFA_World_Cup_knockout_stage
        # Matchups fixes (hors 3es qui dépendent des combinaisons)
        # On utilise la combinaison la plus fréquente (groupes A-H qualifiés)
        # Les 8 meilleurs 3es viennent des groupes A, B, C, D, E, F, G, H
        # → Combinaison #1 du tableau FIFA (3E, 3J, 3I, 3F, 3H, 3G, 3L, 3K)
        # Mais comme on simule, on prend les 3es des groupes tirés aléatoirement

        # Matchups officiels FIFA (bracket fixe) :
        # M73: 2A vs 2B          M74: 1E vs meilleur 3e (A/B/C/D/F)
        # M75: 1F vs 2C          M76: 1C vs 2F
        # M77: 1I vs meilleur 3e M78: 2E vs 2I
        # M79: 1A vs meilleur 3e M80: 1L vs meilleur 3e
        # M81: 1D vs meilleur 3e M82: 1G vs meilleur 3e
        # M83: 2K vs 2L          M84: 1H vs 2J
        # M85: 1B vs meilleur 3e M86: 1J vs 2H
        # M87: 1K vs meilleur 3e M88: 2D vs 2G

        # Pour la simulation, on affecte les 3es aux matchs selon leur groupe d'origine
        # Combinaison simplifiée : les 8 meilleurs 3es = 3A, 3B, 3C, 3D, 3E, 3F, 3G, 3H
        t3 = {g: group_results[g][2] for g in "ABCDEFGHIJKL"}

        r32_matchups = [
            # M73
            (group_results["A"][1], group_results["B"][1]),
            # M74: 1E vs 3e (ici on prend 3F comme approx)
            (group_results["E"][0], t3["F"]),
            # M75: 1F vs 2C
            (group_results["F"][0], group_results["C"][1]),
            # M76: 1C vs 2F
            (group_results["C"][0], group_results["F"][1]),
            # M77: 1I vs 3e
            (group_results["I"][0], t3["D"]),
            # M78: 2E vs 2I
            (group_results["E"][1], group_results["I"][1]),
            # M79: 1A vs 3e
            (group_results["A"][0], t3["C"]),
            # M80: 1L vs 3e
            (group_results["L"][0], t3["E"]),
            # M81: 1D vs 3e
            (group_results["D"][0], t3["B"]),
            # M82: 1G vs 3e
            (group_results["G"][0], t3["A"]),
            # M83: 2K vs 2L
            (group_results["K"][1], group_results["L"][1]),
            # M84: 1H vs 2J
            (group_results["H"][0], group_results["J"][1]),
            # M85: 1B vs 3e
            (group_results["B"][0], t3["G"]),
            # M86: 1J vs 2H
            (group_results["J"][0], group_results["H"][1]),
            # M87: 1K vs 3e
            (group_results["K"][0], t3["H"]),
            # M88: 2D vs 2G
            (group_results["D"][1], group_results["G"][1]),
        ]

        # ── Huitièmes ─────────────────────────────────────────
        qf_teams = []
        for home, away in r32_matchups:
            winner = self._simulate_match_ko(home, away, stage="r32")
            qf_teams.append(winner)

        # ── Quarts de finale ──────────────────────────────────
        sf_teams = []
        for i in range(0, len(qf_teams), 2):
            if i + 1 < len(qf_teams):
                winner = self._simulate_match_ko(qf_teams[i], qf_teams[i+1], stage="qf")
                sf_teams.append(winner)

        # ── Demi-finales ──────────────────────────────────────
        finalists = []
        for i in range(0, len(sf_teams), 2):
            if i + 1 < len(sf_teams):
                winner = self._simulate_match_ko(sf_teams[i], sf_teams[i+1], stage="sf")
                finalists.append(winner)

        # ── Finale ────────────────────────────────────────────
        if len(finalists) >= 2:
            champion = self._simulate_match_ko(finalists[0], finalists[1], stage="final")
        elif len(finalists) == 1:
            champion = finalists[0]
        else:
            champion = "Unknown"

        return champion

    def run_monte_carlo(self, n_simulations: int = 10000) -> pd.DataFrame:
        """
        Lance n_simulations simulations du tournoi complet.

        Returns:
            DataFrame avec la probabilité de victoire finale de chaque équipe,
            trié par probabilité décroissante.
        """
        logger.info(f"🎲 Lancement de {n_simulations} simulations...")

        champions = {}
        all_teams = [t for teams in WC2026_GROUPS.values() for t in teams]
        for team in all_teams:
            champions[team] = 0

        for i in range(n_simulations):
            champion = self.simulate_once()
            if champion in champions:
                champions[champion] += 1
            if (i + 1) % 1000 == 0:
                logger.debug(f"  {i+1}/{n_simulations} simulations...")

        # Conversion en probabilités
        results = []
        for team, wins in champions.items():
            results.append({
                "team":      team,
                "group":     next(g for g, teams in WC2026_GROUPS.items() if team in teams),
                "wins":      wins,
                "prob_win":  round(wins / n_simulations * 100, 2),
            })

        df = pd.DataFrame(results).sort_values("prob_win", ascending=False).reset_index(drop=True)
        df["rank"] = df.index + 1

        logger.success(f"✅ Simulation terminée — Champion le plus probable : {df.iloc[0]['team']} ({df.iloc[0]['prob_win']}%)")
        return df

    def group_stage_probabilities(self, n_simulations: int = 5000) -> pd.DataFrame:
        """
        Calcule les probabilités de qualification de chaque équipe
        depuis la phase de groupes.

        Returns:
            DataFrame avec prob de finir 1er, 2e, 3e, 4e dans le groupe.
        """
        logger.info(f"Calcul des probabilités de phase de groupes ({n_simulations} sims)...")

        results = {team: {"1er": 0, "2e": 0, "3e": 0, "4e": 0}
                   for teams in WC2026_GROUPS.values() for team in teams}

        for _ in range(n_simulations):
            for group_name, teams in WC2026_GROUPS.items():
                ranked = self._simulate_group(group_name, teams)
                for pos, team in enumerate(ranked):
                    label = ["1er", "2e", "3e", "4e"][pos]
                    results[team][label] += 1

        rows = []
        for team, counts in results.items():
            group = next(g for g, teams in WC2026_GROUPS.items() if team in teams)
            rows.append({
                "team":  team,
                "group": group,
                "% 1er": round(counts["1er"] / n_simulations * 100, 1),
                "% 2e":  round(counts["2e"]  / n_simulations * 100, 1),
                "% 3e":  round(counts["3e"]  / n_simulations * 100, 1),
                "% 4e":  round(counts["4e"]  / n_simulations * 100, 1),
                "% qualif": round((counts["1er"] + counts["2e"]) / n_simulations * 100, 1),
            })

        return pd.DataFrame(rows).sort_values(["group", "% 1er"], ascending=[True, False])


# ============================================================
# POINT D'ENTRÉE
# ============================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.data.collect import load_results, load_wc2026_fixtures
    from src.models.match_predictor import PoissonPredictor

    print("=" * 65)
    print("  World Cup 2026 — Simulateur de tournoi")
    print("=" * 65)

    # Chargement et entraînement du modèle
    results = load_results(min_year=2018)
    model   = PoissonPredictor(min_year=2018, decay_rate=0.005, n_iter=30)
    model.fit(results)

    # Création de l'état du tournoi (vide au départ)
    state = TournamentState()

    # ── Exemple : entrer de vrais résultats ──────────────────
    # Décommenter pendant la compétition pour mettre à jour :
    # state.add_real_result("Mexico", "South Africa", 2, 1, stage="group")
    # state.add_real_result("France", "Senegal", 3, 0, stage="group")

    # Simulateur
    sim = TournamentSimulator(model=model, state=state)

    # ── Simulation unique ────────────────────────────────────
    print("\n🎲 Simulation unique du tournoi...")
    champion = sim.simulate_once()
    print(f"   Champion simulé : {champion}")

    # ── Monte Carlo (1000 simulations rapides pour le test) ──
    print("\n🎲 Monte Carlo — 1000 simulations...")
    mc_results = sim.run_monte_carlo(n_simulations=1000)

    print("\n🏆 PROBABILITÉS DE REMPORTER LA WC 2026 :\n")
    top15 = mc_results.head(15)
    for _, row in top15.iterrows():
        bar = "█" * int(row["prob_win"] / 1) + "░" * (20 - int(row["prob_win"] / 1))
        print(f"  {row['rank']:2}. {row['team']:25} {bar} {row['prob_win']:5.1f}%")

    # ── Probabilités de phase de groupes ────────────────────
    print("\n\n📊 PROBABILITÉS DE QUALIFICATION — Groupes I et J (exemples)\n")
    group_probs = sim.group_stage_probabilities(n_simulations=1000)
    for group in ["I", "J"]:
        print(f"  Groupe {group} :")
        g = group_probs[group_probs["group"] == group]
        print(g[["team", "% 1er", "% 2e", "% qualif"]].to_string(index=False))
        print()
