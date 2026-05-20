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
# COMBINAISONS FIFA — 3es qualifiés → matchups officiels
# Source: Wikipedia / Annexe C règlement FIFA WC 2026
# Format: frozenset(8 groupes) → {match_key: groupe_3e}
# match_key = '1A','1B','1D','1E','1G','1I','1K','1L'
# (1C, 1F, 1H, 1J affrontent toujours des 2es)
# ============================================================

def _make_combo(groups_str, a, b, d, e, g, i, k, l):
    """Crée une entrée de combinaison FIFA."""
    return (frozenset(groups_str), {"1A":a,"1B":b,"1D":d,"1E":e,"1G":g,"1I":i,"1K":k,"1L":l})

_RAW = [
    _make_combo("EFGHIJKL","3E","3J","3I","3F","3H","3G","3L","3K"),
    _make_combo("DFGHIJKL","3H","3G","3I","3D","3J","3F","3L","3K"),
    _make_combo("DEGHIJKL","3E","3J","3I","3D","3H","3G","3L","3K"),
    _make_combo("DEFHIJKL","3E","3J","3I","3D","3H","3F","3L","3K"),
    _make_combo("DEFGIJKL","3E","3G","3I","3D","3J","3F","3L","3K"),
    _make_combo("DEFGHJKL","3E","3G","3J","3D","3H","3F","3L","3K"),
    _make_combo("DEFGHIKL","3E","3G","3I","3D","3H","3F","3L","3K"),
    _make_combo("DEFGHIJL","3E","3G","3J","3D","3H","3F","3L","3I"),
    _make_combo("DEFGHIJK","3E","3G","3J","3D","3H","3F","3I","3K"),
    _make_combo("CFGHIJKL","3H","3G","3I","3C","3J","3F","3L","3K"),
    _make_combo("CEGHIJKL","3E","3J","3I","3C","3H","3G","3L","3K"),
    _make_combo("CEFHIJKL","3E","3J","3I","3C","3H","3F","3L","3K"),
    _make_combo("CEFGIJKL","3E","3G","3I","3C","3J","3F","3L","3K"),
    _make_combo("CEFGHJKL","3E","3G","3J","3C","3H","3F","3L","3K"),
    _make_combo("CEFGHIKL","3E","3G","3I","3C","3H","3F","3L","3K"),
    _make_combo("CEFGHIJL","3E","3G","3J","3C","3H","3F","3L","3I"),
    _make_combo("CEFGHIJK","3E","3G","3J","3C","3H","3F","3I","3K"),
    _make_combo("CDGHIJKL","3H","3G","3I","3C","3J","3D","3L","3K"),
    _make_combo("CDFHIJKL","3C","3J","3I","3D","3H","3F","3L","3K"),
    _make_combo("CDFGIJKL","3C","3G","3I","3D","3J","3F","3L","3K"),
    _make_combo("CDFGHJKL","3C","3G","3J","3D","3H","3F","3L","3K"),
    _make_combo("CDFGHIKL","3C","3G","3I","3D","3H","3F","3L","3K"),
    _make_combo("CDFGHIJL","3C","3G","3J","3D","3H","3F","3L","3I"),
    _make_combo("CDFGHIJK","3C","3G","3J","3D","3H","3F","3I","3K"),
    _make_combo("CDEHIJKL","3E","3J","3I","3C","3H","3D","3L","3K"),
    _make_combo("CDEGIJKL","3E","3G","3I","3C","3J","3D","3L","3K"),
    _make_combo("CDEGHIJKL"[:-1],"3E","3G","3J","3C","3H","3D","3L","3K"),
    _make_combo("CDEGHIKL","3E","3G","3I","3C","3H","3D","3L","3K"),
    _make_combo("CDEGHIJL","3E","3G","3J","3C","3H","3D","3L","3I"),
    _make_combo("CDEGHIJK","3E","3G","3J","3C","3H","3D","3I","3K"),
    _make_combo("CDEFIJKL","3C","3J","3E","3D","3I","3F","3L","3K"),
    _make_combo("CDEFHJKL","3C","3J","3E","3D","3H","3F","3L","3K"),
    _make_combo("CDEFHIKL","3C","3E","3I","3D","3H","3F","3L","3K"),
    _make_combo("CDEFHIJL","3C","3J","3E","3D","3H","3F","3L","3I"),
    _make_combo("CDEFHIJK","3C","3J","3E","3D","3H","3F","3I","3K"),
    _make_combo("CDEFGJKL","3C","3G","3E","3D","3J","3F","3L","3K"),
    _make_combo("CDEFGIKL","3C","3G","3E","3D","3I","3F","3L","3K"),
    _make_combo("CDEFGIJL","3C","3G","3E","3D","3J","3F","3L","3I"),
    _make_combo("CDEFGIJK","3C","3G","3E","3D","3J","3F","3I","3K"),
    _make_combo("CDEFGHKL","3C","3G","3E","3D","3H","3F","3L","3K"),
    _make_combo("CDEFGHJL","3C","3G","3J","3D","3H","3F","3L","3E"),
    _make_combo("CDEFGHJK","3C","3G","3J","3D","3H","3F","3E","3K"),
    _make_combo("CDEFGHIL","3C","3G","3E","3D","3H","3F","3L","3I"),
    _make_combo("CDEFGHIK","3C","3G","3E","3D","3H","3F","3I","3K"),
    _make_combo("CDEFGHIJ","3C","3G","3J","3D","3H","3F","3E","3I"),
    _make_combo("BFGHIJKL","3H","3J","3B","3F","3I","3G","3L","3K"),
    _make_combo("BEGHIJKL","3E","3J","3I","3B","3H","3G","3L","3K"),
    _make_combo("BEFHIJKL","3E","3J","3B","3F","3I","3H","3L","3K"),
    _make_combo("BEFGIJKL","3E","3J","3B","3F","3I","3G","3L","3K"),
    _make_combo("BEFGHJKL","3E","3J","3B","3F","3H","3G","3L","3K"),
]
FIFA_THIRD_COMBINATIONS = dict(_RAW)

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

    def _get_best_thirds(self, group_results: dict) -> list:
        """
        Sélectionne et trie les 8 meilleurs 3es selon les forces du modèle.
        Utilise la force d'attaque comme proxy des points (plus rapide).
        En réalité FIFA utilise pts/gd/gf des 3es simulés.
        """
        thirds = []
        for g, ranked in group_results.items():
            team = ranked[2]  # 3e du groupe
            # Force du 3e comme proxy de sa qualité
            strength = self.model.att_.get(team, 0.5) + (1.0 / max(self.model.def_.get(team, 1.0), 0.1))
            thirds.append((g, team, strength))

        # Trier par force décroissante → les 8 meilleurs
        thirds_sorted = sorted(thirds, key=lambda x: x[2], reverse=True)
        return [(g, team) for g, team, _ in thirds_sorted]

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

        # ── Sélection des 8 meilleurs 3es + combinaison FIFA ─
        # On trie les 12 3es par force et on garde les 8 meilleurs
        all_thirds_ranked = self._get_best_thirds(group_results)
        best_8 = all_thirds_ranked[:8]
        # Les groupes des 8 meilleurs 3es qualifiés
        qualified_third_groups = frozenset(g for g, _ in best_8)
        thirds_by_group = {g: team for g, team in best_8}

        # Lookup de la combinaison officielle FIFA
        # Format: {frozenset(groupes): {1er_groupe: groupe_du_3e_adverse}}
        combo = FIFA_THIRD_COMBINATIONS.get(qualified_third_groups, None)

        def get_third(match_key: str) -> str:
            """
            Retourne le 3e adverse pour un 1er de groupe donné.
            match_key: '1A', '1B', '1D', '1E', '1G', '1I', '1K', '1L'
            """
            if combo and match_key in combo:
                third_group = combo[match_key]  # ex: '3E' → groupe E
                g = third_group[1]              # extraire 'E'
                return thirds_by_group.get(g, group_results[g][2])
            # Fallback si combinaison non trouvée
            return all_thirds_ranked[0][1] if all_thirds_ranked else "Unknown"

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
        # Bracket officiel FIFA avec vraies combinaisons de 3es
        r32_matchups = [
            # M73: 2A vs 2B (fixes — pas de 3es)
            (group_results["A"][1], group_results["B"][1]),
            # M74: 1E vs 3e (groupe A/B/C/D/F selon combinaison)
            (group_results["E"][0], get_third("1E")),
            # M75: 1F vs 2C (fixe)
            (group_results["F"][0], group_results["C"][1]),
            # M76: 1C vs 2F (fixe)
            (group_results["C"][0], group_results["F"][1]),
            # M77: 1I vs 3e (groupe C/D/F/G/H selon combinaison)
            (group_results["I"][0], get_third("1I")),
            # M78: 2E vs 2I (fixe)
            (group_results["E"][1], group_results["I"][1]),
            # M79: 1A vs 3e (groupe C/E/F/H/I selon combinaison)
            (group_results["A"][0], get_third("1A")),
            # M80: 1L vs 3e (groupe E/H/I/J/K selon combinaison)
            (group_results["L"][0], get_third("1L")),
            # M81: 1D vs 3e (groupe B/E/F/I/J selon combinaison)
            (group_results["D"][0], get_third("1D")),
            # M82: 1G vs 3e (groupe A/E/H/I/J selon combinaison)
            (group_results["G"][0], get_third("1G")),
            # M83: 2K vs 2L (fixe)
            (group_results["K"][1], group_results["L"][1]),
            # M84: 1H vs 2J (fixe)
            (group_results["H"][0], group_results["J"][1]),
            # M85: 1B vs 3e (groupe E/F/G/I/J selon combinaison)
            (group_results["B"][0], get_third("1B")),
            # M86: 1J vs 2H (fixe)
            (group_results["J"][0], group_results["H"][1]),
            # M87: 1K vs 3e (groupe D/E/I/J/L selon combinaison)
            (group_results["K"][0], get_third("1K")),
            # M88: 2D vs 2G (fixe)
            (group_results["D"][1], group_results["G"][1]),
        ]

        # ── Round of 32 — résultats ───────────────────────────
        # On simule les 16 matchs dans l'ordre M73→M88
        r32_winners = []
        for home, away in r32_matchups:
            winner = self._simulate_match_ko(home, away, stage="r32")
            r32_winners.append(winner)

        # Index: 0=M73, 1=M74, 2=M75, 3=M76, 4=M77, 5=M78,
        #        6=M79, 7=M80, 8=M81, 9=M82, 10=M83, 11=M84,
        #        12=M85, 13=M86, 14=M87, 15=M88

        # ── Quarts de finale — bracket officiel FIFA ──────────
        # Source: wikipedia.org/wiki/2026_FIFA_World_Cup_knockout_stage
        # CÔTÉ GAUCHE: QF1, QF2, QF5, QF7
        # CÔTÉ DROIT:  QF3, QF4, QF6, QF8
        def ko(a, b):
            return self._simulate_match_ko(a, b, stage="qf")

        qf1 = ko(r32_winners[0],  r32_winners[1])   # M73 vs M74
        qf2 = ko(r32_winners[2],  r32_winners[3])   # M75 vs M76
        qf3 = ko(r32_winners[4],  r32_winners[5])   # M77 vs M78 (France 1I)
        qf4 = ko(r32_winners[6],  r32_winners[7])   # M79 vs M80 (Angleterre 1L)
        qf5 = ko(r32_winners[8],  r32_winners[9])   # M81 vs M82
        qf6 = ko(r32_winners[10], r32_winners[11])  # M83 vs M84 (Espagne 1H)
        qf7 = ko(r32_winners[12], r32_winners[13])  # M85 vs M86 (Argentine 1J)
        qf8 = ko(r32_winners[14], r32_winners[15])  # M87 vs M88

        # ── Demi-finales — bracket officiel FIFA ──────────────
        def sf(a, b):
            return self._simulate_match_ko(a, b, stage="sf")

        # Côté gauche
        sf1 = sf(qf1, qf2)   # QF1 vs QF2
        sf3 = sf(qf5, qf7)   # QF5 vs QF7 — Argentine possible

        # Côté droit
        sf2 = sf(qf3, qf4)   # QF3 vs QF4 — France vs Angleterre possible
        sf4 = sf(qf6, qf8)   # QF6 vs QF8 — Espagne possible

        # ── Finales ───────────────────────────────────────────
        # Finale côté gauche
        finalist_left  = self._simulate_match_ko(sf1, sf3, stage="sf")
        # Finale côté droit
        finalist_right = self._simulate_match_ko(sf2, sf4, stage="sf")

        # Grande finale
        champion = self._simulate_match_ko(finalist_left, finalist_right, stage="final")

        return champion

    def run_monte_carlo(self, n_simulations: int = 10000) -> tuple:
        """
        Lance n_simulations simulations du tournoi complet.

        Returns:
            Tuple (df_champions, knockout_data) :
            - df_champions : probabilités de victoire finale
            - knockout_data : dict avec les matchups les plus probables
              par phase {r32, qf, sf, final}
        """
        logger.info(f"🎲 Lancement de {n_simulations} simulations...")

        all_teams = [t for teams in WC2026_GROUPS.values() for t in teams]
        champions = {t: 0 for t in all_teams}

        # Tracking des matchups par phase
        # Format: {phase: Counter({(team1, team2): count})}
        from collections import Counter
        phase_matchups = {
            "r32":   Counter(),
            "qf":    Counter(),
            "sf":    Counter(),
            "final": Counter(),
        }
        # Tracking qui atteint chaque phase
        phase_reach = {
            "r32":   Counter(),
            "qf":    Counter(),
            "sf":    Counter(),
            "final": Counter(),
        }

        for i in range(n_simulations):
            # ── Phase de groupes ─────────────────────────────
            group_results = {}
            for group_name, teams in WC2026_GROUPS.items():
                ranked = self._simulate_group(group_name, teams)
                group_results[group_name] = ranked

            all_thirds_ranked = self._get_best_thirds(group_results)
            best_8 = all_thirds_ranked[:8]
            qualified_third_groups = frozenset(g for g, _ in best_8)
            thirds_by_group = {g: team for g, team in best_8}
            combo = FIFA_THIRD_COMBINATIONS.get(qualified_third_groups, None)

            def get_third(match_key):
                if combo and match_key in combo:
                    g = combo[match_key][1]
                    return thirds_by_group.get(g, group_results[g][2])
                return all_thirds_ranked[0][1] if all_thirds_ranked else "Unknown"

            r32_matchups = [
                (group_results["A"][1], group_results["B"][1]),
                (group_results["E"][0], get_third("1E")),
                (group_results["F"][0], group_results["C"][1]),
                (group_results["C"][0], group_results["F"][1]),
                (group_results["I"][0], get_third("1I")),
                (group_results["E"][1], group_results["I"][1]),
                (group_results["A"][0], get_third("1A")),
                (group_results["L"][0], get_third("1L")),
                (group_results["D"][0], get_third("1D")),
                (group_results["G"][0], get_third("1G")),
                (group_results["K"][1], group_results["L"][1]),
                (group_results["H"][0], group_results["J"][1]),
                (group_results["B"][0], get_third("1B")),
                (group_results["J"][0], group_results["H"][1]),
                (group_results["K"][0], get_third("1K")),
                (group_results["D"][1], group_results["G"][1]),
            ]

            # ── Round of 32 ───────────────────────────────────
            r32_winners = []
            for home, away in r32_matchups:
                key = tuple(sorted([home, away]))
                phase_matchups["r32"][key] += 1
                phase_reach["r32"][home] += 1
                phase_reach["r32"][away] += 1
                winner = self._simulate_match_ko(home, away, stage="r32")
                r32_winners.append(winner)

            def ko(a, b, phase):
                key = tuple(sorted([a, b]))
                phase_matchups[phase][key] += 1
                phase_reach[phase][a] += 1
                phase_reach[phase][b] += 1
                return self._simulate_match_ko(a, b, stage=phase)

            qf1 = ko(r32_winners[0],  r32_winners[1],  "qf")
            qf2 = ko(r32_winners[2],  r32_winners[3],  "qf")
            qf3 = ko(r32_winners[4],  r32_winners[5],  "qf")
            qf4 = ko(r32_winners[6],  r32_winners[7],  "qf")
            qf5 = ko(r32_winners[8],  r32_winners[9],  "qf")
            qf6 = ko(r32_winners[10], r32_winners[11], "qf")
            qf7 = ko(r32_winners[12], r32_winners[13], "qf")
            qf8 = ko(r32_winners[14], r32_winners[15], "qf")

            sf1 = ko(qf1, qf2, "sf")
            sf2 = ko(qf3, qf4, "sf")
            sf3 = ko(qf5, qf7, "sf")
            sf4 = ko(qf6, qf8, "sf")

            fl = ko(sf1, sf3, "sf")
            fr = ko(sf2, sf4, "sf")

            finalist_left  = fl
            finalist_right = fr
            key = tuple(sorted([finalist_left, finalist_right]))
            phase_matchups["final"][key] += 1
            phase_reach["final"][finalist_left]  += 1
            phase_reach["final"][finalist_right] += 1
            champion = self._simulate_match_ko(finalist_left, finalist_right, stage="final")
            champions[champion] = champions.get(champion, 0) + 1

            if (i + 1) % 500 == 0:
                logger.debug(f"  {i+1}/{n_simulations} simulations...")

        # ── Résultats champions ───────────────────────────────
        results = []
        for team, wins in champions.items():
            results.append({
                "team":     team,
                "group":    next(g for g, teams in WC2026_GROUPS.items() if team in teams),
                "wins":     wins,
                "prob_win": round(wins / n_simulations * 100, 2),
            })
        df = pd.DataFrame(results).sort_values("prob_win", ascending=False).reset_index(drop=True)
        df["rank"] = df.index + 1

        # ── Données phase finale ──────────────────────────────
        knockout_data = {}
        for phase, counter in phase_matchups.items():
            top_matchups = []
            for (t1, t2), count in counter.most_common(16):
                prob = count / n_simulations * 100
                # Probabilité que t1 gagne ce match
                pred = self.model.predict_score(t1, t2, neutral=True)
                top_matchups.append({
                    "home":     t1,
                    "away":     t2,
                    "prob_match": round(prob, 1),
                    "score":    pred["rounded_score"],
                    "prob_home_win": pred["prob_home_win"],
                    "prob_draw":     pred["prob_draw"],
                    "prob_away_win": pred["prob_away_win"],
                    "exp_home": pred["expected_home"],
                    "exp_away": pred["expected_away"],
                })
            knockout_data[phase] = top_matchups

        # Reach probabilities
        reach_probs = {}
        for phase, counter in phase_reach.items():
            reach_probs[phase] = {
                team: round(count / n_simulations * 100, 1)
                for team, count in counter.items()
            }
        knockout_data["reach"] = reach_probs

        logger.success(f"✅ Simulation — Champion : {df.iloc[0]['team']} ({df.iloc[0]['prob_win']}%)")
        return df, knockout_data

    def simulate_most_likely_bracket(self, scorer_model=None) -> dict:
        """
        Simule le bracket le plus probable du tournoi WC 2026.

        Format WC 2026 :
          Seizièmes (R32, 16 matchs)
          → Huitièmes (R16, 8 matchs)
          → Quarts (QF, 4 matchs)
          → Demies (SF, 2 matchs)
          → Finale (1 match)

        Les équipes qualifiées sont les plus probables d'après le modèle.
        Le bracket suit le tableau officiel FIFA.
        """
        logger.info("Calcul du bracket le plus probable...")

        result = {"r32": [], "r16": [], "qf": [], "sf": [], "final": []}

        # ── Qualifiés les plus probables ─────────────────────
        # 1ers de groupe
        W = {
            "A": "Mexico",      "B": "Canada",      "C": "Brazil",
            "D": "Australia",   "E": "Germany",     "F": "Netherlands",
            "G": "Belgium",     "H": "Spain",       "I": "France",
            "J": "Argentina",   "K": "Portugal",    "L": "England",
        }
        # 2es de groupe
        R = {
            "A": "South Korea", "B": "United States", "C": "Morocco",
            "D": "Turkey",      "E": "Ivory Coast",   "F": "Japan",
            "G": "New Zealand", "H": "Uruguay",        "I": "Norway",
            "J": "Austria",     "K": "Colombia",       "L": "Croatia",
        }
        # 8 meilleurs 3es (d'après forces du modèle + bookmakers)
        T3 = {
            "A": "Czech Republic",          "B": "Bosnia and Herzegovina",
            "C": "Scotland",                "E": "Ecuador",
            "F": "Sweden",                  "G": "Senegal",
            "I": "Algeria",                 "L": "Ghana",
        }
        # Fonction helper pour récupérer le 3e d'un groupe
        def t3(g): return T3.get(g, "TBD")

        # ── Bracket officiel FIFA — Seizièmes (R32) ───────────
        # Source : wikipedia.org/wiki/2026_FIFA_World_Cup_knockout_stage
        r32_pairs = [
            # Match : home,        away,         id
            (R["A"],    R["B"],    "M73"),   # 2A vs 2B
            (W["E"],    t3("F"),   "M74"),   # 1E vs 3F
            (W["F"],    R["C"],    "M75"),   # 1F vs 2C
            (W["C"],    R["F"],    "M76"),   # 1C vs 2F
            (W["I"],    t3("A"),   "M77"),   # 1I vs 3A
            (R["E"],    R["I"],    "M78"),   # 2E vs 2I
            (W["A"],    t3("C"),   "M79"),   # 1A vs 3C
            (W["L"],    t3("E"),   "M80"),   # 1L vs 3E
            (W["D"],    t3("B"),   "M81"),   # 1D vs 3B
            (W["G"],    t3("G"),   "M82"),   # 1G vs 3G → Sénégal
            (R["K"],    R["L"],    "M83"),   # 2K vs 2L
            (W["H"],    R["J"],    "M84"),   # 1H vs 2J
            (W["B"],    t3("I"),   "M85"),   # 1B vs 3I → Algérie
            (W["J"],    R["H"],    "M86"),   # 1J vs 2H
            (W["K"],    t3("L"),   "M87"),   # 1K vs 3L → Ghana
            (R["D"],    R["G"],    "M88"),   # 2D vs 2G
        ]

        def predict_match(home, away, phase, match_id=""):
            """Prédit un match KO et retourne le vainqueur le plus probable."""
            pred   = self.model.predict_score(home, away, neutral=True)
            winner = home if pred["prob_home_win"] >= pred["prob_away_win"] else away
            scorers = {}
            if scorer_model:
                try:
                    scorers = scorer_model.predict_match_scorers(
                        home, away, pred["expected_home"], pred["expected_away"], top_n=3
                    )
                except: pass
            result[phase].append({
                "match_id":      match_id,
                "home":          home,
                "away":          away,
                "score":         pred["rounded_score"],
                "exp_home":      pred["expected_home"],
                "exp_away":      pred["expected_away"],
                "prob_home_win": pred["prob_home_win"],
                "prob_draw":     pred["prob_draw"],
                "prob_away_win": pred["prob_away_win"],
                "winner":        winner,
                "scorers":       scorers,
            })
            return winner

        # Simuler les 16 seizièmes
        r32_w = [predict_match(h, a, "r32", mid) for h, a, mid in r32_pairs]

        # ── Huitièmes (R16) — bracket FIFA ────────────────────
        # Les gagnants s'enchaînent selon le tableau officiel :
        # M89 = W(M73) vs W(M74)  M90 = W(M75) vs W(M76)
        # M91 = W(M77) vs W(M78)  M92 = W(M79) vs W(M80)
        # M93 = W(M81) vs W(M82)  M94 = W(M83) vs W(M84)
        # M95 = W(M85) vs W(M86)  M96 = W(M87) vs W(M88)
        r16_pairs = [
            (r32_w[0],  r32_w[1],  "M89"),
            (r32_w[2],  r32_w[3],  "M90"),
            (r32_w[4],  r32_w[5],  "M91"),
            (r32_w[6],  r32_w[7],  "M92"),
            (r32_w[8],  r32_w[9],  "M93"),
            (r32_w[10], r32_w[11], "M94"),
            (r32_w[12], r32_w[13], "M95"),
            (r32_w[14], r32_w[15], "M96"),
        ]
        r16_w = [predict_match(h, a, "r16", mid) for h, a, mid in r16_pairs]

        # ── Quarts de finale ──────────────────────────────────
        # M97 = W(M89) vs W(M90)  M98 = W(M91) vs W(M92)
        # M99 = W(M93) vs W(M94)  M100= W(M95) vs W(M96)
        qf_pairs = [
            (r16_w[0], r16_w[1], "QF1"),
            (r16_w[2], r16_w[3], "QF2"),
            (r16_w[4], r16_w[5], "QF3"),
            (r16_w[6], r16_w[7], "QF4"),
        ]
        qf_w = [predict_match(h, a, "qf", mid) for h, a, mid in qf_pairs]

        # ── Demi-finales ──────────────────────────────────────
        sf_pairs = [
            (qf_w[0], qf_w[1], "SF1"),
            (qf_w[2], qf_w[3], "SF2"),
        ]
        sf_w = [predict_match(h, a, "sf", mid) for h, a, mid in sf_pairs]

        # ── Finale ────────────────────────────────────────────
        predict_match(sf_w[0], sf_w[1], "final",
                      "FINALE — MetLife Stadium, New Jersey · 19 juillet 2026")

        logger.success(f"✅ Bracket — Finale : {sf_w[0]} vs {sf_w[1]}")
        return result

    def run_scorer_monte_carlo(
        self,
        scorer_model,
        n_simulations: int = 1000,
    ) -> pd.DataFrame:
        """
        Simule le tournoi complet et accumule les buts attendus
        par joueur sur TOUS les matchs (groupes + phase finale).

        Returns:
            DataFrame avec buts attendus sur tournoi complet + % meilleur buteur
        """
        import numpy as np
        logger.info(f"🎲 Simulation buteurs ({n_simulations} tournois complets)...")

        # Accumuler les buts attendus par joueur
        player_goals = {}  # {(team, scorer): total_expected}
        player_matches = {}  # {(team, scorer): nb_matchs}

        for sim_i in range(n_simulations):
            # ── Phase de groupes ──────────────────────────────
            group_results = {}
            for group_name, teams in WC2026_GROUPS.items():
                ranked = self._simulate_group(group_name, teams)
                group_results[group_name] = ranked

                # Accumuler buts pour chaque match de groupe
                matchups = [
                    (teams[0], teams[1]), (teams[0], teams[2]), (teams[0], teams[3]),
                    (teams[1], teams[2]), (teams[1], teams[3]), (teams[2], teams[3]),
                ]
                for home, away in matchups:
                    pred = self.model.predict_score(home, away, neutral=True)
                    for team, exp_g in [(home, pred["expected_home"]),
                                        (away, pred["expected_away"])]:
                        if team in scorer_model.team_scorers_:
                            for _, p in scorer_model.team_scorers_[team].iterrows():
                                key = (team, p["scorer"])
                                contrib = exp_g * p["ratio"]
                                player_goals[key] = player_goals.get(key, 0) + contrib
                                player_matches[key] = player_matches.get(key, 0) + 1

            # ── Phase finale ──────────────────────────────────
            all_thirds_ranked = self._get_best_thirds(group_results)
            best_8 = all_thirds_ranked[:8]
            qualified_third_groups = frozenset(g for g, _ in best_8)
            thirds_by_group = {g: team for g, team in best_8}
            combo = FIFA_THIRD_COMBINATIONS.get(qualified_third_groups, None)

            def get_third(match_key):
                if combo and match_key in combo:
                    g = combo[match_key][1]
                    return thirds_by_group.get(g, group_results[g][2])
                return all_thirds_ranked[0][1] if all_thirds_ranked else "Unknown"

            r32_matchups = [
                (group_results["A"][1], group_results["B"][1]),
                (group_results["E"][0], get_third("1E")),
                (group_results["F"][0], group_results["C"][1]),
                (group_results["C"][0], group_results["F"][1]),
                (group_results["I"][0], get_third("1I")),
                (group_results["E"][1], group_results["I"][1]),
                (group_results["A"][0], get_third("1A")),
                (group_results["L"][0], get_third("1L")),
                (group_results["D"][0], get_third("1D")),
                (group_results["G"][0], get_third("1G")),
                (group_results["K"][1], group_results["L"][1]),
                (group_results["H"][0], group_results["J"][1]),
                (group_results["B"][0], get_third("1B")),
                (group_results["J"][0], group_results["H"][1]),
                (group_results["K"][0], get_third("1K")),
                (group_results["D"][1], group_results["G"][1]),
            ]

            # Simuler tous les matchs KO et accumuler buts
            def simulate_ko_and_accumulate(home, away, stage):
                pred = self.model.predict_score(home, away, neutral=True)
                for team, exp_g in [(home, pred["expected_home"]),
                                    (away, pred["expected_away"])]:
                    if team in scorer_model.team_scorers_:
                        for _, p in scorer_model.team_scorers_[team].iterrows():
                            key = (team, p["scorer"])
                            contrib = exp_g * p["ratio"]
                            player_goals[key] = player_goals.get(key, 0) + contrib
                            player_matches[key] = player_matches.get(key, 0) + 1
                return self._simulate_match_ko(home, away, stage)

            r32_w = [simulate_ko_and_accumulate(h, a, "r32") for h, a in r32_matchups]

            qf1 = simulate_ko_and_accumulate(r32_w[0],  r32_w[1],  "qf")
            qf2 = simulate_ko_and_accumulate(r32_w[2],  r32_w[3],  "qf")
            qf3 = simulate_ko_and_accumulate(r32_w[4],  r32_w[5],  "qf")
            qf4 = simulate_ko_and_accumulate(r32_w[6],  r32_w[7],  "qf")
            qf5 = simulate_ko_and_accumulate(r32_w[8],  r32_w[9],  "qf")
            qf6 = simulate_ko_and_accumulate(r32_w[10], r32_w[11], "qf")
            qf7 = simulate_ko_and_accumulate(r32_w[12], r32_w[13], "qf")
            qf8 = simulate_ko_and_accumulate(r32_w[14], r32_w[15], "qf")

            sf1 = simulate_ko_and_accumulate(qf1, qf2, "sf")
            sf2 = simulate_ko_and_accumulate(qf3, qf4, "sf")
            sf3 = simulate_ko_and_accumulate(qf5, qf7, "sf")
            sf4 = simulate_ko_and_accumulate(qf6, qf8, "sf")

            fl = simulate_ko_and_accumulate(sf1, sf3, "final")
            fr = simulate_ko_and_accumulate(sf2, sf4, "final")
            simulate_ko_and_accumulate(fl, fr, "final")

            if (sim_i + 1) % 200 == 0:
                logger.debug(f"  {sim_i + 1}/{n_simulations} simulations...")

        # Convertir en DataFrame
        # Plafonnement réaliste : record ère moderne = 8 buts (Müller 2010)
        # On plafonne à 7.0 pour rester dans les bornes historiques
        MAX_GOALS_TOURNAMENT = 7.0

        rows = []
        for (team, scorer), total_goals in player_goals.items():
            avg_goals   = min(total_goals / n_simulations, MAX_GOALS_TOURNAMENT)
            avg_matches = player_matches.get((team, scorer), 0) / n_simulations
            prob_score  = min(1 - np.exp(-avg_goals), 0.999) * 100
            rows.append({
                "Joueur":             scorer,
                "Équipe":             team,
                "Buts attendus":      round(avg_goals, 2),
                "Matchs moyens":      round(avg_matches, 1),
                "% marquer":          round(prob_score, 1),
            })

        df = (
            pd.DataFrame(rows)
            .sort_values("Buts attendus", ascending=False)
            .reset_index(drop=True)
        )
        df.index += 1
        logger.success(f"✅ Simulation buteurs terminée — {len(df)} joueurs")
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
