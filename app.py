"""
app.py — World Cup 2026 Predictor
Interface Streamlit complète
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.data.collect import load_results, load_goalscorers, load_wc2026_fixtures
from src.models.match_predictor import PoissonPredictor, BOOKMAKER_PROBS
from src.models.scorer_predictor import ScorerPredictor
from src.simulation.tournament import TournamentSimulator, TournamentState, WC2026_GROUPS

# ============================================================
# CONFIG PAGE
# ============================================================

st.set_page_config(
    page_title="World Cup 2026 Predictor",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

:root {
    --gold:    #F5C842;
    --red:     #E8334A;
    --dark:    #0A0E1A;
    --darker:  #060910;
    --card:    #111827;
    --border:  #1F2937;
    --text:    #F9FAFB;
    --muted:   #6B7280;
    --green:   #10B981;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background: var(--darker);
    color: var(--text);
}

/* Hero */
.hero {
    background: linear-gradient(135deg, #0A0E1A 0%, #1a0a2e 50%, #0A0E1A 100%);
    border-bottom: 1px solid var(--gold);
    padding: 40px 32px 32px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: "2026";
    position: absolute;
    right: -20px;
    top: -20px;
    font-family: 'Bebas Neue', cursive;
    font-size: 200px;
    color: rgba(245,200,66,0.04);
    pointer-events: none;
    line-height: 1;
}
.hero-title {
    font-family: 'Bebas Neue', cursive;
    font-size: 3.5rem;
    color: var(--gold);
    letter-spacing: 3px;
    line-height: 1;
    margin: 0;
}
.hero-sub {
    font-size: 0.9rem;
    color: var(--muted);
    margin-top: 8px;
    letter-spacing: 1px;
    text-transform: uppercase;
}

/* Cards */
.stat-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
}
.stat-value {
    font-family: 'Bebas Neue', cursive;
    font-size: 2.4rem;
    color: var(--gold);
    line-height: 1;
}
.stat-label {
    font-size: 0.72rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 4px;
}

/* Section title */
.section-title {
    font-family: 'Bebas Neue', cursive;
    font-size: 1.4rem;
    color: var(--gold);
    letter-spacing: 2px;
    border-left: 3px solid var(--red);
    padding-left: 12px;
    margin: 28px 0 16px;
}

/* Match card */
.match-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

/* Probability bar */
.prob-bar-container {
    display: flex;
    height: 6px;
    border-radius: 3px;
    overflow: hidden;
    margin: 8px 0;
}
.prob-home { background: var(--green); }
.prob-draw { background: var(--muted); }
.prob-away { background: var(--red); }

/* Team badge */
.team-flag {
    font-size: 1.8rem;
    line-height: 1;
}

/* Group table */
.group-header {
    font-family: 'Bebas Neue', cursive;
    font-size: 1.1rem;
    color: var(--gold);
    letter-spacing: 2px;
    padding: 8px 0 4px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 8px;
}

/* Winner badge */
.winner-badge {
    background: linear-gradient(135deg, #F5C842, #E8334A);
    color: #000;
    font-family: 'Bebas Neue', cursive;
    font-size: 0.85rem;
    letter-spacing: 1px;
    padding: 3px 10px;
    border-radius: 99px;
    display: inline-block;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: var(--card);
    border-bottom: 1px solid var(--border);
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Bebas Neue', cursive;
    letter-spacing: 1.5px;
    font-size: 1rem;
    color: var(--muted) !important;
    border-radius: 0;
    padding: 12px 20px;
}
.stTabs [aria-selected="true"] {
    color: var(--gold) !important;
    border-bottom: 2px solid var(--gold) !important;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    background: var(--card) !important;
}

/* Sidebar */
section[data-testid="stSidebar"] { background: var(--darker); }

/* Metric */
div[data-testid="metric-container"] {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px;
}
div[data-testid="metric-container"] label {
    color: var(--muted) !important;
    font-size: 0.72rem !important;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: var(--gold) !important;
    font-family: 'Bebas Neue', cursive !important;
    font-size: 1.8rem !important;
}

/* Hide Streamlit chrome */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
.stDeployButton { display: none; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# FLAGS — drapeaux emoji par pays
# ============================================================
FLAGS = {
    "France": "🇫🇷", "Spain": "🇪🇸", "England": "🏴󠁧󠁢󠁥󠁮󠁧󠁿",
    "Argentina": "🇦🇷", "Brazil": "🇧🇷", "Germany": "🇩🇪",
    "Portugal": "🇵🇹", "Netherlands": "🇳🇱", "Belgium": "🇧🇪",
    "Croatia": "🇭🇷", "Morocco": "🇲🇦", "Japan": "🇯🇵",
    "South Korea": "🇰🇷", "United States": "🇺🇸", "Mexico": "🇲🇽",
    "Canada": "🇨🇦", "Switzerland": "🇨🇭", "Norway": "🇳🇴",
    "Senegal": "🇸🇳", "Colombia": "🇨🇴", "Uruguay": "🇺🇾",
    "Ecuador": "🇪🇨", "Australia": "🇦🇺", "Turkey": "🇹🇷",
    "Denmark": "🇩🇰", "Poland": "🇵🇱", "Austria": "🇦🇹",
    "Algeria": "🇩🇿", "Tunisia": "🇹🇳", "Egypt": "🇪🇬",
    "Saudi Arabia": "🇸🇦", "Iran": "🇮🇷", "Qatar": "🇶🇦",
    "Scotland": "🏴󠁧󠁢󠁳󠁣󠁴󠁿", "Ghana": "🇬🇭", "Ivory Coast": "🇨🇮",
    "South Africa": "🇿🇦", "DR Congo": "🇨🇩", "Cameroon": "🇨🇲",
    "Paraguay": "🇵🇾", "Panama": "🇵🇦", "Costa Rica": "🇨🇷",
    "Haiti": "🇭🇹", "Curaçao": "🇨🇼", "Jamaica": "🇯🇲",
    "Sweden": "🇸🇪", "New Zealand": "🇳🇿", "Bosnia and Herzegovina": "🇧🇦",
    "Jordan": "🇯🇴", "Iraq": "🇮🇶", "Uzbekistan": "🇺🇿",
    "Cape Verde": "🇨🇻", "Serbia": "🇷🇸", "Czech Republic": "🇨🇿",
}

def flag(team):
    return FLAGS.get(team, "🏴")


# ============================================================
# CHARGEMENT DES DONNÉES (mis en cache)
# ============================================================

@st.cache_resource
def load_models():
    """Charge et entraîne tous les modèles — mis en cache."""
    results     = load_results(min_year=2018)
    goalscorers = load_goalscorers(min_year=2020)
    fixtures    = load_wc2026_fixtures()

    match_model = PoissonPredictor(min_year=2018, decay_rate=0.005, n_iter=30)
    match_model.fit(results)

    scorer_model = ScorerPredictor(min_year=2020)
    scorer_model.fit(goalscorers)

    all_preds = match_model.predict_all_fixtures(fixtures)

    return match_model, scorer_model, fixtures, all_preds


@st.cache_data
def get_tournament_probs(n_sims=2000):
    """Monte Carlo — probabilités de remporter le tournoi + données phase finale."""
    match_model, _, _, _ = load_models()
    state = TournamentState()
    sim   = TournamentSimulator(model=match_model, state=state)
    df, ko_data = sim.run_monte_carlo(n_simulations=n_sims)
    return df, ko_data


# ============================================================
# HERO
# ============================================================

st.markdown("""
<div class="hero">
    <div class="hero-title">🏆 World Cup 2026 Predictor</div>
    <div class="hero-sub">Modèle de Poisson · Bookmakers · Monte Carlo · USA · Canada · Mexico</div>
</div>
""", unsafe_allow_html=True)

# Chargement
with st.spinner("Entraînement du modèle de Poisson..."):
    try:
        match_model, scorer_model, fixtures, all_preds = load_models()
        data_ok = True
    except Exception as e:
        st.error(f"❌ Erreur : {e}")
        data_ok = False

if not data_ok:
    st.stop()

# Métriques globales
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Équipes", "48")
c2.metric("Matchs prédits", len(all_preds))
c3.metric("Données historiques", "7 952 matchs")
c4.metric("Buteurs analysés", "1 079 joueurs")
c5.metric("Début du tournoi", "11 juin 2026")

st.divider()


# ============================================================
# ONGLETS
# ============================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏆 Pronostics tournoi",
    "⚽ Matchs de groupe",
    "🥇 Buteurs",
    "👥 Groupes",
    "🔍 Match par match",
    "🏟️ Phase finale",
])


# ──────────────────────────────────────────────────────────
# ONGLET 1 — PRONOSTICS TOURNOI
# ──────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-title">Probabilités de remporter la Coupe du Monde</div>', unsafe_allow_html=True)
    st.caption("Basé sur 2 000 simulations Monte Carlo — modèle Poisson + classement FIFA + cotes bookmakers (Polymarket/Kalshi, ~1Md$)")

    col_mc, col_book = st.columns([3, 2], gap="large")

    with col_mc:
        st.markdown("**📊 Notre modèle**")
        with st.spinner("Simulation en cours (2000 tournois)..."):
            mc, ko_data = get_tournament_probs(n_sims=2000)

        top20 = mc.head(20)
        for _, row in top20.iterrows():
            team  = row["team"]
            prob  = row["prob_win"]
            book  = BOOKMAKER_PROBS.get(team, 0.003) * 100
            bar_w = int(prob / 30 * 100)

            st.markdown(f"""
            <div style="display:flex; align-items:center; gap:12px; margin-bottom:10px;
                        background:#111827; border:1px solid #1F2937; border-radius:8px; padding:10px 14px;">
                <div style="font-size:1.5rem; width:36px">{flag(team)}</div>
                <div style="flex:1">
                    <div style="font-weight:600; font-size:0.9rem">{team}</div>
                    <div style="height:5px; background:#1F2937; border-radius:3px; margin-top:5px;">
                        <div style="height:5px; width:{min(bar_w,100)}%; background:linear-gradient(90deg,#F5C842,#E8334A); border-radius:3px;"></div>
                    </div>
                </div>
                <div style="text-align:right; min-width:80px">
                    <div style="font-family:'Bebas Neue',cursive; font-size:1.3rem; color:#F5C842">{prob:.1f}%</div>
                    <div style="font-size:0.68rem; color:#6B7280">book: {book:.1f}%</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_book:
        st.markdown("**📈 Cotes bookmakers (Polymarket)**")
        st.caption("Source : marchés de prédiction, ~1 milliard $ de volume")

        book_teams = [
            ("France", 17.4), ("Spain", 16.5), ("England", 11.3),
            ("Argentina", 10.5), ("Brazil", 9.5), ("Germany", 6.5),
            ("Portugal", 5.5), ("Netherlands", 4.0), ("Morocco", 2.0),
            ("Colombia", 1.4), ("Uruguay", 1.2), ("United States", 1.6),
            ("Switzerland", 1.2), ("Croatia", 1.2), ("Norway", 1.0),
        ]
        for team, prob in book_teams:
            st.markdown(f"""
            <div style="display:flex; align-items:center; gap:10px; margin-bottom:8px;">
                <div style="width:24px; text-align:center">{flag(team)}</div>
                <div style="flex:1; font-size:0.85rem">{team}</div>
                <div style="font-family:'Bebas Neue',cursive; color:#F5C842; font-size:1.1rem">{prob:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div style="background:#111827; border:1px solid #1F2937; border-radius:8px;
                    padding:12px; margin-top:16px; font-size:0.75rem; color:#6B7280;">
        ⚠️ <b>Points d'attention modèle</b><br>
        Les probabilités du modèle peuvent différer des bookmakers car elles
        reflètent uniquement les performances statistiques récentes et le bracket FIFA.
        Les bookmakers intègrent aussi des facteurs humains (blessures, moral, etc.).
        </div>
        """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────
# ONGLET 2 — MATCHS DE GROUPE
# ──────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-title">Prédictions — Phase de groupes</div>', unsafe_allow_html=True)

    # Filtre par groupe
    groups = sorted(WC2026_GROUPS.keys())
    selected_group = st.selectbox(
        "Filtrer par groupe",
        ["Tous les groupes"] + [f"Groupe {g}" for g in groups]
    )

    df_display = all_preds.copy()
    if selected_group != "Tous les groupes":
        g_letter = selected_group.split()[-1]
        group_teams = WC2026_GROUPS.get(g_letter, [])
        df_display = df_display[
            df_display["home"].isin(group_teams) |
            df_display["away"].isin(group_teams)
        ]

    st.caption(f"{len(df_display)} matchs affichés")

    for _, row in df_display.iterrows():
        ph = row["% victoire"]
        pd_ = row["% nul"]
        pa = row["% défaite"]
        bh = row["buts_home"]
        ba = row["buts_away"]

        st.markdown(f"""
        <div style="background:#111827; border:1px solid #1F2937; border-radius:10px;
                    padding:14px 18px; margin-bottom:8px;">
            <div style="display:flex; align-items:center; justify-content:space-between; gap:12px;">
                <div style="flex:2; text-align:right;">
                    <span style="font-size:1.3rem">{flag(row['home'])}</span>
                    <span style="font-weight:600; margin-left:8px">{row['home']}</span>
                </div>
                <div style="flex:1; text-align:center;">
                    <div style="font-family:'Bebas Neue',cursive; font-size:1.4rem; color:#F5C842">
                        {row['score_prédit']}
                    </div>
                    <div style="font-size:0.68rem; color:#6B7280">{row['date']}</div>
                </div>
                <div style="flex:2; text-align:left;">
                    <span style="font-weight:600; margin-right:8px">{row['away']}</span>
                    <span style="font-size:1.3rem">{flag(row['away'])}</span>
                </div>
            </div>
            <div style="display:flex; gap:4px; margin-top:10px; height:5px; border-radius:3px; overflow:hidden;">
                <div style="width:{ph}%; background:#10B981;"></div>
                <div style="width:{pd_}%; background:#4B5563;"></div>
                <div style="width:{pa}%; background:#E8334A;"></div>
            </div>
            <div style="display:flex; justify-content:space-between; margin-top:4px; font-size:0.72rem; color:#6B7280;">
                <span>✓ {ph:.0f}%</span>
                <span>= {pd_:.0f}%</span>
                <span>{pa:.0f}% ✓</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────
# ONGLET 3 — BUTEURS
# ──────────────────────────────────────────────────────────
with tab3:
    # ── Top buteurs tournoi complet (simulation Monte Carlo) ─
    st.markdown('<div class="section-title">🏆 Top buteurs — Tournoi complet (simulation)</div>', unsafe_allow_html=True)
    st.caption("Buts attendus sur l'ensemble du tournoi (groupes + phase finale) · 500 simulations Monte Carlo")

    @st.cache_data
    def get_full_scorer_rankings():
        match_model, scorer_model_c, _, _ = load_models()
        state = TournamentState()
        sim   = TournamentSimulator(model=match_model, state=state)
        return sim.run_scorer_monte_carlo(scorer_model_c, n_simulations=500)

    with st.spinner("Simulation des buteurs sur le tournoi complet..."):
        full_scorers = get_full_scorer_rankings()

    col_full, col_team = st.columns([3, 2], gap="large")

    with col_full:
        st.caption("Phase de groupes + huitièmes + quarts + demi-finales + finale")
        for i, (_, row) in enumerate(full_scorers.head(20).iterrows(), 1):
            team  = row["Équipe"]
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
            buts  = row["Buts attendus"]
            pct   = row["% marquer"]
            st.markdown(f"""
            <div style="display:flex; align-items:center; gap:12px; margin-bottom:8px;
                        background:#111827; border:1px solid #1F2937; border-radius:8px;
                        padding:10px 14px;">
                <div style="width:28px; text-align:center; font-size:0.9rem; color:#6B7280">{medal}</div>
                <div style="font-size:1.3rem">{flag(team)}</div>
                <div style="flex:1">
                    <div style="font-weight:600; font-size:0.9rem">{row['Joueur']}</div>
                    <div style="font-size:0.72rem; color:#6B7280">{team} · {row['Matchs moyens']:.1f} matchs en moyenne</div>
                </div>
                <div style="text-align:right">
                    <div style="font-family:'Bebas Neue',cursive; font-size:1.3rem; color:#F5C842">{buts:.2f}</div>
                    <div style="font-size:0.68rem; color:#6B7280">{pct:.0f}% de marquer</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_team:
        st.markdown('<div class="section-title">Par équipe</div>', unsafe_allow_html=True)
        st.caption("Buts attendus dans la compétition · % meilleur buteur de la sélection")

        _, scorer_model_t, fixtures_t, all_preds_t = load_models()
        team_options = sorted([t for t in WC2026_GROUPS.values() for t in t])
        selected_team = st.selectbox("Choisir une équipe", team_options)

        if selected_team:
            # Buts attendus sur les matchs de groupe pour cette équipe
            team_preds = all_preds_t[
                (all_preds_t["home"] == selected_team) |
                (all_preds_t["away"] == selected_team)
            ]
            total_exp = 0
            player_exp = {}
            for _, row in team_preds.iterrows():
                if row["home"] == selected_team:
                    exp_g = row["buts_home"]
                else:
                    exp_g = row["buts_away"]
                total_exp += exp_g
                if selected_team in scorer_model_t.team_scorers_:
                    for _, p in scorer_model_t.team_scorers_[selected_team].iterrows():
                        key = p["scorer"]
                        player_exp[key] = player_exp.get(key, 0) + exp_g * p["ratio"]

            st.markdown(f"**{flag(selected_team)} {selected_team}**")
            st.caption(f"Buts attendus phase de groupes : {total_exp:.2f}")

            if player_exp:
                sorted_players = sorted(player_exp.items(), key=lambda x: x[1], reverse=True)
                max_exp = sorted_players[0][1] if sorted_players else 1
                for joueur, exp_g in sorted_players[:8]:
                    prob_top = (exp_g / total_exp * 100) if total_exp > 0 else 0
                    bar_w = int(exp_g / max_exp * 100)
                    st.markdown(f"""
                    <div style="background:#111827; border:1px solid #1F2937; border-radius:6px;
                                padding:8px 12px; margin-bottom:6px;">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <div style="font-size:0.85rem; font-weight:500">{joueur}</div>
                            <div style="text-align:right">
                                <span style="font-family:'Bebas Neue',cursive; color:#F5C842">{exp_g:.2f} buts</span>
                                <span style="font-size:0.68rem; color:#6B7280; margin-left:6px">{prob_top:.0f}% top buteur</span>
                            </div>
                        </div>
                        <div style="height:3px; background:#1F2937; border-radius:2px; margin-top:6px;">
                            <div style="height:3px; width:{bar_w}%; background:linear-gradient(90deg,#F5C842,#E8334A); border-radius:2px;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────
# ONGLET 4 — GROUPES
# ──────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-title">Composition des 12 groupes</div>', unsafe_allow_html=True)
    st.caption("Probabilités de qualification calculées sur 1000 simulations")

    @st.cache_data
    def get_group_probs():
        match_model_g, _, _, _ = load_models()
        state_g = TournamentState()
        sim_g   = TournamentSimulator(model=match_model_g, state=state_g)
        return sim_g.group_stage_probabilities(n_simulations=1000)

    with st.spinner("Calcul des probabilités de groupe..."):
        group_probs = get_group_probs()

    # Affichage en grille 3 colonnes
    group_keys = list(WC2026_GROUPS.keys())
    for row_i in range(0, len(group_keys), 3):
        cols = st.columns(3, gap="medium")
        for col_i, g in enumerate(group_keys[row_i:row_i+3]):
            with cols[col_i]:
                st.markdown(f'<div class="group-header">GROUPE {g}</div>', unsafe_allow_html=True)
                gdf = group_probs[group_probs["group"] == g].sort_values("% qualif", ascending=False)
                for _, tr in gdf.iterrows():
                    team   = tr["team"]
                    qualif = tr["% qualif"]
                    first  = tr["% 1er"]
                    color  = "#F5C842" if qualif >= 60 else "#10B981" if qualif >= 40 else "#6B7280"
                    st.markdown(f"""
                    <div style="display:flex; align-items:center; gap:8px; margin-bottom:6px;
                                padding:6px 10px; background:#111827; border-radius:6px;
                                border-left:3px solid {color};">
                        <span>{flag(team)}</span>
                        <span style="flex:1; font-size:0.82rem">{team}</span>
                        <span style="font-size:0.75rem; color:#6B7280">{first:.0f}% 1er</span>
                        <span style="font-family:'Bebas Neue',cursive; color:{color}; font-size:0.95rem">{qualif:.0f}%</span>
                    </div>
                    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────
# ONGLET 5 — MATCH PAR MATCH
# ──────────────────────────────────────────────────────────
with tab5:
    st.markdown('<div class="section-title">Analyse détaillée — Match par match</div>', unsafe_allow_html=True)

    all_teams = sorted(set(
        t for teams in WC2026_GROUPS.values() for t in teams
    ))

    col_h, col_vs, col_a = st.columns([2, 1, 2])
    with col_h:
        home = st.selectbox("Équipe domicile", all_teams, index=all_teams.index("France"))
    with col_vs:
        st.markdown("<br><div style='text-align:center; font-family:Bebas Neue; font-size:1.5rem; color:#F5C842'>VS</div>", unsafe_allow_html=True)
    with col_a:
        away = st.selectbox("Équipe extérieur", all_teams, index=all_teams.index("Argentina"))

    if home != away:
        pred    = match_model.predict_score(home, away, neutral=True)
        scorers = scorer_model.predict_match_scorers(
            home, away, pred["expected_home"], pred["expected_away"], top_n=5
        )

        # Résumé du match
        st.markdown(f"""
        <div style="background:#111827; border:1px solid #F5C842; border-radius:12px;
                    padding:24px; margin:16px 0; text-align:center;">
            <div style="display:flex; align-items:center; justify-content:center; gap:24px;">
                <div>
                    <div style="font-size:2.5rem">{flag(home)}</div>
                    <div style="font-weight:700; margin-top:4px">{home}</div>
                    <div style="font-family:'Bebas Neue',cursive; font-size:1.5rem; color:#10B981">{pred['prob_home_win']}%</div>
                </div>
                <div>
                    <div style="font-family:'Bebas Neue',cursive; font-size:3rem; color:#F5C842">
                        {pred['rounded_score']}
                    </div>
                    <div style="font-size:0.75rem; color:#6B7280">score prédit</div>
                    <div style="font-size:0.8rem; color:#4B5563; margin-top:4px">
                        Nul : {pred['prob_draw']}%
                    </div>
                </div>
                <div>
                    <div style="font-size:2.5rem">{flag(away)}</div>
                    <div style="font-weight:700; margin-top:4px">{away}</div>
                    <div style="font-family:'Bebas Neue',cursive; font-size:1.5rem; color:#E8334A">{pred['prob_away_win']}%</div>
                </div>
            </div>
            <div style="margin-top:12px; font-size:0.78rem; color:#6B7280">
                Buts attendus : {pred['expected_home']:.2f} — {pred['expected_away']:.2f}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Matrice des scores
        st.markdown('<div class="section-title">Distribution des scores</div>', unsafe_allow_html=True)
        matrix = pred["score_matrix"]
        max_goals_show = 5
        matrix_data = {}
        for a in range(max_goals_show + 1):
            matrix_data[f"{away} {a}"] = [
                f"{matrix[h, a]*100:.1f}%" for h in range(max_goals_show + 1)
            ]
        df_matrix = pd.DataFrame(
            matrix_data,
            index=[f"{home} {h}" for h in range(max_goals_show + 1)]
        )
        st.dataframe(df_matrix, use_container_width=True)

        # Buteurs probables
        st.markdown('<div class="section-title">Buteurs probables</div>', unsafe_allow_html=True)
        col_sh, col_sa = st.columns(2, gap="large")

        for col, team in [(col_sh, home), (col_sa, away)]:
            with col:
                st.markdown(f"**{flag(team)} {team}**")
                team_scorers = scorers.get(team, pd.DataFrame())
                if not team_scorers.empty and "scorer" in team_scorers.columns:
                    for _, sr in team_scorers.iterrows():
                        pct = sr.get("% chance de marquer", 0)
                        st.markdown(f"""
                        <div style="display:flex; align-items:center; gap:10px; margin-bottom:6px;
                                    background:#111827; border:1px solid #1F2937; border-radius:6px;
                                    padding:8px 12px;">
                            <div style="flex:1; font-size:0.85rem">{sr['scorer']}</div>
                            <div style="height:4px; width:60px; background:#1F2937; border-radius:2px; overflow:hidden;">
                                <div style="height:4px; width:{min(pct*2,100):.0f}%; background:#F5C842;"></div>
                            </div>
                            <div style="font-family:'Bebas Neue',cursive; color:#F5C842; font-size:1rem; min-width:40px; text-align:right">{pct:.0f}%</div>
                        </div>
                        """, unsafe_allow_html=True)
    else:
        st.info("Sélectionne deux équipes différentes.")


# ──────────────────────────────────────────────────────────
# ONGLET 6 — PHASE FINALE
# ──────────────────────────────────────────────────────────
with tab6:
    st.markdown('<div class="section-title">🏟️ Phase finale — Scénario le plus probable</div>', unsafe_allow_html=True)
    st.caption("Simulation déterministe basée sur les forces du modèle · Les équipes les plus probables à chaque étape")

    st.markdown("""
    <div style="background:#111827; border:1px solid #F5C842; border-radius:8px;
                padding:12px 16px; margin-bottom:20px; font-size:0.78rem; color:#9CA3AF;">
    ⚠️ <b>Mise en garde</b> — Ces prédictions seront mises à jour au fur et à mesure des vrais résultats.
    Record de buts sur une Coupe du Monde : <b>13 buts</b> (Just Fontaine, 1958) ·
    Ère moderne : <b>8 buts</b> · Maximum possible : <b>7 matchs</b> par équipe (8 avec la phase de groupes).
    </div>
    """, unsafe_allow_html=True)

    @st.cache_data
    def get_most_likely_bracket():
        match_model_b, scorer_model_b, _, _ = load_models()
        state_b = TournamentState()
        sim_b   = TournamentSimulator(model=match_model_b, state=state_b)
        return sim_b.simulate_most_likely_bracket(scorer_model=scorer_model_b)

    with st.spinner("Calcul du bracket le plus probable..."):
        bracket = get_most_likely_bracket()

    PHASE_NAMES = {
        "r32":   ("⚔️ Seizièmes de finale — Round of 32", 4),
        "r16":   ("🔥 Huitièmes de finale — Round of 16", 4),
        "qf":    ("🏅 Quarts de finale",                  2),
        "sf":    ("🥈 Demi-finales",                       2),
        "final": ("🏆 Finale",                             1),
    }

    for phase, (phase_label, n_cols) in PHASE_NAMES.items():
        matches = bracket.get(phase, [])
        if not matches:
            continue
        if not matches:
            continue

        st.markdown(f'<div class="section-title">{phase_label}</div>', unsafe_allow_html=True)

        if n_cols > 1:
            cols = st.columns(n_cols, gap="medium")
        else:
            cols = [st.container()]

        for i, m in enumerate(matches):
            home = m["home"]
            away = m["away"]
            winner = m["winner"]
            ph = m["prob_home_win"]
            pd_ = m["prob_draw"]
            pa = m["prob_away_win"]
            score = m["score"]
            mid = m.get("match_id", "")

            winner_color_h = "#F5C842" if winner == home else "#6B7280"
            winner_color_a = "#F5C842" if winner == away else "#6B7280"

            col = cols[i % n_cols] if n_cols > 1 else cols[0]
            with col:
                st.markdown(f"""
                <div style="background:#111827; border:1px solid {'#F5C842' if phase=='final' else '#1F2937'};
                            border-radius:10px; padding:14px 16px; margin-bottom:10px;">
                    <div style="font-size:0.65rem; color:#6B7280; text-align:center; margin-bottom:8px;
                                text-transform:uppercase; letter-spacing:1px;">{mid}</div>
                    <div style="display:flex; align-items:center; justify-content:space-between; gap:8px;">
                        <div style="flex:2; text-align:right;">
                            <div style="font-size:1.2rem">{flag(home)}</div>
                            <div style="font-weight:600; font-size:0.85rem; color:{winner_color_h}">{home}</div>
                            <div style="font-family:'Bebas Neue',cursive; font-size:1.1rem; color:#10B981">{ph:.0f}%</div>
                        </div>
                        <div style="flex:1; text-align:center;">
                            <div style="font-family:'Bebas Neue',cursive; font-size:1.6rem; color:#F5C842">{score}</div>
                            <div style="font-size:0.65rem; color:#6B7280">Nul {pd_:.0f}%</div>
                        </div>
                        <div style="flex:2; text-align:left;">
                            <div style="font-size:1.2rem">{flag(away)}</div>
                            <div style="font-weight:600; font-size:0.85rem; color:{winner_color_a}">{away}</div>
                            <div style="font-family:'Bebas Neue',cursive; font-size:1.1rem; color:#E8334A">{pa:.0f}%</div>
                        </div>
                    </div>
                    <div style="display:flex; gap:2px; margin-top:8px; height:4px; border-radius:2px; overflow:hidden;">
                        <div style="width:{ph}%; background:#10B981;"></div>
                        <div style="width:{pd_}%; background:#4B5563;"></div>
                        <div style="width:{pa}%; background:#E8334A;"></div>
                    </div>
                    <div style="text-align:center; margin-top:8px;">
                        <span style="background:{'linear-gradient(135deg,#F5C842,#E8334A)' if phase=='final' else '#1F2937'};
                                     color:{'#000' if phase=='final' else '#F5C842'};
                                     font-family:'Bebas Neue',cursive; font-size:0.85rem; letter-spacing:1px;
                                     padding:3px 12px; border-radius:99px;">
                            ➤ {winner}
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        if phase == "final" and bracket.get("final"):
            finalist = bracket["final"][0]
            st.markdown(f"""
            <div style="text-align:center; margin-top:24px; padding:24px;
                        background:linear-gradient(135deg,#0A0E1A,#1a1a0a);
                        border:2px solid #F5C842; border-radius:16px;">
                <div style="font-size:0.75rem; color:#6B7280; text-transform:uppercase; letter-spacing:2px; margin-bottom:8px;">
                    🏆 CHAMPION PRÉDIT
                </div>
                <div style="font-size:4rem">{flag(finalist['winner'])}</div>
                <div style="font-family:'Bebas Neue',cursive; font-size:2.5rem; color:#F5C842; letter-spacing:3px;">
                    {finalist['winner']}
                </div>
                <div style="font-size:0.8rem; color:#6B7280; margin-top:8px;">
                    Basé sur le modèle de Poisson v5 + bookmakers Polymarket + expérience WC
                </div>
            </div>
            """, unsafe_allow_html=True)
