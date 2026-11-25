"""
app.py — World Cup 2026 Predictor (français / anglais, clair / sombre)
Charte commune : jl_ui.py et .streamlit/config.toml.
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from jl_ui import SEQ, footer, fr_num, header, kpi, note, setup, t  # noqa: E402

setup("World Cup 2026 Predictor — Julie Landrevie", "World Cup 2026 Predictor — Julie Landrevie", icon="🏆")

import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, to_hex  # noqa: E402

from src.data.collect import load_goalscorers, load_results, load_wc2026_fixtures  # noqa: E402
from src.models.match_predictor import BOOKMAKER_PROBS, PoissonPredictor  # noqa: E402
from src.models.scorer_predictor import ScorerPredictor  # noqa: E402
from src.simulation.tournament import WC2026_GROUPS, TournamentSimulator, TournamentState  # noqa: E402

# Encadrés : couleurs de la charte via les variables CSS de jl_ui (suivent le thème clair / sombre)
st.markdown("""<style>
.wc-card{background:var(--jl-card);border:1px solid var(--jl-grid);border-radius:10px;padding:10px 14px;margin-bottom:8px;}
.wc-card.hl{border-color:var(--jl-rust);}
.wc-num{font-family:'DM Serif Display',Georgia,serif;color:var(--jl-ink);}
.wc-small{font-size:0.75rem;color:var(--jl-ink2);}
.wc-bar{height:6px;background:var(--jl-grid);border-radius:3px;overflow:hidden;}
.wc-bar > div{height:6px;background:var(--jl-teal);}
.wc-3{display:flex;gap:2px;height:6px;border-radius:3px;overflow:hidden;margin-top:8px;}
</style>""", unsafe_allow_html=True)


def pct(x, dec=1):
    return fr_num(x, dec, pct=True)


def three_bar(ph, pdraw, pa):
    """Victoire / nul / défaite : vert d'eau / gris / rouille."""
    return (f'<div class="wc-3"><div style="width:{ph}%;background:var(--jl-teal)"></div>'
            f'<div style="width:{pdraw}%;background:var(--jl-compare)"></div>'
            f'<div style="width:{pa}%;background:var(--jl-rust)"></div></div>')

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

@st.cache_resource(show_spinner=False)
def load_models():
    """
    Charge et entraîne tous les modèles.
    cache_resource = partagé entre tous les utilisateurs, persist en mémoire.
    À vider manuellement si les données changent (bouton sidebar).
    """
    results     = load_results(min_year=2018)
    goalscorers = load_goalscorers(min_year=2020)
    fixtures    = load_wc2026_fixtures()

    match_model = PoissonPredictor(min_year=2018, decay_rate=0.005, n_iter=30)
    match_model.fit(results)

    scorer_model = ScorerPredictor(min_year=2020)
    scorer_model.fit(goalscorers)

    all_preds = match_model.predict_all_fixtures(fixtures)

    return match_model, scorer_model, fixtures, all_preds


@st.cache_data(show_spinner=False, ttl=3600)
def get_tournament_probs(n_sims=2000, _results_hash=""):
    """
    Monte Carlo — probabilités de remporter le tournoi + données phase finale.
    TTL=1h — se recalcule automatiquement toutes les heures.
    _results_hash : paramètre qui force le recalcul si les vrais résultats changent.
    """
    match_model, _, _, _ = load_models()
    state = TournamentState()
    # Charger les vrais résultats depuis st.session_state si disponibles
    for r in st.session_state.get("real_results", []):
        state.add_real_result(r["home"], r["away"], r["home_score"], r["away_score"], r.get("stage", "group"))
    sim   = TournamentSimulator(model=match_model, state=state)
    df, ko_data = sim.run_monte_carlo(n_simulations=n_sims)
    return df, ko_data


@st.cache_data(show_spinner=False, ttl=3600)
def get_group_stage_probs(_results_hash=""):
    """Probabilités de groupe — recalcul si vrais résultats changent."""
    match_model, _, _, _ = load_models()
    state = TournamentState()
    for r in st.session_state.get("real_results", []):
        state.add_real_result(r["home"], r["away"], r["home_score"], r["away_score"], r.get("stage", "group"))
    sim = TournamentSimulator(model=match_model, state=state)
    return sim.group_stage_probabilities(n_simulations=1000)



def results_hash():
    return str(len(st.session_state.get("real_results", [])))


def clear_sim_caches():
    get_tournament_probs.clear()
    get_group_stage_probs.clear()


# ============================================================
# EN-TÊTE
# ============================================================

header("World Cup 2026 Predictor",
       t("Pronostic d’avant tournoi de la Coupe du monde 2026 (États-Unis, Canada, Mexique) : modèle de Poisson, "
         "cotes des marchés de prédiction et simulations de Monte-Carlo.",
         "Pre-tournament forecast of the 2026 World Cup (USA, Canada, Mexico): Poisson model, prediction-market odds "
         "and Monte Carlo simulations."))

# ============================================================
# BARRE LATÉRALE : vrais résultats pendant la compétition
# ============================================================

STAGES = {"group": ("Phase de groupes", "Group stage"), "r32": ("Seizièmes", "Round of 32"),
          "r16": ("Huitièmes", "Round of 16"), "qf": ("Quarts", "Quarter-finals"),
          "sf": ("Demi-finales", "Semi-finals"), "final": ("Finale", "Final")}

with st.sidebar:
    st.markdown(f"### {t('Résultats réels', 'Actual results')}")
    st.caption(t("Ajoute des résultats pour recalculer les probabilités.", "Add results to recompute the probabilities."))
    if "real_results" not in st.session_state:
        st.session_state.real_results = []

    with st.expander(t("Ajouter un résultat", "Add a result"), expanded=False):
        all_teams_sidebar = sorted(set(x for teams in WC2026_GROUPS.values() for x in teams))
        home_s = st.selectbox(t("Équipe 1", "Team 1"), all_teams_sidebar, key="sidebar_home")
        away_s = st.selectbox(t("Équipe 2", "Team 2"), [x for x in all_teams_sidebar if x != home_s], key="sidebar_away")
        col_sh, col_sa = st.columns(2)
        score_h = col_sh.number_input(t("Buts équipe 1", "Team 1 goals"), 0, 20, 0, key="sidebar_sh")
        score_a = col_sa.number_input(t("Buts équipe 2", "Team 2 goals"), 0, 20, 0, key="sidebar_sa")
        stage_s = st.selectbox(t("Phase", "Stage"), list(STAGES), key="sidebar_stage", format_func=lambda k: t(*STAGES[k]))
        if st.button(t("Enregistrer ce résultat", "Save this result"), width="stretch"):
            st.session_state.real_results.append({"home": home_s, "away": away_s, "home_score": int(score_h),
                                                  "away_score": int(score_a), "stage": stage_s})
            clear_sim_caches()
            st.rerun()

    if st.session_state.real_results:
        st.markdown(t(f"**{len(st.session_state.real_results)} résultat(s) enregistré(s)**",
                      f"**{len(st.session_state.real_results)} result(s) saved**"))
        for i, r in enumerate(st.session_state.real_results):
            col_r, col_del = st.columns([4, 1])
            col_r.markdown(f"{flag(r['home'])} **{r['home_score']}-{r['away_score']}** {flag(r['away'])}")
            if col_del.button("✕", key=f"del_{i}"):
                st.session_state.real_results.pop(i)
                clear_sim_caches()
                st.rerun()
        if st.button(t("Recalculer", "Recompute"), width="stretch", type="primary"):
            clear_sim_caches()
            st.rerun()
        if st.button(t("Effacer tous les résultats", "Clear all results"), width="stretch"):
            st.session_state.real_results = []
            clear_sim_caches()
            st.rerun()

    st.divider()
    if st.button(t("Vider les caches", "Clear caches"), width="stretch"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()
    st.caption(t("Modèle de Poisson v5 · Monte-Carlo", "Poisson model v5 · Monte Carlo"))


with st.spinner(t("Entraînement du modèle de Poisson…", "Training the Poisson model…")):
    try:
        match_model, scorer_model, fixtures, all_preds = load_models()
    except Exception as e:
        st.error(t(f"Erreur : {e}", f"Error: {e}"))
        st.stop()

c = st.columns(5)
kpi(c[0], "48", t("Équipes", "Teams"))
kpi(c[1], str(len(all_preds)), t("Matchs prédits", "Matches predicted"))
kpi(c[2], t("7 952", "7,952"), t("Matchs historiques", "Historical matches"))
kpi(c[3], t("1 079", "1,079"), t("Buteurs analysés", "Scorers analysed"))
kpi(c[4], t("11 juin", "11 June"), t("Début du tournoi (2026)", "Tournament start (2026)"))
st.write("")


# ============================================================
# ONGLETS
# ============================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    t("Vainqueur", "Winner"), t("Matchs de groupe", "Group matches"), t("Buteurs", "Scorers"),
    t("Groupes", "Groups"), t("Match par match", "Match by match"), t("Phase finale", "Knockouts")])


# ── ONGLET 1 : VAINQUEUR ───────────────────────────────────
with tab1:
    st.subheader(t("Probabilité de remporter la Coupe du monde", "Probability of winning the World Cup"))
    st.caption(t("2 000 tournois simulés (Monte-Carlo) : modèle de Poisson, classement FIFA et cotes des marchés de prédiction.",
                 "2,000 simulated tournaments (Monte Carlo): Poisson model, FIFA ranking and prediction-market odds."))
    col_mc, col_book = st.columns([3, 2], gap="large")
    with st.spinner(t("Simulation en cours (2 000 tournois)…", "Simulating 2,000 tournaments…")):
        mc, ko_data = get_tournament_probs(n_sims=2000, _results_hash=results_hash())

    with col_mc:
        st.markdown(f"**{t('Notre modèle', 'Our model')}**")
        html = ""
        for i, (_, row) in enumerate(mc.head(20).iterrows()):
            team, prob = row["team"], row["prob_win"]
            book = BOOKMAKER_PROBS.get(team, 0.003) * 100
            html += (f'<div class="wc-card{" hl" if i == 0 else ""}" style="display:flex;align-items:center;gap:12px">'
                     f'<div style="font-size:1.4rem;width:34px">{flag(team)}</div>'
                     f'<div style="flex:1"><div style="font-weight:700">{team}</div>'
                     f'<div class="wc-bar" style="margin-top:5px"><div style="width:{min(prob / 30 * 100, 100):.0f}%"></div></div></div>'
                     f'<div style="text-align:right;min-width:86px"><div class="wc-num" style="font-size:1.3rem">{pct(prob)}</div>'
                     f'<div class="wc-small">{t("marchés :", "markets:")} {pct(book)}</div></div></div>')
        st.markdown(html, unsafe_allow_html=True)

    with col_book:
        st.markdown(f"**{t('Cotes des marchés de prédiction (Polymarket)', 'Prediction-market odds (Polymarket)')}**")
        st.caption(t("Polymarket et Kalshi, environ 1 milliard de dollars de volume (mai 2026).", "Polymarket and Kalshi, about $1 billion in volume (May 2026)."))
        book_teams = [("France", 17.4), ("Spain", 16.5), ("England", 11.3), ("Argentina", 10.5), ("Brazil", 9.5),
                      ("Germany", 6.5), ("Portugal", 5.5), ("Netherlands", 4.0), ("Morocco", 2.0), ("Colombia", 1.4),
                      ("Uruguay", 1.2), ("United States", 1.6), ("Switzerland", 1.2), ("Croatia", 1.2), ("Norway", 1.0)]
        st.markdown("".join(
            f'<div style="display:flex;align-items:center;gap:10px;padding:5px 0;border-bottom:1px solid var(--jl-grid)">'
            f'<div style="width:24px;text-align:center">{flag(team)}</div><div style="flex:1">{team}</div>'
            f'<div class="wc-num" style="font-size:1.05rem">{pct(prob)}</div></div>' for team, prob in book_teams),
            unsafe_allow_html=True)
        note(t("<b>À garder en tête</b> : le modèle ne voit que les performances statistiques récentes et le tableau FIFA ; "
               "les marchés intègrent aussi blessures, forme du moment et autres facteurs humains.",
               "<b>Keep in mind</b>: the model only sees recent statistical performance and the FIFA bracket; markets also "
               "price in injuries, current form and other human factors."), warn=True)


# ── ONGLET 2 : MATCHS DE GROUPE ────────────────────────────
with tab2:
    st.subheader(t("Prédictions de la phase de groupes", "Group-stage predictions"))
    groups = sorted(WC2026_GROUPS.keys())
    selected_group = st.selectbox(t("Groupe", "Group"), ["__all__"] + groups,
                                  format_func=lambda g: t("Tous les groupes", "All groups") if g == "__all__" else t(f"Groupe {g}", f"Group {g}"))
    df_display = all_preds.copy()
    if selected_group != "__all__":
        group_teams = WC2026_GROUPS.get(selected_group, [])
        df_display = df_display[df_display["home"].isin(group_teams) | df_display["away"].isin(group_teams)]
    st.caption(t(f"{len(df_display)} matchs · barre : victoire équipe 1 (vert d’eau), nul (gris), victoire équipe 2 (rouille)",
                 f"{len(df_display)} matches · bar: team 1 win (teal), draw (grey), team 2 win (rust)"))
    html = ""
    for _, row in df_display.iterrows():
        ph, pdr, pa = row["% victoire"], row["% nul"], row["% défaite"]
        html += (f'<div class="wc-card"><div style="display:flex;align-items:center;justify-content:space-between;gap:12px">'
                 f'<div style="flex:2;text-align:right"><b>{row["home"]}</b> <span style="font-size:1.2rem">{flag(row["home"])}</span></div>'
                 f'<div style="flex:1;text-align:center"><div class="wc-num" style="font-size:1.4rem">{row["score_prédit"]}</div>'
                 f'<div class="wc-small">{row["date"]}</div></div>'
                 f'<div style="flex:2"><span style="font-size:1.2rem">{flag(row["away"])}</span> <b>{row["away"]}</b></div></div>'
                 f'{three_bar(ph, pdr, pa)}'
                 f'<div class="wc-small" style="display:flex;justify-content:space-between;margin-top:4px">'
                 f'<span>{pct(ph, 0)}</span><span>{t("nul", "draw")} {pct(pdr, 0)}</span><span>{pct(pa, 0)}</span></div></div>')
    st.markdown(html, unsafe_allow_html=True)


# ── ONGLET 3 : BUTEURS ─────────────────────────────────────
with tab3:
    st.subheader(t("Meilleurs buteurs sur tout le tournoi (simulation)", "Top scorers over the whole tournament (simulation)"))
    st.caption(t("Buts attendus sur l’ensemble du tournoi (groupes et phase finale) · 500 simulations",
                 "Expected goals over the whole tournament (groups and knockouts) · 500 simulations"))

    @st.cache_data
    def get_full_scorer_rankings(_results_hash=""):
        match_model_c, scorer_model_c, _, _ = load_models()
        state = TournamentState()
        for r in st.session_state.get("real_results", []):
            state.add_real_result(r["home"], r["away"], r["home_score"], r["away_score"], r.get("stage", "group"))
        sim = TournamentSimulator(model=match_model_c, state=state)
        return sim.run_scorer_monte_carlo(scorer_model_c, n_simulations=500)

    with st.spinner(t("Simulation des buteurs…", "Simulating scorers…")):
        full_scorers = get_full_scorer_rankings(_results_hash=results_hash())

    col_full, col_team = st.columns([3, 2], gap="large")
    with col_full:
        html = ""
        for i, (_, row) in enumerate(full_scorers.head(20).iterrows(), 1):
            team = row["Équipe"]
            html += (f'<div class="wc-card{" hl" if i == 1 else ""}" style="display:flex;align-items:center;gap:12px">'
                     f'<div class="wc-small" style="width:24px;text-align:center">{i}</div>'
                     f'<div style="font-size:1.2rem">{flag(team)}</div>'
                     f'<div style="flex:1"><div style="font-weight:700">{row["Joueur"]}</div>'
                     f'<div class="wc-small">{team} · {fr_num(row["Matchs moyens"], 1)} {t("matchs en moyenne", "matches on average")}</div></div>'
                     f'<div style="text-align:right"><div class="wc-num" style="font-size:1.3rem">{fr_num(row["Buts attendus"], 2)}</div>'
                     f'<div class="wc-small">{pct(row["% marquer"], 0)} {t("de marquer", "to score")}</div></div></div>')
        st.markdown(html, unsafe_allow_html=True)

    with col_team:
        st.markdown(f"**{t('Par équipe', 'By team')}**")
        st.caption(t("Buts attendus en phase de groupes et part de chaque joueur", "Expected group-stage goals and each player’s share"))
        team_options = sorted(x for teams in WC2026_GROUPS.values() for x in teams)
        selected_team = st.selectbox(t("Équipe", "Team"), team_options)
        if selected_team:
            team_preds = all_preds[(all_preds["home"] == selected_team) | (all_preds["away"] == selected_team)]
            total_exp, player_exp = 0, {}
            for _, row in team_preds.iterrows():
                exp_g = row["buts_home"] if row["home"] == selected_team else row["buts_away"]
                total_exp += exp_g
                if selected_team in scorer_model.team_scorers_:
                    for _, p in scorer_model.team_scorers_[selected_team].iterrows():
                        player_exp[p["scorer"]] = player_exp.get(p["scorer"], 0) + exp_g * p["ratio"]
            st.markdown(f"**{flag(selected_team)} {selected_team}** · "
                        + t(f"{fr_num(total_exp, 2)} buts attendus", f"{fr_num(total_exp, 2)} expected goals"))
            if player_exp:
                sorted_players = sorted(player_exp.items(), key=lambda x: x[1], reverse=True)
                max_exp = sorted_players[0][1] or 1
                st.markdown("".join(
                    f'<div class="wc-card"><div style="display:flex;justify-content:space-between">'
                    f'<span style="font-weight:700">{j}</span><span><span class="wc-num">{fr_num(e, 2)}</span> '
                    f'<span class="wc-small">· {pct(e / total_exp * 100 if total_exp else 0, 0)} {t("des buts", "of goals")}</span></span></div>'
                    f'<div class="wc-bar" style="margin-top:6px"><div style="width:{e / max_exp * 100:.0f}%"></div></div></div>'
                    for j, e in sorted_players[:8]), unsafe_allow_html=True)


# ── ONGLET 4 : GROUPES ─────────────────────────────────────
with tab4:
    st.subheader(t("Les 12 groupes", "The 12 groups"))
    st.caption(t("Probabilité de qualification (1 000 simulations) ; entre parenthèses, probabilité de finir 1er.",
                 "Qualification probability (1,000 simulations); in brackets, probability of finishing first."))
    with st.spinner(t("Calcul des probabilités de groupe…", "Computing group probabilities…")):
        group_probs = get_group_stage_probs(_results_hash=results_hash())
    group_keys = list(WC2026_GROUPS.keys())
    for row_i in range(0, len(group_keys), 3):
        cols = st.columns(3, gap="medium")
        for col_i, g in enumerate(group_keys[row_i:row_i + 3]):
            with cols[col_i]:
                st.markdown(f"**{t('Groupe', 'Group')} {g}**")
                gdf = group_probs[group_probs["group"] == g].sort_values("% qualif", ascending=False)
                st.markdown("".join(
                    f'<div class="wc-card" style="display:flex;align-items:center;gap:8px;padding:6px 10px;'
                    f'border-left:3px solid {"var(--jl-teal)" if tr["% qualif"] >= 50 else "var(--jl-grid)"}">'
                    f'<span>{flag(tr["team"])}</span><span style="flex:1">{tr["team"]}</span>'
                    f'<span class="wc-small">({pct(tr["% 1er"], 0)})</span>'
                    f'<span class="wc-num" style="font-size:1rem">{pct(tr["% qualif"], 0)}</span></div>'
                    for _, tr in gdf.iterrows()), unsafe_allow_html=True)


# ── ONGLET 5 : MATCH PAR MATCH ─────────────────────────────
with tab5:
    st.subheader(t("Analyse détaillée d’un match", "Detailed match analysis"))
    all_teams = sorted(set(x for teams in WC2026_GROUPS.values() for x in teams))
    col_h, col_a = st.columns(2)
    home = col_h.selectbox(t("Équipe 1", "Team 1"), all_teams, index=all_teams.index("France"))
    away = col_a.selectbox(t("Équipe 2", "Team 2"), all_teams, index=all_teams.index("Argentina"))
    if home != away:
        pred = match_model.predict_score(home, away, neutral=True)
        scorers = scorer_model.predict_match_scorers(home, away, pred["expected_home"], pred["expected_away"], top_n=5)
        st.markdown(f"""
<div class="wc-card" style="padding:22px;text-align:center;margin:12px 0">
  <div style="display:flex;align-items:center;justify-content:center;gap:28px;flex-wrap:wrap">
    <div><div style="font-size:2.2rem">{flag(home)}</div><div style="font-weight:700">{home}</div>
      <div class="wc-num" style="font-size:1.5rem">{pct(pred['prob_home_win'])}</div></div>
    <div><div class="wc-num" style="font-size:3rem">{pred['rounded_score']}</div>
      <div class="wc-small">{t('score le plus probable', 'most likely score')}</div>
      <div class="wc-small">{t('Nul :', 'Draw:')} {pct(pred['prob_draw'])}</div></div>
    <div><div style="font-size:2.2rem">{flag(away)}</div><div style="font-weight:700">{away}</div>
      <div class="wc-num" style="font-size:1.5rem">{pct(pred['prob_away_win'])}</div></div>
  </div>
  {three_bar(pred['prob_home_win'], pred['prob_draw'], pred['prob_away_win'])}
  <div class="wc-small" style="margin-top:10px">{t('Buts attendus :', 'Expected goals:')} {fr_num(pred['expected_home'], 2)} – {fr_num(pred['expected_away'], 2)}</div>
</div>""", unsafe_allow_html=True)

        st.markdown(f"**{t('Probabilité de chaque score', 'Probability of each score')}**")
        matrix = pred["score_matrix"]
        n = 5
        df_matrix = pd.DataFrame({f"{away} {a}": [matrix[h, a] * 100 for h in range(n + 1)] for a in range(n + 1)},
                                 index=[f"{home} {h}" for h in range(n + 1)])
        cmap = LinearSegmentedColormap.from_list("jl", SEQ)
        vmax = float(df_matrix.values.max()) or 1.0

        def _cell(v):   # fond du dégradé du thème actif, texte foncé ou clair selon ce fond
            r, g, b, _ = cmap(min(v / vmax, 1.0))
            ink = "#1c1917" if 0.299 * r + 0.587 * g + 0.114 * b > 0.55 else "#f2ede4"
            return f"background-color: {to_hex((r, g, b))}; color: {ink}"
        st.dataframe(df_matrix.style.format(lambda v: pct(v)).map(_cell), width="stretch")

        st.markdown(f"**{t('Buteurs probables', 'Likely scorers')}**")
        col_sh, col_sa = st.columns(2, gap="large")
        for col, team in [(col_sh, home), (col_sa, away)]:
            with col:
                st.markdown(f"{flag(team)} **{team}**")
                ts = scorers.get(team, pd.DataFrame())
                if not ts.empty and "scorer" in ts.columns:
                    st.markdown("".join(
                        f'<div class="wc-card" style="display:flex;align-items:center;gap:10px;padding:8px 12px">'
                        f'<div style="flex:1">{sr["scorer"]}</div>'
                        f'<div class="wc-bar" style="width:70px"><div style="width:{min(sr.get("% chance de marquer", 0) * 2, 100):.0f}%"></div></div>'
                        f'<div class="wc-num" style="min-width:46px;text-align:right">{pct(sr.get("% chance de marquer", 0), 0)}</div></div>'
                        for _, sr in ts.iterrows()), unsafe_allow_html=True)
    else:
        st.info(t("Choisis deux équipes différentes.", "Pick two different teams."))


# ── ONGLET 6 : PHASE FINALE ────────────────────────────────
with tab6:
    st.subheader(t("Phase finale : le scénario le plus probable", "Knockouts: the most likely scenario"))
    st.caption(t("Simulation déterministe : à chaque tour, l’équipe la plus probable passe.",
                 "Deterministic simulation: in each round, the most likely team goes through."))
    note(t("Record de buts sur une Coupe du monde : <b>13</b> (Just Fontaine, 1958) ; ère moderne : <b>8</b>. "
           "Au plus <b>8 matchs</b> par équipe avec la phase de groupes.",
           "Record goals at a single World Cup: <b>13</b> (Just Fontaine, 1958); modern era: <b>8</b>. "
           "At most <b>8 matches</b> per team including the group stage."))

    @st.cache_data
    def get_most_likely_bracket(_results_hash=""):
        match_model_b, scorer_model_b, _, _ = load_models()
        state_b = TournamentState()
        for r in st.session_state.get("real_results", []):
            state_b.add_real_result(r["home"], r["away"], r["home_score"], r["away_score"], r.get("stage", "group"))
        sim_b = TournamentSimulator(model=match_model_b, state=state_b)
        return sim_b.simulate_most_likely_bracket(scorer_model=scorer_model_b)

    with st.spinner(t("Calcul du tableau le plus probable…", "Computing the most likely bracket…")):
        bracket = get_most_likely_bracket(_results_hash=results_hash())

    PHASES = {"r32": (("Seizièmes de finale", "Round of 32"), 4), "r16": (("Huitièmes de finale", "Round of 16"), 4),
              "qf": (("Quarts de finale", "Quarter-finals"), 2), "sf": (("Demi-finales", "Semi-finals"), 2),
              "final": (("Finale", "Final"), 1)}
    for phase, (labels, n_cols) in PHASES.items():
        matches = bracket.get(phase, [])
        if not matches:
            continue
        st.markdown(f"#### {t(*labels)}")
        cols = st.columns(n_cols, gap="medium") if n_cols > 1 else [st.container()]
        for i, m in enumerate(matches):
            h, a, w = m["home"], m["away"], m["winner"]
            with cols[i % n_cols]:
                st.markdown(f"""
<div class="wc-card{' hl' if phase == 'final' else ''}" style="padding:12px 14px">
  <div class="wc-small" style="text-align:center;margin-bottom:6px">{m.get('match_id', '')}</div>
  <div style="display:flex;align-items:center;justify-content:space-between;gap:8px">
    <div style="flex:2;text-align:right"><div>{flag(h)}</div>
      <div style="font-weight:{700 if w == h else 400};color:var({'--jl-ink' if w == h else '--jl-ink2'})">{h}</div>
      <div class="wc-small">{pct(m['prob_home_win'], 0)}</div></div>
    <div style="flex:0 0 auto;text-align:center"><div class="wc-num" style="font-size:1.5rem;white-space:nowrap">{m['score']}</div>
      <div class="wc-small">{t('nul', 'draw')} {pct(m['prob_draw'], 0)}</div></div>
    <div style="flex:2"><div>{flag(a)}</div>
      <div style="font-weight:{700 if w == a else 400};color:var({'--jl-ink' if w == a else '--jl-ink2'})">{a}</div>
      <div class="wc-small">{pct(m['prob_away_win'], 0)}</div></div>
  </div>
  {three_bar(m['prob_home_win'], m['prob_draw'], m['prob_away_win'])}
  <div style="text-align:center;margin-top:8px"><span class="jl-tag">→ {w}</span></div>
</div>""", unsafe_allow_html=True)

        if phase == "final":
            champ = matches[0]["winner"]
            st.markdown(f"""
<div class="wc-card hl" style="text-align:center;padding:24px;margin-top:16px">
  <div class="wc-small" style="text-transform:uppercase;letter-spacing:2px">{t('Champion prédit', 'Predicted champion')}</div>
  <div style="font-size:3.4rem">{flag(champ)}</div>
  <div class="wc-num" style="font-size:2.4rem;color:var(--jl-rust)">{champ}</div>
  <div class="wc-small" style="margin-top:6px">{t('Modèle de Poisson v5, marchés de prédiction et expérience en Coupe du monde',
                                                   'Poisson model v5, prediction markets and World Cup experience')}</div>
</div>""", unsafe_allow_html=True)

footer("Données : International Football Results (Kaggle, martj42) · Polymarket et Kalshi (mai 2026)",
       "Data: International Football Results (Kaggle, martj42) · Polymarket and Kalshi (May 2026)")
