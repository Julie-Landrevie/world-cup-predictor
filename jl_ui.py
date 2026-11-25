"""Charte commune des applications de Julie Landrevie (direction « Éditorial »).

Même charte que le site julie-landrevie.github.io et les visuels LinkedIn :
fond crème / encre, vert d'eau, une seule mise en avant rouille, DM Serif Display + Lato.

- Thème clair et sombre : définis dans .streamlit/config.toml ([theme.light], [theme.dark]) ;
  le visiteur choisit dans le menu ⋮ (System / Light / Dark). Tout suit le thème affiché :
  `setup()` lit le thème (st.context.theme) et remplit C, SEQ et DIV avec la palette claire ou
  sombre ; les figures matplotlib, les terrains, Plotly et Altair utilisent donc les bonnes
  couleurs. Quand le visiteur change de thème, la page est relancée automatiquement.
  Règle pour le code des applications : jamais de couleur écrite en dur (ni "white", ni "#1c1917"),
  toujours C["ink"], C["bg"], C["teal"]… ; et ne jamais mettre en cache une figure sans passer
  theme_key() en argument.
- Langue : sélecteur FR / EN en haut de la barre latérale, mémorisé dans l'adresse (?lang=en).

Usage :
    from jl_ui import setup, t, kpi, footer, C, mpl_style, plotly_style, altair_style
    setup("Titre FR", "Title EN", icon="⚽")
    st.markdown(t("Bonjour", "Hello"))
"""
from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components

# Palette claire (référence) ; C, SEQ et DIV prennent la palette active à chaque setup().
LIGHT = dict(teal="#0d8f84", rust="#cc4a12", compare="#c9c0b1", ink="#1c1917", ink2="#57534e",
         muted="#8a8580", grid="#e4dccd", bg="#f6f1e7", card="#fbf8f2")
# Version sombre (pour le CSS des encadrés, appliquée automatiquement).
DARK = dict(teal="#2fb3a6", rust="#e8692f", compare="#6f675d", ink="#f2ede4", ink2="#b9b1a6",
            muted="#8a8379", grid="#302a24", bg="#13110f", card="#1c1916")
SEQ_LIGHT = ["#e8f3f1", "#9fd3cc", "#4fb0a5", "#0d8f84", "#08645c"]   # dégradé vert d'eau (du fond vers la donnée)
SEQ_DARK = ["#1d2a27", "#1e554f", "#267b71", "#2fb3a6", "#89d5cc"]
DIV_LIGHT = ["#cc4a12", "#e7a481", "#f6f1e7", "#7fbdb5", "#0d8f84"]   # rouille → vert d'eau
DIV_DARK = ["#e8692f", "#8a4a2c", "#1c1916", "#1f6b63", "#2fb3a6"]
C = dict(LIGHT)
SEQ = list(SEQ_LIGHT)
DIV = list(DIV_LIGHT)
FONT = "Lato, 'Helvetica Neue', Arial, sans-serif"
TITLE_FONT = "'DM Serif Display', Georgia, serif"


# ------------------------------------------------------------------------------ langue
def lang() -> str:
    return st.session_state.get("jl_lang", "fr")


def t(fr: str, en: str) -> str:
    """Texte dans la langue choisie."""
    return en if lang() == "en" else fr


def _language_selector() -> None:
    if "jl_lang" not in st.session_state:
        q = st.query_params.get("lang", "fr")
        st.session_state.jl_lang = "en" if str(q).lower().startswith("en") else "fr"
    kw = dict(default="EN" if lang() == "en" else "FR", key="jl_lang_widget", label_visibility="collapsed")
    try:
        choice = st.sidebar.segmented_control("Langue / Language", ["FR", "EN"], required=True, **kw)
    except TypeError:                                   # Streamlit plus ancien : pas d'option required
        choice = st.sidebar.segmented_control("Langue / Language", ["FR", "EN"], **kw)
    new = "en" if choice == "EN" else "fr"
    if choice and new != st.session_state.jl_lang:
        st.session_state.jl_lang = new
    if st.session_state.jl_lang == "en":
        st.query_params["lang"] = "en"
    elif "lang" in st.query_params:
        del st.query_params["lang"]
    st.sidebar.caption(t("☀ / ☾ Thème clair ou sombre : menu ⋮ en haut à droite.",
                         "☀ / ☾ Light or dark theme: ⋮ menu, top right."))


# ------------------------------------------------------------------------------ thème
def _detected_theme():
    """'dark', 'light' ou None si cette version de Streamlit ne donne pas le thème."""
    try:
        th = st.context.theme.type
    except Exception:
        return None
    return th if th in ("dark", "light") else None


def is_dark() -> bool:
    return _detected_theme() == "dark"


def theme_key() -> str:
    """À passer en argument des fonctions mises en cache qui dessinent une figure."""
    return "dark" if is_dark() else "light"


def rgba(color: str, alpha: float) -> str:
    """Couleur hexadécimale avec transparence, au format CSS / Plotly."""
    h = color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _apply_palette() -> None:
    dark = is_dark()
    C.clear(); C.update(DARK if dark else LIGHT)
    SEQ[:] = SEQ_DARK if dark else SEQ_LIGHT
    DIV[:] = DIV_DARK if dark else DIV_LIGHT


# ------------------------------------------------------------------------------ style
def _css() -> str:
    def block(p: dict) -> str:
        return (f"--jl-teal:{p['teal']};--jl-rust:{p['rust']};--jl-ink:{p['ink']};--jl-ink2:{p['ink2']};"
                f"--jl-muted:{p['muted']};--jl-grid:{p['grid']};--jl-card:{p['card']};--jl-compare:{p['compare']};")
    return f"""
<style>
:root{{{block(C)}}}
body[data-jl-theme="dark"]{{{block(DARK)}}}
h1, h2, h3 {{ font-weight: 400 !important; letter-spacing: 0; }}
h1 {{ line-height: 1.1 !important; }}
.jl-sub {{ color: var(--jl-ink2); font-size: 1.05rem; margin-top: -0.4rem; }}
.jl-kpi {{ background: var(--jl-card); border: 1px solid var(--jl-grid); border-radius: 10px; padding: 0.9rem 1rem; height: 100%; }}
.jl-kpi .v {{ font-family: {TITLE_FONT}; font-size: 2.1rem; color: var(--jl-ink); line-height: 1.1; }}
.jl-kpi .v.accent {{ color: var(--jl-rust); }}
.jl-kpi .l {{ color: var(--jl-ink2); font-size: 0.9rem; }}
.jl-note {{ border-left: 3px solid var(--jl-teal); background: var(--jl-card); padding: 0.7rem 1rem; border-radius: 0 8px 8px 0;
  color: var(--jl-ink); margin: 0.6rem 0; }}
.jl-note.warn {{ border-left-color: var(--jl-rust); }}
.jl-tag {{ display: inline-block; border: 1px solid var(--jl-grid); border-radius: 999px; padding: 0.1rem 0.6rem;
  font-size: 0.8rem; color: var(--jl-ink2); margin: 0 0.3rem 0.3rem 0; }}
.jl-foot {{ color: var(--jl-ink2); font-size: 0.85rem; border-top: 1px solid var(--jl-grid); margin-top: 2rem;
  padding-top: 0.8rem; display: flex; justify-content: space-between; gap: 1rem; flex-wrap: wrap; }}
.jl-foot b {{ color: var(--jl-ink); }}
.jl-mono {{ display: inline-block; border: 2px solid var(--jl-teal); border-radius: 6px; color: var(--jl-teal);
  font-weight: 700; padding: 0 0.4rem; margin-left: 0.4rem; }}
[data-testid="stImage"] img {{ border-radius: 10px; }}
</style>"""


def _theme_sync() -> None:
    """Pose data-jl-theme="dark" ou "light" sur la page selon le fond affiché (encadrés HTML),
    et relance l'application quand le thème affiché ne correspond plus à celui des figures
    (bascule dans le menu ⋮), pour que graphiques et terrains changent aussi."""
    server = _detected_theme() or ""
    script = """<script>
(function(){
  var w = window.parent && window.parent.document ? window.parent : window;
  w.__jlServerTheme = "%s";
  if (w.__jlThemeSync) return; w.__jlThemeSync = true;
  var d = w.document;
  function lum(c){ var m = c.match(/\\d+/g); if(!m) return 1; return (0.299*m[0]+0.587*m[1]+0.114*m[2])/255; }
  function rerun(){
    d.dispatchEvent(new w.KeyboardEvent('keydown', {key: 'r', code: 'KeyR', keyCode: 82, which: 82, bubbles: true}));
  }
  function sync(){
    var app = d.querySelector('[data-testid="stApp"]') || d.body;
    var th = lum(w.getComputedStyle(app).backgroundColor) < 0.5 ? 'dark' : 'light';
    d.body.setAttribute('data-jl-theme', th);
    var srv = w.__jlServerTheme;
    if (srv && th !== srv && w.__jlAsked !== th) { w.__jlAsked = th; rerun(); }
    if (srv && th === srv) { w.__jlAsked = null; }
  }
  sync(); w.setInterval(sync, 400);
})();
</script>""" % server
    try:
        st.html(script, unsafe_allow_javascript=True)      # Streamlit récent
    except TypeError:
        components.html(script, height=0)                   # versions plus anciennes


def setup(title_fr: str, title_en: str, icon: str = "⚽", layout: str = "wide") -> None:
    """À appeler en tout premier : page, langue, style."""
    st.set_page_config(page_title=title_fr, page_icon=icon, layout=layout)
    _apply_palette()
    _language_selector()
    st.markdown(_css(), unsafe_allow_html=True)
    _theme_sync()


# ------------------------------------------------------------------------------ éléments
def header(title: str, subtitle: str = "") -> None:
    st.markdown(f"# {title}")
    if subtitle:
        st.markdown(f'<p class="jl-sub">{subtitle}</p>', unsafe_allow_html=True)


def kpi(col, value: str, label: str, accent: bool = False) -> None:
    col.markdown(f'<div class="jl-kpi"><div class="v{" accent" if accent else ""}">{value}</div>'
                 f'<div class="l">{label}</div></div>', unsafe_allow_html=True)


def note(text: str, warn: bool = False) -> None:
    st.markdown(f'<div class="jl-note{" warn" if warn else ""}">{text}</div>', unsafe_allow_html=True)


def footer(source_fr: str = "Data: StatsBomb (Open Data)", source_en: str | None = None) -> None:
    src = t(source_fr, source_en or source_fr)
    st.markdown(f"""<div class="jl-foot"><span>{src}</span><span><b>Julie Landrevie</b> ·
Data &amp; performance football <span class="jl-mono">JL</span> ·
<a href="https://julie-landrevie.github.io{'/?lang=en' if lang() == 'en' else ''}" target="_blank">Portfolio</a></span></div>""",
                unsafe_allow_html=True)


def fr_num(x: float, dec: int = 1, pct: bool = False) -> str:
    """Nombre au format de la langue choisie (virgule et espace fine en français)."""
    if lang() == "en":
        s = f"{x:,.{dec}f}"
        return s + ("%" if pct else "")
    s = f"{x:,.{dec}f}".replace(",", " ").replace(".", ",")
    return s + (" %" if pct else "")


# ------------------------------------------------------------------------------ graphiques
def altair_style(chart):
    """Polices et grille de la charte ; fond et texte laissés au thème Streamlit (clair ou sombre)."""
    return (chart.configure(font=FONT)
            .configure_view(stroke=None)
            .configure_axis(domain=False, ticks=False, labelFont=FONT, titleFont=FONT,
                            titleFontWeight="normal", labelLimit=260, gridColor=C["grid"])
            .configure_legend(labelFont=FONT, titleFont=FONT, titleFontWeight="normal"))


def plotly_style(fig, height: int | None = None):
    """Plotly : polices et couleurs de la charte ; fond et texte suivent le thème (theme="streamlit")."""
    fig.update_layout(font_family=FONT, colorway=[C["teal"], C["rust"], C["compare"], C["ink2"], C["muted"]],
                      margin=dict(l=10, r=10, t=40, b=10), legend_title_text="",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(gridcolor=C["grid"], zerolinecolor=C["grid"])
    fig.update_yaxes(gridcolor=C["grid"], zerolinecolor=C["grid"])
    if height:
        fig.update_layout(height=height)
    return fig


_MPL_DONE = False


def mpl_style():
    """matplotlib / mplsoccer : couleurs du thème actif (fond crème ou sombre) et polices de la charte."""
    global _MPL_DONE
    import matplotlib as mpl
    if not _MPL_DONE:
        _register_fonts()
        _MPL_DONE = True
    mpl.rcParams.update({
        "figure.facecolor": C["bg"], "axes.facecolor": C["bg"], "savefig.facecolor": C["bg"],
        "axes.edgecolor": C["grid"], "axes.labelcolor": C["ink2"], "text.color": C["ink"],
        "xtick.color": C["ink2"], "ytick.color": C["ink2"], "grid.color": C["grid"],
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.prop_cycle": mpl.cycler(color=[C["teal"], C["rust"], C["compare"], C["ink2"]]),
        "legend.labelcolor": C["ink"], "axes.titlecolor": C["ink"],
        "font.family": ["Lato", "DejaVu Sans"], "axes.titleweight": "normal",
        "legend.frameon": False,
    })
    return C


def pitch_kwargs(**extra) -> dict:
    """Arguments mplsoccer Pitch / VerticalPitch dans la charte."""
    kw = dict(pitch_color=C["bg"], line_color=C["compare"], linewidth=1.2)
    kw.update(extra)
    return kw


def _register_fonts() -> None:
    """Télécharge Lato et DM Serif Display une fois (Google Fonts, licence OFL) pour matplotlib.
    En cas d'échec (hors ligne), matplotlib garde sa police par défaut."""
    import os
    import urllib.request
    from matplotlib import font_manager as fm
    base = "https://raw.githubusercontent.com/google/fonts/main/ofl/"
    files = ["lato/Lato-Regular.ttf", "lato/Lato-Bold.ttf", "lato/Lato-Black.ttf",
             "dmserifdisplay/DMSerifDisplay-Regular.ttf"]
    d = os.path.join(os.path.expanduser("~"), ".fonts_jl")
    try:
        os.makedirs(d, exist_ok=True)
        for f in files:
            path = os.path.join(d, os.path.basename(f))
            if not os.path.exists(path):
                urllib.request.urlretrieve(base + f, path)
            fm.fontManager.addfont(path)
    except Exception:
        pass


def title_font():
    """FontProperties DM Serif Display pour un titre matplotlib (ou None si indisponible)."""
    import os
    from matplotlib import font_manager as fm
    path = os.path.join(os.path.expanduser("~"), ".fonts_jl", "DMSerifDisplay-Regular.ttf")
    return fm.FontProperties(fname=path) if os.path.exists(path) else None
