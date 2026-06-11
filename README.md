# 🏆 World Cup 2026 Predictor

> Modèle de Poisson vectorisé · Monte Carlo · Bookmakers · Buteurs · Bracket FIFA officiel

Application Streamlit de prédiction pour la Coupe du Monde 2026 (USA · Canada · Mexique, 11 juin – 19 juillet 2026).

🔗 **[Voir l'application en ligne](https://worldcup-2026-predictor.streamlit.app)** 

---

## 🎯 Fonctionnalités

| Onglet | Description |
|--------|-------------|
| 🏆 **Pronostics tournoi** | Probabilités de victoire finale (2 000 simulations Monte Carlo) comparées aux cotes bookmakers Polymarket/Kalshi (~1Md$) |
| ⚽ **Matchs de groupe** | 72 matchs prédits avec score, barre de probabilités victoire/nul/défaite, filtrable par groupe |
| 🥇 **Buteurs** | Top buteurs sur le tournoi complet (groupes + phase finale) · Buteurs par équipe avec buts attendus |
| 👥 **Groupes** | 12 groupes avec probabilités de qualification simulées (1 000 simulations) |
| 🔍 **Match par match** | Analyse détaillée : score prédit, matrice des scores, buteurs probables |
| 🏟️ **Phase finale** | Bracket complet R32→Finale avec équipes les plus probables, scores prédits et champion prédit |

---

## 🧠 Modèle

### Modèle de Poisson vectorisé (v5)
- **Données** : 7 952 matchs internationaux (2018–2026), 255 équipes
- **Entraînement** : vectorisé NumPy — < 1 seconde (vs 4h30 non vectorisé)
- **Décroissance temporelle** : `decay_rate=0.005` (matchs récents plus pondérés)
- **Poids des compétitions** : FIFA WC=4.0, UEFA Euro=3.5, Copa América=3.0, Qualifs WC=3.0, UEFA NL=2.2, Amicaux=0.2

### Facteurs de correction
| Facteur | Poids | Description |
|---------|-------|-------------|
| **Bookmakers** | 75% | Polymarket + Kalshi (~1Md$ de volume, mai 2026) |
| **Classement FIFA** | 25% | Classement FIFA avril 2026 |
| **Confédérations** | ×1.12–×0.86 | UEFA +12%, CONMEBOL +9%, AFC −10%, OFC −14% |
| **Expérience WC** | +0–+17% | Points 2018+2022 (Champion=10, Final=7, SF=5, QF=3, R16=1) |
| **Correction défense** | 50–70% | Corrélée aux cotes pour éviter les défenses irréalistes (AFC/OFC 70%) |

### Forces finales (mai 2026)
| Équipe | Attaque | Défense | Cote bookmaker |
|--------|---------|---------|----------------|
| Espagne | 2.036 | 0.573 | 16.5% |
| France | 1.882 | 0.573 | 17.4% |
| Angleterre | 1.741 | 0.598 | 11.3% |
| Argentine | 1.669 | 0.655 | 10.5% |
| Brésil | 1.448 | 0.747 | 9.5% |

### Simulation Monte Carlo
- **2 000 simulations** du tournoi complet
- Bracket officiel FIFA WC 2026 (Annexe C — 50 combinaisons des meilleurs 3es)
- Résultats : Espagne 24.8%, France 21.1%, Angleterre 13.4%, Argentine 10.7%

### Prédiction des buteurs
- **Base** : ratios historiques par joueur depuis 2020 (7 739 buts)
- **Listes officielles** : France (14 mai), Brésil (18 mai), Portugal (19 mai), Croatie, Autriche
- **Boosts manuels** : basés sur les stats 2025-26 (Dembélé ×1.35, Yamal ×1.30, Wirtz ×1.30...)
- **Plafonnement** : max 7.0 buts (record ère moderne = 8, Just Fontaine 1958 = 13)
- **Exclusions** : Griezmann, Di María, Giroud, Morata, Perišić, Džeko, Diogo Jota (†), Rodrygo, Neymar (remis dans Brésil)...

---

## 📊 Résultats clés

### Probabilités de remporter la WC 2026
```
🥇 Espagne      24.8%   (bookmakers : 16.5%)
🥈 France       21.1%   (bookmakers : 17.4%)
🥉 Angleterre   13.4%   (bookmakers : 11.3%)
4. Argentine    10.7%   (bookmakers : 10.5%)
5. Pays-Bas      4.5%
6. Allemagne     4.1%
7. Portugal      4.1%
8. Brésil        3.6%
```

### Top buteurs attendus (phase de groupes)
```
1. Kylian Mbappé      3.27 buts  (France)
2. Erling Haaland     2.99 buts  (Norvège)
3. Harry Kane         2.44 buts  (Angleterre)
4. Lionel Messi       2.10 buts  (Argentine)
5. Cristiano Ronaldo  1.73 buts  (Portugal)
```

---

## 🏗️ Architecture

```
world-cup-predictor/
├── app.py                          # Interface Streamlit (6 onglets)
├── requirements.txt
├── README.md
├── data/
│   └── raw/
│       ├── results.csv             # 49 287 matchs historiques (Kaggle)
│       ├── goalscorers.csv         # 47 601 buteurs historiques
│       └── shootouts.csv
└── src/
    ├── data/
    │   └── collect.py              # Chargement et preprocessing
    ├── models/
    │   ├── match_predictor.py      # Modèle Poisson v5 vectorisé NumPy
    │   └── scorer_predictor.py     # Prédiction buteurs par match et tournoi
    └── simulation/
        └── tournament.py           # Simulateur Monte Carlo + bracket FIFA
```

---

## 🚀 Installation

```bash
git clone https://github.com/Julie-Landrevie/world-cup-predictor.git
cd world-cup-predictor

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

L'application sera accessible sur `http://localhost:8501`.

---

## 📦 Données

Les données historiques viennent du dataset [International Football Results (1872-2024)](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017) sur Kaggle.

Télécharger les fichiers CSV et les placer dans `data/raw/` :
- `results.csv`
- `goalscorers.csv`
- `shootouts.csv`

---

## 🔄 Mise à jour pendant la compétition

Les vraies résultats peuvent être intégrés via `TournamentState` :

```python
from src.simulation.tournament import TournamentState, TournamentSimulator

state = TournamentState()
state.add_real_result("France", "Senegal", 2, 1)   # France gagne 2-1
state.add_real_result("Norway", "Iraq",   1, 0)    # Norvège gagne 1-0

sim = TournamentSimulator(model=match_model, state=state)
results = sim.run_monte_carlo(n_simulations=2000)
```

Les listes officielles (`OFFICIAL_SQUADS` dans `scorer_predictor.py`) seront mises à jour au fur et à mesure des annonces (deadline FIFA : 2 juin 2026).

---

## ⚠️ Points d'attention

- Les probabilités du modèle peuvent différer des bookmakers : le modèle est purement statistique, les bookmakers intègrent aussi des facteurs humains (blessures de dernière minute, moral, conditions météo).
- **Record de buts** : ère moderne = 8 buts (Müller 2010), record absolu = 13 buts (Just Fontaine 1958). Les buts attendus sont plafonnés à 7.0.
- **Expérience WC** : le Maroc (SF 2022) et la Croatie (Final 2018 + 3e 2022) ont des bonus significatifs qui peuvent paraître élevés mais reflètent leurs vraies performances récentes.
- **Listes officielles** : certaines équipes n'ont pas encore annoncé leur sélection définitive (deadline : 2 juin 2026). Le modèle utilise les listes disponibles et des estimations pour les autres.

---

## 🛠️ Tech stack

- **Python 3.11+**
- **Streamlit** — interface web
- **NumPy / Pandas** — modèle vectorisé et manipulation des données
- **SciPy** — distribution de Poisson
- **Loguru** — logging

---

## 👩‍💻 Auteure

**Julie Landrevie** — Data Scientist

[![GitHub](https://img.shields.io/badge/GitHub-Julie--Landrevie-black?logo=github)](https://github.com/Julie-Landrevie)

*Ce projet fait partie d'un portfolio data science incluant également le [MPG Optimizer](https://github.com/Julie-Landrevie/mpg-optimizer) (en ligne sur [mpg-optimizer.streamlit.app](https://mpg-optimizer.streamlit.app)).*
