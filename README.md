# 🏆 World Cup 2026 Predictor

> Prédictions de scores et de buteurs pour la Coupe du Monde 2026 — modèle statistique de Poisson, données historiques internationales.

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)
![Status](https://img.shields.io/badge/status-en_développement-orange)
![License](https://img.shields.io/badge/license-MIT-green)

---

## 🌐 Application en ligne

**👉 Coming soon — déploiement Streamlit prévu avant le coup d'envoi (juin 2026)**

---

## 🎯 Objectifs du projet

Prédire les résultats de la Coupe du Monde 2026 match par match :

| Module | Description | Statut |
|--------|-------------|--------|
| 📊 **Données historiques** | 47 000 matchs internationaux depuis 1872 | 🔨 En cours |
| 🧮 **Modèle de Poisson** | Prédiction du score exact par match | 🔨 En cours |
| ⚽ **Prédiction buteurs** | Top scoreurs probables par équipe et par match | 🔜 Prévu |
| 🏟️ **Simulation tournoi** | Simulation complète phase de groupes → finale | 🔜 Prévu |
| 🌐 **Interface Streamlit** | App publique pour explorer les prédictions | 🔜 Prévu |

---

## 🧮 Approche statistique — Modèle de Poisson

Le modèle de Dixon-Coles (extension du modèle de Poisson) est la référence académique pour prédire les scores de football. C'est le modèle utilisé par de nombreux bookmakers professionnels.

**Principe :**
- Chaque équipe a une **force d'attaque** et une **force de défense** calculées sur ses résultats historiques
- Le nombre de buts d'une équipe suit une **loi de Poisson** paramétrée par ces forces
- On pondère les matchs récents plus fortement (les performances de 2020 comptent plus que celles de 2000)

```
λ_home = attaque_A × défense_B × avantage_domicile
λ_away = attaque_B × défense_A

P(score A-B) = Poisson(λ_home) × Poisson(λ_away)
```

**Résultat :** pour chaque match, le modèle prédit une distribution de scores possibles avec leurs probabilités.

---

## 📦 Sources de données

### Résultats historiques internationaux
- **Source** : [Kaggle — International Football Results](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017)
- 47 000+ matchs internationaux depuis 1872
- Données : date, équipes, score, compétition, lieu

### Calendrier & groupes WC 2026
- Groupes encodés manuellement depuis la [FIFA](https://www.fifa.com/fifaplus/en/tournaments/mens/worldcup/canadamexicousa2026)
- 48 équipes, 12 groupes de 4, phase finale étendue

### Stats joueurs (buteurs)
- **Source** : API [football-data.org](https://www.football-data.org/) (gratuite)
- Statistiques individuelles des joueurs en sélection

---

## 🗂️ Structure du projet

```
world-cup-predictor/
├── app.py                              # Interface Streamlit
├── requirements.txt
├── data/
│   └── raw/
│       ├── international_results.csv   # Historique matchs (Kaggle)
│       └── wc2026_groups.csv           # Groupes WC 2026
├── src/
│   ├── data/
│   │   └── collect.py                  # Chargement & nettoyage données
│   ├── models/
│   │   ├── match_predictor.py          # Modèle Poisson — prédiction scores
│   │   └── scorer_predictor.py         # Prédiction buteurs
│   └── simulation/
│       └── tournament.py               # Simulation du tournoi complet
└── notebooks/
    └── wc2026_exploration.ipynb        # Exploration & calibration du modèle
```

---

## 🚀 Installation locale

```bash
# 1. Cloner le repo
git clone https://github.com/Julie-Landrevie/world-cup-predictor.git
cd world-cup-predictor

# 2. Créer un environnement virtuel
python -m venv venv
source venv/bin/activate    # macOS / Linux

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Placer les données dans data/raw/
# → Télécharger international_results.csv depuis Kaggle

# 5. Lancer l'application
streamlit run app.py
```

---

## 🗺️ Roadmap

- [ ] Collecte et nettoyage des données historiques
- [ ] Implémentation du modèle de Poisson
- [ ] Calibration et validation du modèle (backtesting WC 2022)
- [ ] Prédiction des scores phase de groupes WC 2026
- [ ] Simulation de la phase finale
- [ ] Prédiction des buteurs par match
- [ ] Interface Streamlit publique
- [ ] Déploiement avant le coup d'envoi (juin 2026)

---

## 📄 Licence

MIT — voir [LICENSE](LICENSE)
