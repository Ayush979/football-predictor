# ⚽ Premier League Match Predictor — Story Mode 🏆  
**An AI-driven football prediction and storytelling system built in Python + Streamlit**

---

## 📖 Overview

This project predicts outcomes of **Premier League** matches using machine learning models trained on historical data.  
It then presents the predictions through an interactive **Streamlit web interface** enriched with visuals, team logos, head-to-head history, and a *Match Story Mode* that narrates classic rivalries and simulated match insights.

This system can:
- 📊 Train a model from historical Premier League data (past 25 years)
- 🧮 Predict upcoming fixtures (via CSV or UI)
- 🏟️ Provide interactive match insights and visualizations
- 🎨 Offer a creative *story mode* for any two selected teams
- 🎥 Link to classic match highlights on YouTube

---

## 🚀 Features

### 🧠 Model & Prediction Engine
- Uses **RandomForestClassifier** (scikit-learn)
- Trained on historical match data (`Date, HomeTeam, AwayTeam, HomeGoals, AwayGoals, Result`)
- Predicts probabilities for:
  - **Home Win**
  - **Draw**
  - **Away Win**

### 💻 Interactive Streamlit UI
- Team selection using clickable **team logos**
- Color-coded prediction cards using **official team colors**
- Shows **prediction probabilities** (e.g., 65% Home Win, 20% Draw, 15% Away Win)
- Displays recent and historical **match data**

### 🎬 Match Story Mode (Phase 5)
- Dynamically builds a **narrative** from head-to-head data
- Shows:
  - Total matches, average goals, win counts
  - Top 3 classic matches with context & YouTube highlight links
  - Interactive **goal timeline** (synthetic yet realistic)
  - Simulated top scorers (if player data unavailable)
- Designed to feel like a “football storytelling experience”

### 📦 Batch Prediction Mode
- Upload an `upcoming_fixtures.csv` file with columns:

- The system computes team stats and outputs:
- Automatically saves results to `data/predictions.csv`

---

## 🧰 Tech Stack

| Component | Library/Tool |
|------------|--------------|
| **Backend / Model** | Python, scikit-learn |
| **Web UI** | Streamlit |
| **Data Processing** | pandas, numpy |
| **Visualization** | Plotly, Matplotlib |
| **Storage** | CSV files |
| **Logos / Design** | Custom team logos (PNG format) |

---

## 📂 Project Structure

football-predictor/
│
├── data/
│ ├── historic_data.csv # 25 years of Premier League matches
│ ├── upcoming_fixtures.csv # User-provided upcoming fixtures
│ ├── predictions.csv # Model predictions (output)
│ └── team_logos/ # Folder of team logos
│
├── models/
│ └── match_predictor.pkl # Trained RandomForest model
│
├── src/
│ ├── train_model.py # Trains model on historical data
│ ├── predict_upcoming.py # Predicts matches in upcoming_fixtures.csv
│ └── app.py # Streamlit app (Story Mode + UI)
│
├── requirements.txt
└── README.md # This file

---

## ⚙️ Setup & Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/<your-username>/football-predictor.git
cd football-predictor

Developed with ❤️ using Python, Streamlit, and Football analytics.

This project is licensed under the MIT License — feel free to use and modify.

“Data tells you what happened. Models predict what will happen.
But football... football tells the story in between.” ⚽