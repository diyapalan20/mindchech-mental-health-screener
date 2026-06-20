from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import matplotlib
matplotlib.use('Agg')  # no GUI backend needed, safe for Flask servers
import matplotlib.pyplot as plt
import io
import base64
import sqlite3
import os
from datetime import datetime

app = Flask(__name__)

DB_PATH = 'history.db'

# Consistent color palette (teal theme to match Bootstrap UI)
TEAL       = '#0d9488'
TEAL_LIGHT = '#5eead4'
GREY       = '#94a3b8'
RED        = '#dc2626'


# ── Database setup ────────────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            age INTEGER,
            gender TEXT,
            family_history TEXT,
            work_interfere TEXT,
            remote_work TEXT,
            benefits TEXT,
            seek_help TEXT,
            anonymity TEXT,
            result TEXT,
            risk_percent REAL,
            confidence REAL
        )
    ''')
    conn.commit()
    conn.close()


def save_prediction(data):
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''
        INSERT INTO predictions
        (timestamp, age, gender, family_history, work_interfere, remote_work,
         benefits, seek_help, anonymity, result, risk_percent, confidence)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        data['timestamp'], data['age'], data['gender'], data['family_history'],
        data['work_interfere'], data['remote_work'], data['benefits'],
        data['seek_help'], data['anonymity'], data['result'],
        data['risk_percent'], data['confidence'],
    ))
    conn.commit()
    conn.close()


def get_all_predictions():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        'SELECT * FROM predictions ORDER BY id DESC LIMIT 100'
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_history_stats():
    conn = sqlite3.connect(DB_PATH)
    total = conn.execute('SELECT COUNT(*) FROM predictions').fetchone()[0]
    high_risk = conn.execute(
        "SELECT COUNT(*) FROM predictions WHERE result = 'High Risk'"
    ).fetchone()[0]
    conn.close()
    avg_risk_pct = round((high_risk / total) * 100, 1) if total > 0 else 0
    return {'total': total, 'high_risk': high_risk, 'high_risk_pct': avg_risk_pct}


init_db()


# ── Train model once at startup ──────────────────────────────────────────────
def train_model():
    raw_df = pd.read_csv('survey.csv')

    cols = ['Age', 'Gender', 'family_history', 'work_interfere',
            'remote_work', 'benefits', 'seek_help', 'anonymity', 'treatment']
    df = raw_df[cols].copy()

    # Clean Age
    df = df[(df['Age'] >= 15) & (df['Age'] <= 75)]

    # Clean Gender
    df['Gender'] = df['Gender'].str.lower()
    df['Gender'] = df['Gender'].apply(
        lambda x: 'male' if 'male' in str(x) else ('female' if 'female' in str(x) else 'other')
    )

    # Fill missing values
    df['work_interfere'] = df['work_interfere'].fillna('Never')
    df.dropna(inplace=True)

    # Keep a readable copy for charts BEFORE encoding
    chart_df = df.copy()

    # Encode
    le = LabelEncoder()
    for col in ['Gender', 'family_history', 'work_interfere',
                'remote_work', 'benefits', 'seek_help', 'anonymity', 'treatment']:
        df[col] = le.fit_transform(df[col])

    X = df.drop('treatment', axis=1)
    y = df['treatment']
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    acc = round(accuracy_score(y_test, clf.predict(X_test)) * 100, 1)
    return clf, acc, feature_names, chart_df


model, MODEL_ACCURACY, FEATURE_NAMES, CHART_DF = train_model()


# ── Helper: convert a matplotlib figure to a base64 <img> source ────────────
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=110, transparent=True)
    plt.close(fig)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    return f"data:image/png;base64,{encoded}"


def style_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#e2e8f0')
    ax.spines['bottom'].set_color('#e2e8f0')
    ax.tick_params(colors='#64748b', labelsize=9)
    ax.grid(axis='y', color='#e2e8f0', linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


# ── Build all dataset insight charts once at startup ─────────────────────────
def build_insight_charts(df):
    charts = {}

    # 1. Age distribution
    bins   = [15, 20, 25, 30, 35, 40, 45, 50, 75]
    labels = ['15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50+']
    age_buckets = pd.cut(df['Age'], bins=bins, labels=labels, right=False)
    age_dist = age_buckets.value_counts().reindex(labels).fillna(0).astype(int)

    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.bar(labels, age_dist.values, color=TEAL, width=0.6, zorder=3)
    ax.set_title('Age Distribution', fontsize=11, fontweight='bold', color='#1e293b', pad=10)
    style_axes(ax)
    charts['age'] = fig_to_base64(fig)

    # 2. Treatment rate by gender
    gender_treatment = (
        df.groupby('Gender')['treatment']
        .apply(lambda s: round((s == 'Yes').mean() * 100, 1))
        .reindex(['male', 'female', 'other']).fillna(0)
    )
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.bar(['Male', 'Female', 'Other'], gender_treatment.values,
           color=[TEAL, TEAL_LIGHT, GREY], width=0.5, zorder=3)
    ax.set_title('Treatment Rate by Gender (%)', fontsize=11, fontweight='bold', color='#1e293b', pad=10)
    ax.set_ylim(0, 100)
    style_axes(ax)
    charts['gender'] = fig_to_base64(fig)

    # 3. Work interference vs treatment
    interfere_order = ['Never', 'Rarely', 'Sometimes', 'Often']
    interfere_treatment = (
        df.groupby('work_interfere')['treatment']
        .apply(lambda s: round((s == 'Yes').mean() * 100, 1))
        .reindex(interfere_order).fillna(0)
    )
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(interfere_order, interfere_treatment.values, marker='o',
            color=TEAL, linewidth=2.5, markersize=7, zorder=3)
    ax.fill_between(interfere_order, interfere_treatment.values, color=TEAL, alpha=0.12, zorder=2)
    ax.set_title('Work Interference vs Treatment Rate (%)', fontsize=11, fontweight='bold', color='#1e293b', pad=10)
    ax.set_ylim(0, 100)
    style_axes(ax)
    charts['interfere'] = fig_to_base64(fig)

    # 4. Remote vs on-site
    remote_treatment = (
        df.groupby('remote_work')['treatment']
        .apply(lambda s: round((s == 'Yes').mean() * 100, 1))
        .reindex(['Yes', 'No']).fillna(0)
    )
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.barh(['Remote', 'On-site'], remote_treatment.values, color=[TEAL, GREY], height=0.5, zorder=3)
    ax.set_title('Remote vs On-site Treatment Rate (%)', fontsize=11, fontweight='bold', color='#1e293b', pad=10)
    ax.set_xlim(0, 100)
    style_axes(ax)
    charts['remote'] = fig_to_base64(fig)

    # 5. Family history impact
    family_treatment = (
        df.groupby('family_history')['treatment']
        .apply(lambda s: round((s == 'Yes').mean() * 100, 1))
        .reindex(['Yes', 'No']).fillna(0)
    )
    fig, ax = plt.subplots(figsize=(8, 2.6))
    ax.barh(['Has family history', 'No family history'], family_treatment.values,
            color=[RED, TEAL], height=0.5, zorder=3)
    ax.set_title('Family History Impact on Treatment Rate (%)', fontsize=11, fontweight='bold', color='#1e293b', pad=10)
    ax.set_xlim(0, 100)
    style_axes(ax)
    charts['family'] = fig_to_base64(fig)

    charts['total_responses']  = len(df)
    charts['interfere_order']  = interfere_order
    charts['interfere_values'] = interfere_treatment.tolist()

    return charts


INSIGHT_CHARTS = build_insight_charts(CHART_DF)


# ── Helper: build the "you vs average" comparison chart for the result page ─
def build_comparison_chart(work_interfere_choice, your_risk_percent):
    labels = INSIGHT_CHARTS['interfere_order']
    values = INSIGHT_CHARTS['interfere_values']
    colors = [TEAL if lbl == work_interfere_choice else '#cbd5e1' for lbl in labels]

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(labels, values, color=colors, width=0.55, zorder=3)
    ax.set_title('Treatment Rate by Work-Interference Level', fontsize=11, fontweight='bold', color='#1e293b', pad=10)
    ax.set_ylim(0, 100)
    style_axes(ax)

    # annotate the user's own bar
    idx = labels.index(work_interfere_choice)
    ax.annotate('You are here', xy=(idx, values[idx]), xytext=(idx, values[idx] + 12),
                ha='center', fontsize=9, fontweight='bold', color=TEAL,
                arrowprops=dict(arrowstyle='-|>', color=TEAL, lw=1.5))

    return fig_to_base64(fig)


# ── Routes ───────────────────────────────────────────────────────────────────
@app.route('/')
def home():
    return render_template('index.html', accuracy=MODEL_ACCURACY)


@app.route('/about')
def about():
    return render_template('about.html', accuracy=MODEL_ACCURACY)


@app.route('/insights')
def insights():
    return render_template(
        'insights.html',
        accuracy=MODEL_ACCURACY,
        total_responses=INSIGHT_CHARTS['total_responses'],
        chart_age=INSIGHT_CHARTS['age'],
        chart_gender=INSIGHT_CHARTS['gender'],
        chart_interfere=INSIGHT_CHARTS['interfere'],
        chart_remote=INSIGHT_CHARTS['remote'],
        chart_family=INSIGHT_CHARTS['family'],
    )


@app.route('/history')
def history():
    records = get_all_predictions()
    stats = get_history_stats()
    return render_template(
        'history.html',
        accuracy=MODEL_ACCURACY,
        records=records,
        stats=stats,
    )


@app.route('/predict', methods=['POST'])
def predict():
    errors = []

    # ── Validate Age ──
    age_raw = request.form.get('age', '').strip()
    if not age_raw:
        errors.append("Age is required.")
        age = None
    elif not age_raw.isdigit():
        errors.append("Age must be a whole number.")
        age = None
    else:
        age = int(age_raw)
        if age < 15 or age > 100:
            errors.append("Age must be between 15 and 100.")

    # ── Validate dropdowns ──
    gender         = request.form.get('gender', '')
    family_history = request.form.get('family_history', '')
    work_interfere = request.form.get('work_interfere', '')
    remote_work    = request.form.get('remote_work', '')
    benefits       = request.form.get('benefits', '')
    seek_help      = request.form.get('seek_help', '')
    anonymity      = request.form.get('anonymity', '')

    required_fields = {
        'Gender': gender,
        'Family history': family_history,
        'Work interference': work_interfere,
        'Remote work': remote_work,
        'Benefits': benefits,
        'Seek help': seek_help,
        'Anonymity': anonymity,
    }
    for label, value in required_fields.items():
        if not value:
            errors.append(f"{label} is required.")

    # ── If anything failed, send the user back to the form with messages ──
    if errors:
        return render_template('index.html', accuracy=MODEL_ACCURACY, errors=errors)

    # Encode exactly as training did
    gender_enc         = 1 if gender == 'male' else (0 if gender == 'female' else 2)
    family_history_enc = 1 if family_history == 'Yes' else 0
    work_interfere_enc = {'Never': 2, 'Rarely': 3, 'Sometimes': 4, 'Often': 1}[work_interfere]
    remote_work_enc    = 1 if remote_work == 'Yes' else 0
    benefits_enc       = {'Yes': 2, 'No': 0, "Don't know": 1}[benefits]
    seek_help_enc      = {'Yes': 2, 'No': 0, "Don't know": 1}[seek_help]
    anonymity_enc      = {'Yes': 2, 'No': 0, "Don't know": 1}[anonymity]

    features = pd.DataFrame(
        [[age, gender_enc, family_history_enc, work_interfere_enc,
          remote_work_enc, benefits_enc, seek_help_enc, anonymity_enc]],
        columns=FEATURE_NAMES
    )

    prediction   = model.predict(features)[0]
    proba        = model.predict_proba(features)[0]          # [prob_low, prob_high]
    confidence   = round(max(proba) * 100, 1)
    risk_percent = round(proba[1] * 100, 1)                  # probability of needing treatment

    if prediction == 1:
        result    = "High Risk"
        emoji     = "🔴"
        message   = "Based on your answers, you may benefit from speaking to a mental health professional."
        tips = [
            "Consider reaching out to a licensed therapist or counsellor.",
            "Talk to someone you trust — a friend, family member, or mentor.",
            "Practice stress-reduction techniques: deep breathing, journaling, or walking.",
            "iCall India Helpline: 9152987821 (free, confidential).",
        ]
        color = "danger"
    else:
        result    = "Low Risk"
        emoji     = "🟢"
        message   = "Your responses suggest a healthy mental state. Keep nurturing your well-being!"
        tips = [
            "Maintain regular sleep and exercise routines.",
            "Stay connected with friends and family.",
            "Check in with yourself regularly — mental health changes over time.",
            "Remember: seeking help early is always a sign of strength.",
        ]
        color = "success"

    comparison_chart = build_comparison_chart(work_interfere, risk_percent)

    # ── Save this prediction to the database ──
    save_prediction({
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'age': age,
        'gender': gender,
        'family_history': family_history,
        'work_interfere': work_interfere,
        'remote_work': remote_work,
        'benefits': benefits,
        'seek_help': seek_help,
        'anonymity': anonymity,
        'result': result,
        'risk_percent': risk_percent,
        'confidence': confidence,
    })

    return render_template(
        'result.html',
        result=result,
        emoji=emoji,
        message=message,
        tips=tips,
        color=color,
        confidence=confidence,
        risk_percent=risk_percent,
        accuracy=MODEL_ACCURACY,
        comparison_chart=comparison_chart,
    )


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)