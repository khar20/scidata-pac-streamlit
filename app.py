import os
import random

# ── Reproducibility: env vars MUST be set before any TF/numpy import ─────────
SEED = 42
os.environ["PYTHONHASHSEED"]        = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"]  = "1"   # disables non-deterministic TF ops
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"  # deterministic cuDNN (GPU)

import warnings
warnings.filterwarnings("ignore")

import numpy as np

# Reset all RNG sources in one place; call this before every model build
def set_seeds(seed=SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass

set_seeds()
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease · CRISP-DM",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── global styles ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* sidebar */
    [data-testid="stSidebar"] { background: #1a1a2e; }
    [data-testid="stSidebar"] * { color: #e0e0e0 !important; }
    [data-testid="stSidebar"] .stRadio label { font-size: 0.95rem; }

    /* metric cards */
    [data-testid="metric-container"] {
        background: #f8f9fc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 14px 18px;
    }

    /* section headers */
    .section-header {
        font-size: 1.05rem;
        font-weight: 700;
        color: #1e3a5f;
        border-left: 4px solid #3b82f6;
        padding-left: 10px;
        margin: 18px 0 10px 0;
    }

    /* badge pill */
    .badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .badge-blue  { background: #dbeafe; color: #1d4ed8; }
    .badge-green { background: #dcfce7; color: #166534; }
    .badge-red   { background: #fee2e2; color: #991b1b; }

    /* prediction box */
    .pred-positive {
        background: #fef2f2; border: 2px solid #ef4444;
        border-radius: 12px; padding: 18px; text-align: center;
    }
    .pred-negative {
        background: #f0fdf4; border: 2px solid #22c55e;
        border-radius: 12px; padding: 18px; text-align: center;
    }
    .pred-title { font-size: 1.4rem; font-weight: 800; margin-bottom: 6px; }
    .pred-sub   { font-size: 0.9rem; color: #555; }

    /* divider */
    hr { border: none; border-top: 1px solid #e2e8f0; margin: 20px 0; }

    /* hide streamlit default header decoration */
    #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── colour palette ────────────────────────────────────────────────────────────
COLORS = {
    "Reg. Logística":    "#FF6B6B",
    "Random Forest":     "#4ECDC4",
    "RF Optimizado":     "#2AB7CA",
    "Gradient Boosting": "#45B7D1",
    "GB Optimizado":     "#1A85A0",
    "DNN-MLP":           "#96CEB4",
    "DNN-Profunda":      "#CE93D8",
}
BLUE   = "#2196F3"
RED    = "#F44336"
PLOTLY_TEMPLATE = "plotly_white"

# ─────────────────────────────────────────────────────────────────────────────
# DATA & MODEL LOADING  (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Cargando dataset…")
def load_data():
    try:
        from ucimlrepo import fetch_ucirepo
        hd = fetch_ucirepo(id=45)
        df = pd.concat([hd.data.features, hd.data.targets], axis=1)
        df.columns = [
            "age","sex","cp","trestbps","chol","fbs","restecg",
            "thalach","exang","oldpeak","slope","ca","thal","target",
        ]
    except Exception:
        url = "https://raw.githubusercontent.com/dsrscientist/dataset1/master/heart_Disease.csv"
        df = pd.read_csv(url)
        df.columns = [
            "age","sex","cp","trestbps","chol","fbs","restecg",
            "thalach","exang","oldpeak","slope","ca","thal","target",
        ]
    df["target"] = (df["target"] > 0).astype(int)
    df.replace("?", np.nan, inplace=True)
    for col in ["ca", "thal"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col].fillna(df[col].median(), inplace=True)
    df.dropna(inplace=True)
    return df


@st.cache_resource(show_spinner="Entrenando modelos… (puede tardar 1-2 min)")
def train_models(df):
    from imblearn.over_sampling import SMOTE

    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)

    smote = SMOTE(random_state=42)
    X_bal, y_bal = smote.fit_resample(X_tr_sc, y_train)

    # ── ML models ─────────────────────────────────────────────────────────────
    lr = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    lr.fit(X_bal, y_bal)

    rf = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_split=4,
                                random_state=42, n_jobs=-1)
    rf.fit(X_bal, y_bal)

    gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                    max_depth=4, subsample=0.8, random_state=42)
    gb.fit(X_bal, y_bal)

    # ── GridSearch ────────────────────────────────────────────────────────────
    cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    gs_rf = GridSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1),
        {"n_estimators": [100, 200], "max_depth": [4, 6, 8], "min_samples_split": [2, 4]},
        cv=cv_inner, scoring="roc_auc", n_jobs=-1
    )
    gs_rf.fit(X_bal, y_bal)
    rf_best = gs_rf.best_estimator_

    gs_gb = GridSearchCV(
        GradientBoostingClassifier(random_state=42),
        {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "max_depth": [3, 4]},
        cv=cv_inner, scoring="roc_auc", n_jobs=-1
    )
    gs_gb.fit(X_bal, y_bal)
    gb_best = gs_gb.best_estimator_

    # ── Deep Learning (optional – skip if TF unavailable) ────────────────────
    nn_probs, nn2_probs = None, None
    nn_preds, nn2_preds = None, None
    nn_history, nn2_history = None, None

    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers, regularizers

        # keras.utils.set_random_seed resets Python, NumPy and TF seeds
        # together and enables TF deterministic ops at the Keras level (TF>=2.7)
        keras.utils.set_random_seed(SEED)

        def build_mlp(dim):
            mdl = keras.Sequential([
                layers.Input(shape=(dim,)),
                layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4)),
                layers.BatchNormalization(), layers.Dropout(0.3),
                layers.Dense(32, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4)),
                layers.BatchNormalization(), layers.Dropout(0.2),
                layers.Dense(16, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4)),
                layers.Dense(1, activation="sigmoid"),
            ])
            mdl.compile(optimizer=keras.optimizers.AdamW(learning_rate=5e-4, weight_decay=1e-3),
                        loss="binary_crossentropy",
                        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])
            return mdl

        def build_dnn(dim):
            inp = keras.Input(shape=(dim,))
            x = layers.Dense(64, kernel_regularizer=regularizers.l2(5e-3))(inp)
            x = layers.LeakyReLU(negative_slope=0.1)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.5)(x)
            x = layers.Dense(32, kernel_regularizer=regularizers.l2(5e-3))(x)
            x = layers.LeakyReLU(negative_slope=0.1)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.4)(x)
            x = layers.Dense(16, kernel_regularizer=regularizers.l2(5e-3))(x)
            x = layers.LeakyReLU(negative_slope=0.1)(x)
            x = layers.Dropout(0.3)(x)
            out = layers.Dense(1, activation="sigmoid")(x)
            mdl = keras.Model(inputs=inp, outputs=out)
            mdl.compile(optimizer=keras.optimizers.Adam(learning_rate=5e-4, weight_decay=5e-4),
                        loss="binary_crossentropy",
                        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])
            return mdl

        es  = keras.callbacks.EarlyStopping(monitor="val_auc", mode="max",
                                             patience=10, restore_best_weights=True, min_delta=1e-4)
        rlr = keras.callbacks.ReduceLROnPlateau(monitor="val_auc", mode="max",
                                                factor=0.5, patience=5, min_lr=1e-6)

        # Reset ALL four RNG sources immediately before weight initialisation
        set_seeds()
        keras.utils.set_random_seed(SEED)
        nn = build_mlp(X_bal.shape[1])
        h1 = nn.fit(X_bal, y_bal, validation_split=0.20, epochs=200, batch_size=64,
                    callbacks=[es, rlr], verbose=0)
        nn_probs  = nn.predict(X_te_sc, verbose=0).flatten()
        nn_preds  = (nn_probs >= 0.5).astype(int)
        nn_history = h1.history

        es2  = keras.callbacks.EarlyStopping(monitor="val_auc", mode="max",
                                              patience=12, restore_best_weights=True, min_delta=1e-4)
        rlr2 = keras.callbacks.ReduceLROnPlateau(monitor="val_auc", mode="max",
                                                  factor=0.5, patience=6, min_lr=1e-6)

        # Reset again before the second model so it is independent of the first
        set_seeds()
        keras.utils.set_random_seed(SEED)
        nn2 = build_dnn(X_bal.shape[1])
        h2  = nn2.fit(X_bal, y_bal, validation_split=0.20, epochs=200, batch_size=64,
                      callbacks=[es2, rlr2], verbose=0)
        nn2_probs  = nn2.predict(X_te_sc, verbose=0).flatten()
        nn2_preds  = (nn2_probs >= 0.5).astype(int)
        nn2_history = h2.history

    except Exception:
        pass  # DL silently skipped if TF not available

    # ── predictions ───────────────────────────────────────────────────────────
    preds = {
        "Reg. Logística":    (lr.predict(X_te_sc),    lr.predict_proba(X_te_sc)[:, 1]),
        "Random Forest":     (rf.predict(X_te_sc),    rf.predict_proba(X_te_sc)[:, 1]),
        "RF Optimizado":     (rf_best.predict(X_te_sc), rf_best.predict_proba(X_te_sc)[:, 1]),
        "Gradient Boosting": (gb.predict(X_te_sc),    gb.predict_proba(X_te_sc)[:, 1]),
        "GB Optimizado":     (gb_best.predict(X_te_sc), gb_best.predict_proba(X_te_sc)[:, 1]),
    }
    if nn_probs is not None:
        preds["DNN-MLP"]      = (nn_preds,  nn_probs)
        preds["DNN-Profunda"] = (nn2_preds, nn2_probs)

    # ── metrics table ─────────────────────────────────────────────────────────
    rows = []
    for name, (ypd, ypr) in preds.items():
        rows.append({
            "Modelo":     name,
            "Accuracy":   round(accuracy_score(y_test, ypd), 4),
            "Precision":  round(precision_score(y_test, ypd), 4),
            "Recall":     round(recall_score(y_test, ypd), 4),
            "F1-Score":   round(f1_score(y_test, ypd), 4),
            "AUC-ROC":    round(roc_auc_score(y_test, ypr), 4),
        })
    results = pd.DataFrame(rows).set_index("Modelo")

    # ── feature importance ────────────────────────────────────────────────────
    fi_df = pd.DataFrame({"Feature": X.columns,
                           "Importance": rf_best.feature_importances_}
                         ).sort_values("Importance", ascending=False)

    return dict(
        X=X, y=y, X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        X_tr_sc=X_tr_sc, X_te_sc=X_te_sc, X_bal=X_bal, y_bal=y_bal,
        scaler=scaler,
        lr=lr, rf=rf, gb=gb, rf_best=rf_best, gb_best=gb_best,
        preds=preds, results=results, fi_df=fi_df,
        nn_history=nn_history, nn2_history=nn2_history,
    )


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def metric_card(col, label, value, delta=None, suffix=""):
    with col:
        st.metric(label, f"{value:.4f}{suffix}", delta=delta)


def section(title):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)


def roc_figure(results_df, preds, y_test):
    fig = go.Figure()
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                  line=dict(color="lightgrey", dash="dash"))
    for name, (_, ypr) in preds.items():
        fpr, tpr, _ = roc_curve(y_test, ypr)
        auc = roc_auc_score(y_test, ypr)
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines", name=f"{name} (AUC={auc:.3f})",
            line=dict(color=COLORS.get(name, "#888"), width=2.5)
        ))
    fig.update_layout(
        xaxis_title="Tasa de Falsos Positivos",
        yaxis_title="Tasa de Verdaderos Positivos",
        legend=dict(x=0.55, y=0.05, font_size=11),
        template=PLOTLY_TEMPLATE, height=420, margin=dict(t=20, b=40)
    )
    return fig


def confusion_figure(y_test, y_pred, name):
    cm = confusion_matrix(y_test, y_pred)
    labels = ["Sin Enf.", "Con Enf."]
    fig = go.Figure(go.Heatmap(
        z=cm, x=labels, y=labels,
        colorscale="Blues", showscale=False,
        text=cm, texttemplate="%{text}",
        textfont=dict(size=20, color="white"),
    ))
    fig.update_layout(
        xaxis_title="Predicción", yaxis_title="Real",
        template=PLOTLY_TEMPLATE, height=320,
        margin=dict(t=20, b=40, l=60, r=20)
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🫀 Heart Disease")
    st.markdown("**Pipeline CRISP-DM**")
    st.markdown("---")
    page = st.radio(
        "Sección",
        [
            "Resumen Ejecutivo",
            "Exploración de Datos",
            "Modelos y Métricas",
            "Curvas ROC",
            "Matrices de Confusión",
            "Importancia de Variables",
            "Curvas de Entrenamiento DL",
            "Predicción Individual",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("Dataset: UCI Heart Disease (Cleveland)")
    st.caption("297 registros · 13 features · 5 modelos")


# ─────────────────────────────────────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────────────────────────────────────
df = load_data()
m  = train_models(df)
results  = m["results"]
preds    = m["preds"]
y_test   = m["y_test"]
fi_df    = m["fi_df"]
X        = m["X"]
scaler   = m["scaler"]


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: RESUMEN EJECUTIVO
# ─────────────────────────────────────────────────────────────────────────────
if page == "Resumen Ejecutivo":
    st.title("Pipeline de Análisis · Predicción de Enfermedad Cardíaca")
    st.caption("Metodología CRISP-DM · Heart Disease UCI (Cleveland)")

    st.markdown("---")

    # ── dataset stats ─────────────────────────────────────────────────────────
    section("Dataset")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Registros", f"{len(df):,}")
    c2.metric("Features", str(df.shape[1] - 1))
    c3.metric("Train", str(m["X_train"].shape[0]))
    c4.metric("Test",  str(m["X_test"].shape[0]))

    c1, c2, c3 = st.columns(3)
    c1.metric("Sin enfermedad", f"{(df['target']==0).sum()} ({(df['target']==0).mean()*100:.0f}%)")
    c2.metric("Con enfermedad", f"{(df['target']==1).sum()} ({(df['target']==1).mean()*100:.0f}%)")
    c3.metric("Modelos entrenados", str(len(preds)))

    st.markdown("---")

    # ── best model highlight ──────────────────────────────────────────────────
    section("Mejor modelo por métrica")
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"]
    cols = st.columns(len(metrics))
    for col, met in zip(cols, metrics):
        best_model = results[met].idxmax()
        best_val   = results[met].max()
        with col:
            st.markdown(f"**{met}**")
            st.markdown(
                f'<span class="badge badge-green">{best_model}</span><br>'
                f'<span style="font-size:1.3rem;font-weight:700">{best_val:.4f}</span>',
                unsafe_allow_html=True
            )

    st.markdown("---")

    # ── full results table ────────────────────────────────────────────────────
    section("Tabla comparativa de métricas")
    styled = (
        results.style
        .highlight_max(axis=0, color="#dcfce7")
        .format("{:.4f}")
    )
    st.dataframe(styled, use_container_width=True)

    st.markdown("---")

    # ── radar chart ───────────────────────────────────────────────────────────
    section("Radar de métricas por modelo")
    theta = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"]
    fig_radar = go.Figure()
    for name in results.index:
        vals = [results.loc[name, m] for m in theta]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=theta + [theta[0]],
            fill="toself", name=name,
            line=dict(color=COLORS.get(name, "#888"), width=2),
            opacity=0.7,
        ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0.7, 1.0])),
        template=PLOTLY_TEMPLATE, height=450,
        legend=dict(orientation="h", y=-0.15),
        margin=dict(t=20),
    )
    st.plotly_chart(fig_radar, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: EXPLORACIÓN DE DATOS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Exploración de Datos":
    st.title("Exploración de Datos (EDA)")

    # ── variable selector ─────────────────────────────────────────────────────
    section("Distribución de variables por clase")
    numeric_cols  = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    categoric_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
    all_feat = [c for c in df.columns if c != "target"]

    col_sel, col_type = st.columns([3, 1])
    feat = col_sel.selectbox("Variable", all_feat, index=0)

    df_plot = df.copy()
    df_plot["Diagnóstico"] = df_plot["target"].map({0: "Sin enfermedad", 1: "Con enfermedad"})

    if df[feat].nunique() > 10:
        fig_dist = px.histogram(
            df_plot, x=feat, color="Diagnóstico",
            barmode="overlay", opacity=0.7,
            color_discrete_map={"Sin enfermedad": BLUE, "Con enfermedad": RED},
            nbins=25, template=PLOTLY_TEMPLATE,
        )
    else:
        fig_dist = px.histogram(
            df_plot, x=feat.astype(str) if False else feat,
            color="Diagnóstico", barmode="group",
            color_discrete_map={"Sin enfermedad": BLUE, "Con enfermedad": RED},
            template=PLOTLY_TEMPLATE,
        )
    fig_dist.update_layout(height=380, margin=dict(t=20, b=40),
                           legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig_dist, use_container_width=True)

    st.markdown("---")

    # ── scatter ───────────────────────────────────────────────────────────────
    section("Relación entre variables")
    c1, c2 = st.columns(2)
    x_ax = c1.selectbox("Eje X", all_feat, index=0)
    y_ax = c2.selectbox("Eje Y", all_feat, index=7)  # thalach default

    fig_sc = px.scatter(
        df_plot, x=x_ax, y=y_ax, color="Diagnóstico", opacity=0.7,
        color_discrete_map={"Sin enfermedad": BLUE, "Con enfermedad": RED},
        template=PLOTLY_TEMPLATE,
    )
    fig_sc.update_layout(height=400, margin=dict(t=20, b=40),
                         legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig_sc, use_container_width=True)

    st.markdown("---")

    # ── correlation heatmap ───────────────────────────────────────────────────
    section("Matriz de correlación")
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    corr_lower = corr.where(~mask)

    fig_corr = px.imshow(
        corr_lower, color_continuous_scale="RdYlBu_r",
        zmin=-1, zmax=1, text_auto=".2f",
        template=PLOTLY_TEMPLATE,
    )
    fig_corr.update_layout(height=520, margin=dict(t=20))
    fig_corr.update_traces(textfont_size=10)
    st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("---")

    # ── outlier boxplots ──────────────────────────────────────────────────────
    section("Análisis de outliers (IQR)")
    fig_box = go.Figure()
    for col in numeric_cols:
        fig_box.add_trace(go.Box(
            y=df[col], name=col, boxpoints="outliers",
            marker_color="#3b82f6", line_color="#1d4ed8",
        ))
    fig_box.update_layout(height=380, template=PLOTLY_TEMPLATE,
                          margin=dict(t=20, b=40),
                          showlegend=False)
    st.plotly_chart(fig_box, use_container_width=True)

    # outlier table
    outlier_data = []
    for col in numeric_cols:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        n = int(((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)).sum())
        outlier_data.append({"Variable": col, "Q1": round(Q1,2), "Q3": round(Q3,2),
                              "IQR": round(IQR,2), "Outliers": n})
    st.dataframe(pd.DataFrame(outlier_data).set_index("Variable"), use_container_width=True)

    st.markdown("---")

    # ── stats table ───────────────────────────────────────────────────────────
    section("Estadísticas descriptivas")
    st.dataframe(df.describe().T.round(2), use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: MODELOS Y MÉTRICAS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Modelos y Métricas":
    st.title("Modelos y Métricas de Evaluación")

    # ── bar chart per metric ───────────────────────────────────────────────────
    section("Comparación visual por métrica")
    met_sel = st.selectbox("Métrica", ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"])

    df_bar = results[[met_sel]].reset_index()
    fig_bar = px.bar(
        df_bar, x="Modelo", y=met_sel,
        color="Modelo",
        color_discrete_map=COLORS,
        text=met_sel, template=PLOTLY_TEMPLATE,
    )
    fig_bar.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    fig_bar.update_layout(height=420, showlegend=False, margin=dict(t=20, b=60),
                          yaxis_range=[0.6, 1.05])
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")

    # ── grouped bar all metrics ────────────────────────────────────────────────
    section("Todas las métricas por modelo")
    metrics_cols = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"]
    df_melt = results[metrics_cols].reset_index().melt(
        id_vars="Modelo", var_name="Métrica", value_name="Score"
    )
    fig_grp = px.bar(
        df_melt, x="Métrica", y="Score", color="Modelo",
        barmode="group", color_discrete_map=COLORS,
        template=PLOTLY_TEMPLATE,
    )
    fig_grp.update_layout(height=420, margin=dict(t=20, b=60),
                          yaxis_range=[0.6, 1.05],
                          legend=dict(orientation="h", y=-0.3))
    st.plotly_chart(fig_grp, use_container_width=True)

    st.markdown("---")

    # ── per-model detail ──────────────────────────────────────────────────────
    section("Detalle por modelo")
    model_sel = st.selectbox("Selecciona modelo", list(preds.keys()))
    y_pred_s, y_prob_s = preds[model_sel]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy",  f"{accuracy_score(y_test, y_pred_s):.4f}")
    c2.metric("Precision", f"{precision_score(y_test, y_pred_s):.4f}")
    c3.metric("Recall",    f"{recall_score(y_test, y_pred_s):.4f}")
    c4.metric("F1-Score",  f"{f1_score(y_test, y_pred_s):.4f}")
    c5.metric("AUC-ROC",   f"{roc_auc_score(y_test, y_prob_s):.4f}")

    col_cm, col_roc = st.columns(2)
    with col_cm:
        st.caption("Matriz de Confusión")
        st.plotly_chart(confusion_figure(y_test, y_pred_s, model_sel),
                        use_container_width=True)
    with col_roc:
        st.caption("Curva ROC")
        fig_r = go.Figure()
        fig_r.add_shape(type="line", x0=0,y0=0,x1=1,y1=1,
                        line=dict(color="lightgrey", dash="dash"))
        fpr, tpr, _ = roc_curve(y_test, y_prob_s)
        fig_r.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                   line=dict(color=COLORS.get(model_sel,"#888"), width=2.5),
                                   name=model_sel))
        fig_r.update_layout(height=320, template=PLOTLY_TEMPLATE,
                             margin=dict(t=10, b=40),
                             xaxis_title="FPR", yaxis_title="TPR")
        st.plotly_chart(fig_r, use_container_width=True)

    st.markdown("---")

    # ── 5-fold cross-validation ────────────────────────────────────────────────
    section("Validación cruzada 5-Fold (modelos ML)")

    @st.cache_data(show_spinner="Calculando validación cruzada…")
    def get_cv(_models_dict, X_sc, y):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        out = {}
        for name, model in _models_dict.items():
            scores = cross_val_score(model, X_sc, y, cv=cv, scoring="roc_auc", n_jobs=-1)
            out[name] = scores
        return out

    X_full_sc = scaler.fit_transform(X)
    cv_models  = {
        "Reg. Logística":    m["lr"],
        "Random Forest":     m["rf"],
        "Gradient Boosting": m["gb"],
    }
    cv_res = get_cv(frozenset(cv_models.keys()), X_full_sc, m["y"])

    fig_cv = go.Figure()
    for name, scores in cv_res.items():
        fig_cv.add_trace(go.Box(
            y=scores, name=name, boxpoints="all",
            marker_color=COLORS.get(name, "#888"),
            line_color=COLORS.get(name, "#888"),
        ))
    fig_cv.add_hline(y=0.85, line_dash="dash", line_color="red",
                     annotation_text="Umbral 0.85", annotation_position="top right")
    fig_cv.update_layout(height=380, template=PLOTLY_TEMPLATE,
                         yaxis_title="AUC-ROC", showlegend=False,
                         margin=dict(t=20, b=40))
    st.plotly_chart(fig_cv, use_container_width=True)

    cv_table = pd.DataFrame({
        name: {"Media": round(s.mean(),4), "Desv. Std.": round(s.std(),4),
               "Min": round(s.min(),4), "Max": round(s.max(),4)}
        for name, s in cv_res.items()
    }).T
    st.dataframe(cv_table, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: CURVAS ROC
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Curvas ROC":
    st.title("Curvas ROC")

    section("Comparación de todos los modelos")
    st.plotly_chart(roc_figure(results, preds, y_test), use_container_width=True)

    st.markdown("---")
    section("AUC-ROC por modelo")
    auc_df = results[["AUC-ROC"]].sort_values("AUC-ROC", ascending=True).reset_index()
    fig_auc = px.bar(
        auc_df, x="AUC-ROC", y="Modelo", orientation="h",
        color="Modelo", color_discrete_map=COLORS,
        text="AUC-ROC", template=PLOTLY_TEMPLATE,
    )
    fig_auc.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    fig_auc.update_layout(height=380, showlegend=False,
                          xaxis_range=[0.8, 1.0], margin=dict(t=20, b=40))
    st.plotly_chart(fig_auc, use_container_width=True)

    st.markdown("---")
    section("Comparación de dos modelos")
    c1, c2 = st.columns(2)
    m1 = c1.selectbox("Modelo A", list(preds.keys()), index=0)
    m2 = c2.selectbox("Modelo B", list(preds.keys()), index=len(preds)-1)

    fig_cmp = go.Figure()
    fig_cmp.add_shape(type="line", x0=0,y0=0,x1=1,y1=1,
                      line=dict(color="lightgrey", dash="dash"))
    for name in [m1, m2]:
        _, ypr = preds[name]
        fpr, tpr, _ = roc_curve(y_test, ypr)
        auc = roc_auc_score(y_test, ypr)
        fig_cmp.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                     line=dict(color=COLORS.get(name,"#888"), width=3),
                                     name=f"{name} (AUC={auc:.4f})"))
    fig_cmp.update_layout(height=420, template=PLOTLY_TEMPLATE,
                          xaxis_title="Tasa de Falsos Positivos",
                          yaxis_title="Tasa de Verdaderos Positivos",
                          legend=dict(x=0.55, y=0.1),
                          margin=dict(t=20, b=40))
    st.plotly_chart(fig_cmp, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: MATRICES DE CONFUSIÓN
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Matrices de Confusión":
    st.title("Matrices de Confusión")

    model_names = list(preds.keys())
    rows = [model_names[i:i+2] for i in range(0, len(model_names), 2)]

    for row_models in rows:
        cols = st.columns(len(row_models))
        for col, name in zip(cols, row_models):
            y_pred_m, _ = preds[name]
            acc = accuracy_score(y_test, y_pred_m)
            rec = recall_score(y_test, y_pred_m)
            with col:
                section(name)
                st.caption(f"Accuracy: {acc:.3f}  ·  Recall: {rec:.3f}")
                st.plotly_chart(confusion_figure(y_test, y_pred_m, name),
                                use_container_width=True)

    st.markdown("---")
    st.info(
        "**Falsos Negativos (FN):** pacientes con enfermedad predichos como sanos. "
        "En contexto de tamizaje clínico, minimizar FN es prioritario aunque aumente el número de FP."
    )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: IMPORTANCIA DE VARIABLES
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Importancia de Variables":
    st.title("Importancia de Variables")

    # ── RF importance ─────────────────────────────────────────────────────────
    section("Random Forest Optimizado — Importancia MDI")
    fi_sorted = fi_df.sort_values("Importance", ascending=True)
    fig_fi = px.bar(
        fi_sorted, x="Importance", y="Feature", orientation="h",
        color="Importance", color_continuous_scale="RdYlGn",
        text="Importance", template=PLOTLY_TEMPLATE,
    )
    fig_fi.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig_fi.update_layout(height=460, showlegend=False,
                         coloraxis_showscale=False,
                         margin=dict(t=20, b=40, r=80))
    st.plotly_chart(fig_fi, use_container_width=True)

    st.markdown("---")

    # ── LR coefficients ───────────────────────────────────────────────────────
    section("Regresión Logística — Coeficientes (log-odds)")
    coef_df = pd.DataFrame({
        "Feature": X.columns,
        "Coeficiente": m["lr"].coef_[0]
    }).sort_values("Coeficiente")

    fig_coef = px.bar(
        coef_df, x="Coeficiente", y="Feature", orientation="h",
        color="Coeficiente", color_continuous_scale="RdBu",
        text="Coeficiente", template=PLOTLY_TEMPLATE,
    )
    fig_coef.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig_coef.update_layout(height=460, showlegend=False,
                           coloraxis_showscale=False,
                           margin=dict(t=20, b=40, r=80))
    st.plotly_chart(fig_coef, use_container_width=True)

    st.markdown("---")

    # ── SHAP (optional) ───────────────────────────────────────────────────────
    section("SHAP — Importancia Global (|SHAP| medio)")
    if st.button("Calcular valores SHAP", type="primary"):
        with st.spinner("Calculando SHAP…"):
            try:
                import shap
                explainer = shap.TreeExplainer(m["rf_best"])
                shap_vals = explainer(m["X_te_sc"])
                sv = shap_vals.values[:, :, 1] if hasattr(shap_vals, "values") and shap_vals.values.ndim == 3 else (shap_vals[1] if isinstance(shap_vals, list) else shap_vals.values)

                mean_shap = np.abs(sv).mean(axis=0)
                shap_df = pd.DataFrame({
                    "Feature": list(X.columns),
                    "SHAP_mean": mean_shap
                }).sort_values("SHAP_mean", ascending=True)

                fig_shap = px.bar(
                    shap_df, x="SHAP_mean", y="Feature", orientation="h",
                    color="SHAP_mean", color_continuous_scale="YlOrRd",
                    text="SHAP_mean", template=PLOTLY_TEMPLATE,
                )
                fig_shap.update_traces(texttemplate="%{text:.4f}", textposition="outside")
                fig_shap.update_layout(height=460, showlegend=False,
                                       coloraxis_showscale=False,
                                       xaxis_title="|SHAP value| medio",
                                       margin=dict(t=20, b=40, r=80))
                st.plotly_chart(fig_shap, use_container_width=True)

                # top 5
                st.markdown("**Top 5 variables más influyentes (SHAP):**")
                top5 = shap_df.tail(5).iloc[::-1]
                for _, row in top5.iterrows():
                    st.markdown(
                        f'<span class="badge badge-blue">{row["Feature"]}</span>'
                        f'  |SHAP| medio = **{row["SHAP_mean"]:.4f}**',
                        unsafe_allow_html=True
                    )
            except ImportError:
                st.warning("Instala SHAP: `pip install shap`")
            except Exception as e:
                st.error(f"Error calculando SHAP: {e}")
    else:
        st.caption("Haz clic para calcular los valores SHAP sobre el conjunto de prueba.")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: CURVAS DE ENTRENAMIENTO DL
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Curvas de Entrenamiento DL":
    st.title("Curvas de Entrenamiento — Deep Learning")

    h1 = m["nn_history"]
    h2 = m["nn2_history"]

    if h1 is None and h2 is None:
        st.warning(
            "TensorFlow no está disponible en este entorno. "
            "Instala TensorFlow para entrenar y visualizar los modelos de Deep Learning."
        )
    else:
        model_tab = st.radio("Modelo", ["DNN-MLP", "DNN-Profunda"], horizontal=True)
        hist = h1 if model_tab == "DNN-MLP" else h2
        color_tr = "#1976D2" if model_tab == "DNN-MLP" else "#7B1FA2"
        color_val = "#D32F2F" if model_tab == "DNN-MLP" else "#E91E63"

        if hist is None:
            st.warning(f"No hay historial para {model_tab}.")
        else:
            epochs = list(range(1, len(hist["loss"]) + 1))
            metrics_dl = [("loss", "Pérdida (Loss)"), ("accuracy", "Exactitud"),  ("auc", "AUC")]

            cols_dl = st.columns(3)
            for col, (key, label) in zip(cols_dl, metrics_dl):
                fig_dl = go.Figure()
                fig_dl.add_trace(go.Scatter(
                    x=epochs, y=hist[key], name="Entrenamiento",
                    line=dict(color=color_tr, width=2)
                ))
                fig_dl.add_trace(go.Scatter(
                    x=epochs, y=hist[f"val_{key}"], name="Validación",
                    line=dict(color=color_val, width=2, dash="dash")
                ))
                fig_dl.update_layout(
                    title=label, height=300,
                    template=PLOTLY_TEMPLATE,
                    legend=dict(orientation="h", y=-0.35, font_size=10),
                    margin=dict(t=40, b=60, l=40, r=20),
                    xaxis_title="Épocas",
                )
                col.plotly_chart(fig_dl, use_container_width=True)

            st.markdown("---")
            section("Tabla de métricas finales de entrenamiento")
            final_row = {k: round(v[-1], 4) for k, v in hist.items()}
            st.dataframe(
                pd.DataFrame([final_row]).rename(columns={
                    "loss": "Loss (train)", "val_loss": "Loss (val)",
                    "accuracy": "Accuracy (train)", "val_accuracy": "Accuracy (val)",
                    "auc": "AUC (train)", "val_auc": "AUC (val)",
                }),
                use_container_width=True
            )
            st.caption(f"Épocas totales entrenadas: {len(hist['loss'])}")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: PREDICCIÓN INDIVIDUAL
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Predicción Individual":
    st.title("Predicción Individual de Riesgo")
    st.caption("Introduce los valores clínicos del paciente para obtener la predicción de cada modelo.")

    st.markdown("---")
    section("Datos del paciente")

    with st.form("patient_form"):
        c1, c2, c3 = st.columns(3)

        age      = c1.slider("Edad (años)",             20, 80, 55)
        sex      = c1.selectbox("Sexo",                 ["Masculino (1)", "Femenino (0)"])
        cp       = c1.selectbox("Tipo dolor torácico (cp)", [0, 1, 2, 3], index=0,
                                 help="0=asintomático, 1=angina típica, 2=atípica, 3=no anginoso")
        trestbps = c1.slider("Presión arterial reposo (mmHg)", 90, 200, 130)
        chol     = c2.slider("Colesterol sérico (mg/dl)",      100, 600, 250)
        fbs      = c2.selectbox("Glucemia ayunas >120 (fbs)",  [0, 1], index=0)
        restecg  = c2.selectbox("ECG en reposo (restecg)",     [0, 1, 2], index=0)
        thalach  = c2.slider("Frec. cardíaca máx. (thalach)", 60, 220, 150)
        exang    = c3.selectbox("Angina ejercicio (exang)",    [0, 1], index=0)
        oldpeak  = c3.slider("Depresión ST (oldpeak)",         0.0, 6.0, 1.0, step=0.1)
        slope    = c3.selectbox("Pendiente ST (slope)",        [0, 1, 2], index=1)
        ca       = c3.selectbox("Vasos coloreados (ca)",       [0, 1, 2, 3], index=0)
        thal     = c3.selectbox("Talasemia (thal)",            [1, 2, 3], index=0,
                                 help="1=normal, 2=defecto fijo, 3=reversible")

        submitted = st.form_submit_button("Predecir", type="primary", use_container_width=True)

    if submitted:
        sex_val = 1 if "Masculino" in sex else 0
        patient = np.array([[age, sex_val, cp, trestbps, chol, fbs, restecg,
                              thalach, exang, oldpeak, slope, ca, thal]])
        patient_sc = scaler.transform(patient)

        st.markdown("---")
        section("Resultados de predicción")

        ml_models = {
            "Reg. Logística":    m["lr"],
            "Random Forest":     m["rf"],
            "RF Optimizado":     m["rf_best"],
            "Gradient Boosting": m["gb"],
            "GB Optimizado":     m["gb_best"],
        }

        pred_rows = []
        all_probs = []
        for name, model in ml_models.items():
            prob = model.predict_proba(patient_sc)[0, 1]
            pred = int(prob >= 0.5)
            all_probs.append(prob)
            pred_rows.append({
                "Modelo": name,
                "Probabilidad (Con Enf.)": f"{prob:.4f}",
                "Predicción": "Con Enfermedad" if pred == 1 else "Sin Enfermedad",
            })

        pred_df = pd.DataFrame(pred_rows)
        positive_votes = sum(1 for r in pred_rows if r["Predicción"] == "Con Enfermedad")
        consensus_prob = np.mean(all_probs)

        c_box, c_table = st.columns([1, 2])
        with c_box:
            if positive_votes >= 3:
                st.markdown(
                    f'<div class="pred-positive">'
                    f'<div class="pred-title" style="color:#dc2626">CON ENFERMEDAD</div>'
                    f'<div class="pred-sub">{positive_votes}/5 modelos positivos</div>'
                    f'<div style="font-size:2rem;font-weight:800;margin-top:8px">{consensus_prob:.1%}</div>'
                    f'<div class="pred-sub">probabilidad promedio</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="pred-negative">'
                    f'<div class="pred-title" style="color:#16a34a">SIN ENFERMEDAD</div>'
                    f'<div class="pred-sub">{5-positive_votes}/5 modelos negativos</div>'
                    f'<div style="font-size:2rem;font-weight:800;margin-top:8px">{consensus_prob:.1%}</div>'
                    f'<div class="pred-sub">probabilidad promedio</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
        with c_table:
            st.dataframe(pred_df.set_index("Modelo"), use_container_width=True)

        st.markdown("---")

        # probability gauge bar
        section("Probabilidad de enfermedad por modelo")
        prob_data = [
            {"Modelo": r["Modelo"],
             "Probabilidad": float(r["Probabilidad (Con Enf.)"])}
            for r in pred_rows
        ]
        fig_prob = px.bar(
            pd.DataFrame(prob_data), x="Modelo", y="Probabilidad",
            color="Modelo", color_discrete_map=COLORS,
            text="Probabilidad", template=PLOTLY_TEMPLATE,
        )
        fig_prob.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig_prob.add_hline(y=0.5, line_dash="dash", line_color="grey",
                           annotation_text="Umbral 0.5")
        fig_prob.update_layout(height=400, showlegend=False,
                               yaxis_range=[0, 1.1],
                               margin=dict(t=20, b=60))
        st.plotly_chart(fig_prob, use_container_width=True)

        st.warning(
            "Esta herramienta es de apoyo académico y no sustituye el criterio médico profesional. "
            "Cualquier resultado debe ser interpretado por un especialista."
        )
