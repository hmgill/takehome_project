"""Image Explorer. Run with: python -m streamlit run app.py  (needs Streamlit >= 1.42)"""

import base64
import hashlib
import html
import importlib
import io
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import linear_shap

linear_shap = importlib.reload(linear_shap)
from image_pipeline import (
    APP_DATABASE,
    EXCLUDED_SPLIT,
    EXTERNAL_SPLIT,
    MODEL_SPLITS,
    MAX_FILE_BYTES,
    catalog,
    connect,
    default_classifier,
    ingest_bytes,
    ingest_directory,
    ingest_npz,
)
from project_config import PATHS

st.set_page_config(page_title="Image Explorer", page_icon="🔎", layout="wide")

# ---------------------------------------------------------------------------
# Look & feel
# ---------------------------------------------------------------------------
ACCENT = "#2563eb"
GALLERY_COLS = 8
PAGE_SIZES = [16, 32, 64]
LIST_SIZE = 100            # rows per page in the sidebar list
VIEW_SIZE = 336            # 12x for 28x28 sources
POS_RGB = (220, 38, 38)    # raises pneumonia score
NEG_RGB = (37, 99, 235)    # lowers pneumonia score

# width="stretch" replaced use_container_width in newer Streamlit releases.
_ver = tuple(int(p) for p in st.__version__.split(".")[:2] if p.isdigit())
STRETCH = {"width": "stretch"} if _ver >= (1, 50) else {"use_container_width": True}

# Clickable cards: each card is a keyed container holding an HTML preview plus a
# transparent Streamlit button stretched over it, so the whole card is the click target.
CLICKABLE = ["gcard_", "gsel_", "sbrow_", "sbsel_"]


def under(suffix=""):
    """Selector list for every clickable card/bar, with `suffix` applied to each item."""
    return ", ".join(f'[class*="st-key-{p}"]{suffix}' for p in CLICKABLE)


st.markdown(
    f"""<style>
.stApp {{background:#f7f8fa;}}
.block-container {{padding-top:2rem; padding-bottom:2rem; max-width:1500px;}}
h1 {{font-weight:650; letter-spacing:-.02em; color:#0f172a; margin-bottom:.25rem;}}
h3 {{color:#0f172a; font-weight:600;}}
[data-testid="stSidebar"] {{background:#ffffff; border-right:1px solid #e5e7eb;}}
[data-testid="stMetric"] {{background:#fff; border:1px solid #e5e7eb; border-radius:10px;
  padding:10px 16px;}}
[data-testid="stMetricValue"] {{color:{ACCENT}; font-size:1.5rem;}}
[data-testid="stVerticalBlockBorderWrapper"] {{background:#fff;}}
.stButton button {{border-radius:8px;}}
.stTabs [data-baseweb="tab-list"] {{gap:20px; border-bottom:1px solid #e5e7eb;}}
.pill {{display:inline-block; background:#eff4ff; color:{ACCENT}; border:1px solid #dbe5fe;
  border-radius:999px; padding:2px 12px; font-weight:600; font-size:.9rem;}}
.muted {{color:#64748b; font-size:.9rem; margin-left:.5rem;}}
.legend {{display:flex; gap:10px; flex-wrap:wrap; margin:2px 0 6px;}}
.legend .item {{display:flex; align-items:center; gap:8px; background:#fff;
  border:1px solid #dbe1ea; border-radius:8px; padding:6px 12px; font-size:.84rem;
  color:#334155;}}
.legend .swatch {{width:16px; height:16px; border-radius:4px;
  border:1px solid rgba(15,23,42,.25);}}

/* selected-image card: slightly darker panel so it reads as separate from the gallery */
.st-key-selected_card {{margin-top:1.5rem; background:#e9edf3 !important;
  border-color:#d5dce6 !important;}}
.st-key-selected_card [data-testid="stExpander"] details {{background:#fff;}}

/* shared click-overlay mechanics */
{under()} {{position:relative !important; gap:0 !important; cursor:pointer;}}
{under(" [data-testid='stElementContainer']:has(.stButton)")} {{position:absolute !important;
  inset:0; width:100% !important; height:100% !important; z-index:2; margin:0;}}
{under(" .stButton")}, {under(" .stButton > div")} {{width:100%; height:100%;}}
{under(" .stButton button")} {{width:100%; height:100%; opacity:0; cursor:pointer;}}
{under(" [data-testid='stMarkdownContainer']")} {{margin:0;}}
{under(" img")} {{transition:opacity .12s ease;}}

/* gallery cards */
.gc {{position:relative;}}
.gc .miss {{position:absolute; top:10px; right:10px; width:11px; height:11px; border-radius:50%;
  background:#dc2626; border:2px solid #fff; box-shadow:0 0 0 1px rgba(15,23,42,.2);}}
.cm {{background:#fff; border:1px solid #e5e7eb; border-radius:10px; padding:10px 12px;}}
.cm-title {{font-weight:600; color:#0f172a; margin-bottom:6px; text-transform:capitalize;}}
.cm table {{width:100%; border-collapse:separate; border-spacing:4px; font-size:.82rem;}}
.cm th {{color:#64748b; font-weight:500; text-align:left; border:none; padding:2px 4px;}}
.cm td {{text-align:center; font-weight:600; color:#0f172a; border:none; border-radius:6px;
  padding:10px 4px;}}
.pill.bad {{background:#fef2f2; color:#b91c1c; border-color:#fecaca; margin-left:.4rem;}}
.pill.good {{background:#f0fdf4; color:#15803d; border-color:#bbf7d0; margin-left:.4rem;}}
.gc {{background:#fff; border:1px solid #e5e7eb; border-radius:10px; padding:6px;
  transition:border-color .12s ease, background .12s ease;}}
.gc img {{display:block; width:100%; aspect-ratio:1; object-fit:cover; border-radius:6px;
  background:#f1f5f9;}}
.gc .nm {{font-size:.74rem; color:#334155; margin-top:5px; white-space:nowrap; overflow:hidden;
  text-overflow:ellipsis; text-align:center;}}
[class*="st-key-gcard_"]:hover .gc {{border-color:#93b4f5;}}
[class*="st-key-gcard_"]:hover img {{opacity:.72;}}
[class*="st-key-gsel_"] .gc {{border:2px solid {ACCENT}; background:#eff4ff; padding:5px;}}
[class*="st-key-gsel_"] img {{opacity:.6;}}
[class*="st-key-gsel_"] .nm {{color:{ACCENT}; font-weight:600;}}

/* sidebar bars */
.st-key-sidebar_list {{max-height:calc(100vh - 215px); overflow-y:auto; gap:4px !important;
  padding-right:4px;}}
.sb {{display:flex; align-items:center; gap:10px; padding:5px 8px; border-radius:8px;
  border:1px solid transparent; transition:background .12s ease;}}
.sb img {{width:34px; height:34px; border-radius:5px; object-fit:cover; flex:none;
  background:#f1f5f9;}}
.sb .nm {{flex:1; font-size:.84rem; color:#1e293b; white-space:nowrap; overflow:hidden;
  text-overflow:ellipsis;}}
.sb .cl {{font-size:.72rem; color:#64748b;}}
[class*="st-key-sbrow_"]:hover .sb {{background:#f1f5f9;}}
[class*="st-key-sbrow_"]:hover img {{opacity:.72;}}
[class*="st-key-sbsel_"] .sb {{background:#eff4ff; border-color:#bfd2fb;}}
[class*="st-key-sbsel_"] img {{opacity:.6;}}
[class*="st-key-sbsel_"] .nm {{color:{ACCENT}; font-weight:600;}}

/* keyboard focus stays visible even though the button itself is transparent */
{under(":has(button:focus-visible)")} {{outline:2px solid {ACCENT}; outline-offset:2px;
  border-radius:10px;}}
</style>""",
    unsafe_allow_html=True,
)
st.title("Image Explorer")

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
FILTER_DEFAULTS = {
    "search": "",
    "statuses": [],
    "classes": [],
    "formats": [],
    "splits": [],
    "confidence": 0.0,
    "unclassified": True,
    "min_width": 0,
    "min_height": 0,
    "max_mb": 25.0,
    "duplicates": False,
    "sort": "filename",
    "descending": False,
    "outcomes": [],
}
for _k, _v in {**FILTER_DEFAULTS, "page": 1, "list_page": 1, "page_size": PAGE_SIZES[1],
               "shap_opacity": 0.6, "shap_on": True}.items():
    st.session_state.setdefault(_k, _v)


def clear_filters():
    for key, value in FILTER_DEFAULTS.items():
        st.session_state[key] = value
    st.session_state["page"] = 1
    st.session_state["list_page"] = 1


def choose(path, idx):
    """Select an image and bring both the gallery and the sidebar list to it."""
    st.session_state["selected_path"] = path
    st.session_state["page"] = idx // st.session_state["page_size"] + 1
    st.session_state["list_page"] = idx // LIST_SIZE + 1


def set_state(key, value):
    st.session_state[key] = value


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------
def fetch_thumbnails(con, image_ids):
    ids = list(dict.fromkeys(image_ids))
    out = {}
    for i in range(0, len(ids), 500):
        chunk = ids[i : i + 500]
        rows = con.execute(
            f"SELECT image_id, thumbnail FROM images WHERE image_id IN ({','.join('?' * len(chunk))})",
            chunk,
        ).fetchall()
        out.update({r[0]: r[1] for r in rows if r[1]})
    return out


def to_uri(img):
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


BLANK = to_uri(Image.new("RGB", (8, 8), (241, 245, 249)))


def tile(blob, size):
    img = Image.open(io.BytesIO(blob)).convert("RGB")
    return ImageOps.pad(img, (size, size), color=(241, 245, 249))


@st.cache_data(show_spinner=False, max_entries=8)
def list_icons(db_path, image_ids, stamp):
    con2 = connect(Path(db_path))
    try:
        blobs = fetch_thumbnails(con2, image_ids)
    finally:
        con2.close()
    icons = {}
    for image_id, blob in blobs.items():
        try:
            icons[image_id] = to_uri(tile(blob, 56))
        except Exception:
            pass
    return icons


def fit(img, size=VIEW_SIZE):
    img = img.convert("RGB")
    scale = size / max(img.size)
    return img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))),
                      Image.LANCZOS)


def features_to_image(x):
    x = np.asarray(x, dtype=float).ravel()
    side = math.isqrt(x.size)
    if side * side != x.size:
        return None
    grid = x.reshape(side, side)
    span = grid.max() - grid.min() or 1.0
    return Image.fromarray(((grid - grid.min()) / span * 255).astype(np.uint8))


def shap_overlay(base, grid, opacity):
    vmax = float(np.abs(grid).max()) or 1.0
    norm = grid / vmax
    rgba = np.zeros((*grid.shape, 4), dtype=np.uint8)
    rgba[norm >= 0, :3] = POS_RGB
    rgba[norm < 0, :3] = NEG_RGB
    rgba[..., 3] = (np.abs(norm) ** 0.7 * 255 * opacity).astype(np.uint8)
    heat = Image.fromarray(rgba, "RGBA").resize(base.size, Image.BILINEAR)
    return Image.alpha_composite(base.convert("RGBA"), heat).convert("RGB")


def get_shap(record, con, classifier):
    """Linear SHAP is cheap, so compute it on selection and cache per image/model."""
    if classifier is None or record.processing_status != "complete":
        return None, None, None
    key = (record.image_id, record.model_version, record.pneumonia_score,
           classifier.version, linear_shap.EXPLANATION_VERSION)
    cache = st.session_state.setdefault("shap_cache", {})
    if key not in cache:
        try:
            features = con.execute(
                "SELECT features FROM images WHERE image_id=?", (record.image_id,)
            ).fetchone()[0]
            X, explanation = linear_shap.explain_catalog_image(
                classifier, features, record.model_version, record.pneumonia_score
            )
            cache[key] = (X, explanation, None)
        except Exception as exc:
            cache[key] = (None, None, str(exc))
        if len(cache) > 64:
            cache.pop(next(iter(cache)))
    return cache[key]


def render_view(base, X, explanation, opacity):
    if base is None and X is not None:
        base = features_to_image(X[0])
    if base is None:
        return None
    base = fit(base)
    if explanation is None or opacity is None or opacity <= 0:
        return base
    values = np.asarray(explanation.values[0], dtype=float).ravel()
    side = math.isqrt(values.size)
    if side * side != values.size:
        return fit(Image.open(io.BytesIO(linear_shap.explanation_png(X[0], explanation))))
    return shap_overlay(base, values.reshape(side, side), opacity)


def meta_rows(fields, prefix=""):
    rows = []
    for key, value in fields.items():
        if value is None or (isinstance(value, float) and math.isnan(value)) or value == "":
            value = "—"
        elif isinstance(value, float):
            value = f"{value:.6g}"
        rows.append({"Field": prefix + str(key).replace("_", " "), "Value": str(value)})
    return rows


def metadata_tables(record):
    rows = meta_rows(json.loads(record.drop(labels=["exif"]).to_json()))
    try:
        exif = json.loads(record.exif) if isinstance(record.exif, str) and record.exif else {}
    except ValueError:
        exif = {}
    if isinstance(exif, dict):
        rows += meta_rows(exif, prefix="EXIF ")
    half = math.ceil(len(rows) / 2)
    config = {"Field": st.column_config.TextColumn("Field", width="small"),
              "Value": st.column_config.TextColumn("Value", width="large")}
    for col, chunk in zip(st.columns(2, gap="medium"), (rows[:half], rows[half:])):
        if chunk:
            col.dataframe(pd.DataFrame(chunk), hide_index=True, column_config=config,
                          height=35 * len(chunk) + 38, **STRETCH)


SHAP_HELP = (
    "SHAP (SHapley Additive exPlanations) splits a model's prediction into a contribution "
    "from each input feature, here each pixel. It answers: which pixels pushed this image's "
    "pneumonia score up or down, compared with a typical image? Values are in log-odds and "
    "add up to the model's output. Hover over the image to read a pixel's value."
)


def interactive_view(composed, explanation):
    """Show the image with per-pixel SHAP values on hover. Returns False if not possible."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return False
    values = np.asarray(explanation.values[0], dtype=float).ravel()
    side = math.isqrt(values.size)
    if side * side != values.size:
        return False
    grid = values.reshape(side, side)
    w, h = composed.size
    cols, rows = np.meshgrid(np.arange(side), np.arange(side))
    buf = io.BytesIO()
    composed.save(buf, "PNG")
    fig = go.Figure()
    fig.add_trace(go.Image(source="data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(),
                           hoverinfo="skip"))
    fig.add_trace(go.Heatmap(
        z=grid, x0=w / side / 2, dx=w / side, y0=h / side / 2, dy=h / side,
        customdata=np.dstack([cols, rows]),
        colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]], showscale=False,
        hovertemplate="Pixel (%{customdata[0]}, %{customdata[1]})<br>"
                      "SHAP <b>%{z:+.4f}</b> log-odds<extra></extra>",
    ))
    fig.update_xaxes(visible=False, range=[0, w], fixedrange=True)
    fig.update_yaxes(visible=False, range=[h, 0], fixedrange=True, scaleanchor="x")
    fig.update_layout(width=w, height=h, margin=dict(l=0, r=0, t=0, b=0),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      hoverlabel=dict(bgcolor="#ffffff", bordercolor="#cbd5e1",
                                      font=dict(color="#0f172a", size=12)))
    size = {"width": "content"} if _ver >= (1, 50) else {"use_container_width": False}
    st.plotly_chart(fig, config={"displayModeBar": False, "scrollZoom": False},
                    key="shap_view", **size)
    return True


OUTCOMES = ["correct", "false positive", "false negative", "unlabeled"]
SPLIT_ORDER = ["train", "val", "validation", "test", EXCLUDED_SPLIT, EXTERNAL_SPLIT]
SPLIT_HELP = (
    "train/val/test: the deduplicated partition the model was trained and "
    f"evaluated on. {EXCLUDED_SPLIT}: dataset copies removed as duplicates. "
    f"{EXTERNAL_SPLIT}: images added from outside the dataset."
)


def split_sort_key(value):
    return (SPLIT_ORDER.index(value) if value in SPLIT_ORDER else 99, value)
SPLIT_COLORS = {"train": "#94a3b8", "val": "#60a5fa", "validation": "#60a5fa", "test": ACCENT,
                "all": "#0f172a"}


def to_class(value):
    """Normalise a ground-truth label (0/1 or text) to 'normal' / 'pneumonia'."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip().lower()
    if text in {"1", "1.0", "pneumonia", "positive", "true"}:
        return "pneumonia"
    if text in {"0", "0.0", "normal", "negative", "false"}:
        return "normal"
    return text or None


def add_outcomes(df):
    df = df.copy()
    df["true_class"] = df["label"].map(to_class) if "label" in df else None
    pred = df["predicted_class"].astype("string").str.lower()
    truth = df["true_class"].astype("string")
    outcome = pd.Series("unlabeled", index=df.index, dtype="object")
    known = truth.notna() & pred.notna()
    outcome[known & (pred == truth)] = "correct"
    outcome[known & (pred == "pneumonia") & (truth == "normal")] = "false positive"
    outcome[known & (pred == "normal") & (truth == "pneumonia")] = "false negative"
    df["outcome"] = outcome
    df["misclassified"] = outcome.isin(["false positive", "false negative"])
    return df


def auc_score(y, score):
    pos, neg = int(y.sum()), int((~y).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    ranks = pd.Series(score).rank(method="average").to_numpy()
    return float((ranks[y].sum() - pos * (pos + 1) / 2) / (pos * neg))


def split_metrics(df):
    """Binary metrics (pneumonia = positive) for one set of labeled, classified images."""
    y = (df["true_class"] == "pneumonia").to_numpy()
    p = (df["predicted_class"].str.lower() == "pneumonia").to_numpy()
    tp, tn = int((y & p).sum()), int((~y & ~p).sum())
    fp, fn = int((~y & p).sum()), int((y & ~p).sum())
    div = lambda a, b: a / b if b else float("nan")
    sens, spec, prec = div(tp, tp + fn), div(tn, tn + fp), div(tp, tp + fp)
    score_col = "pneumonia_score" if "pneumonia_score" in df else None
    auc = auc_score(y, df[score_col].to_numpy()) if score_col and df[score_col].notna().all() else float("nan")
    return {
        "Accuracy": div(tp + tn, len(df)),
        "Balanced accuracy": (sens + spec) / 2,
        "Sensitivity": sens,
        "Specificity": spec,
        "Precision": prec,
        "F1": div(2 * prec * sens, prec + sens),
        "AUC": auc,
    }, {"TP": tp, "FN": fn, "FP": fp, "TN": tn}


def metrics_tab(df):
    labeled = df[df["outcome"] != "unlabeled"]
    # Excluded copies duplicate images already counted in train/val/test;
    # external images are outside the partition. Neither belongs in these metrics.
    outside = ~labeled["split"].isin(MODEL_SPLITS)
    if outside.any():
        st.caption(f"{int(outside.sum())} labeled {EXCLUDED_SPLIT}/{EXTERNAL_SPLIT} "
                   "images are not included in these metrics.")
    labeled = labeled[~outside]
    if labeled.empty:
        st.info("No images have both a ground-truth label and a prediction.")
        return
    splits = sorted(labeled["split"].dropna().unique(), key=split_sort_key)
    groups = [(str(sp), labeled[labeled["split"] == sp]) for sp in splits]
    if labeled["split"].isna().any() or len(groups) != 1:
        groups.append(("all", labeled))
    results = {name: split_metrics(g) for name, g in groups}

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown("**Performance by set**",
                    help="Pneumonia is the positive class. Sensitivity = recall on pneumonia; "
                         "specificity = recall on normal; AUC uses the pneumonia score, "
                         "the other metrics use the predicted class.")
        zoom = st.toggle("Zoom axis", key="radar_zoom", value=True,
                         help="Start the radial axis near the lowest value instead of 0.")
        try:
            import plotly.graph_objects as go
        except ImportError:
            go = None
        if go is not None:
            names = list(next(iter(results.values()))[0].keys())
            fig = go.Figure()
            low = 1.0
            for name, (m, _) in results.items():
                vals = [m[k] if not math.isnan(m[k]) else 0 for k in names]
                low = min(low, min(vals))
                color = SPLIT_COLORS.get(name, "#a78bfa")
                fig.add_trace(go.Scatterpolar(
                    r=vals + vals[:1], theta=names + names[:1], name=name,
                    line=dict(color=color, width=2.5), marker=dict(size=6), fill="toself",
                    hovertemplate="%{theta}: %{r:.3f}<extra>" + name + "</extra>",
                ))
            lo = max(0.0, math.floor((low - 0.05) * 20) / 20) if zoom else 0.0
            fig.update_layout(
                polar=dict(bgcolor="#ffffff",
                           radialaxis=dict(range=[lo, 1], tickformat=".2f", gridcolor="#e5e7eb",
                                           tickfont=dict(size=10, color="#64748b")),
                           angularaxis=dict(gridcolor="#e5e7eb", tickfont=dict(size=12, color="#334155"))),
                legend=dict(orientation="h", y=-0.12, x=0.5, xanchor="center"),
                margin=dict(l=70, r=70, t=30, b=40), height=520,
                paper_bgcolor="rgba(0,0,0,0)",
            )
            for tr in fig.data:
                tr.fillcolor = _alpha(tr.line.color, .06)
            st.plotly_chart(fig, config={"displayModeBar": False}, key="radar", **STRETCH)
    with right:
        table = pd.DataFrame({name: {k: ("—" if math.isnan(v) else f"{v:.3f}") for k, v in m.items()}
                              for name, (m, _) in results.items()})
        table.loc["Images"] = [f"{len(g):,}" for _, g in groups]
        st.markdown("**Metrics**")
        st.dataframe(table, **STRETCH)

    st.markdown("**Confusion matrices**", help="Rows are the true class, columns the prediction.")
    cols = st.columns(len(results), gap="medium")
    for col, (name, (_, cm)) in zip(cols, results.items()):
        total = max(1, sum(cm.values()))
        cell = lambda v, good: (
            f"<td style='background:{_alpha(ACCENT if good else '#dc2626', .08 + .5 * v / total)}'>"
            f"{v:,}</td>")
        col.markdown(
            f"<div class='cm'><div class='cm-title'>{html.escape(name)}</div><table>"
            f"<tr><th></th><th>Pred. pneumonia</th><th>Pred. normal</th></tr>"
            f"<tr><th>True pneumonia</th>{cell(cm['TP'], True)}{cell(cm['FN'], False)}</tr>"
            f"<tr><th>True normal</th>{cell(cm['FP'], False)}{cell(cm['TN'], True)}</tr>"
            f"</table></div>",
            unsafe_allow_html=True,
        )


def _alpha(hex_color, a):
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"


def path_key(path):
    return hashlib.sha256(path.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Selected-image card
# ---------------------------------------------------------------------------
def selected_card(record, idx, paths, con, classifier):
    blob = con.execute(
        "SELECT thumbnail FROM images WHERE image_id=?", (record.image_id,)
    ).fetchone()
    base = Image.open(io.BytesIO(blob[0])) if blob and blob[0] else None
    X, explanation, shap_error = get_shap(record, con, classifier)

    with st.container(border=True, key="selected_card"):
        view, info = st.columns([1, 1.6], gap="large")
        with info:
            nav = st.columns([1, 1, 2.5], vertical_alignment="center")
            nav[0].button("‹ Previous", key="card_prev", disabled=idx <= 0,
                          on_click=choose, args=(paths[idx - 1] if idx > 0 else None, idx - 1),
                          **STRETCH)
            nav[1].button("Next ›", key="card_next", disabled=idx >= len(paths) - 1,
                          on_click=choose,
                          args=(paths[idx + 1] if idx < len(paths) - 1 else None, idx + 1),
                          **STRETCH)
            nav[2].markdown(f"<span class='muted'>{idx + 1:,} of {len(paths):,}</span>",
                            unsafe_allow_html=True)

            st.subheader(record.filename)
            label = record.predicted_class or str(record.processing_status).replace("_", " ")
            score = (f'<span class="muted">score {record.confidence:.3f}</span>'
                     if pd.notna(record.confidence) else "")
            outcome_chip = ""
            if record.outcome == "correct":
                outcome_chip = '<span class="pill good">Correct</span>'
            elif record.misclassified:
                outcome_chip = (f'<span class="pill bad">{record.outcome.capitalize()} · '
                                f'true {html.escape(str(record.true_class))}</span>')
            st.markdown(f'<span class="pill">{html.escape(str(label))}</span>{outcome_chip}{score}',
                        unsafe_allow_html=True)
            st.caption(f"{record.width} × {record.height} px, {record.format or 'unknown format'}")
            image_id = record.get("image_id")
            if image_id is not None and pd.notna(image_id) and str(image_id):
                st.markdown(
                    f"<div class='muted' style='margin-left:0;word-break:break-all'>"
                    f"ID <code>{html.escape(str(image_id))}</code></div>",
                    unsafe_allow_html=True,
                )
            if record.error:
                st.error(record.error)

            opacity = None
            if explanation is not None:
                st.markdown("**SHAP overlay**", help=SHAP_HELP)
                show = st.toggle("Show overlay", key="shap_on")
                st.slider("Opacity", 0.0, 1.0, step=0.05, key="shap_opacity", disabled=not show)
                opacity = st.session_state["shap_opacity"] if show else 0.0
                st.markdown(
                    f'<div class="legend">'
                    f'<div class="item"><span class="swatch" style="background:rgb{POS_RGB}"></span>'
                    f"Raises pneumonia score</div>"
                    f'<div class="item"><span class="swatch" style="background:rgb{NEG_RGB}"></span>'
                    f"Lowers pneumonia score</div></div>",
                    unsafe_allow_html=True,
                )
            elif shap_error:
                st.caption(f"SHAP unavailable: {shap_error}")

        composed = render_view(base, X, explanation, opacity)
        with view:
            if composed is not None:
                if explanation is None or not interactive_view(composed, explanation):
                    st.image(composed, width=VIEW_SIZE)
            else:
                st.info("No preview available for this image.")

        with info:
            if composed is not None:
                buf = io.BytesIO()
                composed.save(buf, "PNG")
                suffix = f"_shap{int(round((opacity or 0) * 100))}" if opacity else ""
                st.download_button(
                    "Save snapshot",
                    buf.getvalue(),
                    f"{Path(record.filename).stem}{suffix}.png",
                    "image/png",
                    type="primary",
                )

        with st.expander("Metadata", expanded=True):
            metadata_tables(record)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
database = Path(os.environ.get("IMAGE_CATALOG_DB", str(APP_DATABASE)))
con = connect(database)
try:
    try:
        classifier = default_classifier()
    except Exception:
        classifier = None
    if classifier is None:
        st.warning("No classifier loaded. Images will be cataloged without predictions.")

    # --- Sidebar: uploads -------------------------------------------------
    with st.sidebar:
        with st.expander("Add images"):
            uploads = st.file_uploader(
                "PNG, JPEG, BMP, TIFF or WebP", accept_multiple_files=True, key="uploads"
            )
            retry = st.checkbox("Retry failed images")
            if st.button("Process", disabled=not uploads, type="primary"):
                outcomes, selected_upload = [], None
                with st.spinner("Processing…"):
                    for upload in uploads:
                        name = Path(upload.name.replace("\\", "/")).name
                        try:
                            if upload.size > MAX_FILE_BYTES:
                                raise ValueError("file is larger than 25 MiB")
                            data_bytes = upload.getvalue()
                            source = f"upload:{hashlib.sha256(data_bytes).hexdigest()}:{name}"
                            status = ingest_bytes(con, data_bytes, name, source, classifier, retry)
                            outcomes.append({"filename": name, "status": status})
                            selected_upload = source
                        except Exception as exc:
                            outcomes.append({"filename": name, "status": f"error: {exc}"})
                st.session_state["upload_report"] = outcomes
                if selected_upload:
                    clear_filters()
                    st.session_state["selected_path"] = selected_upload
                    st.session_state["jump_to_selected"] = True
            report = st.session_state.get("upload_report")
            if report:
                failed = [r for r in report if r["status"] not in {"complete", "awaiting_model"}]
                st.caption(f"{len(report) - len(failed)} added, {len(failed)} failed")
                if failed:
                    st.dataframe(pd.DataFrame(failed), hide_index=True)
            if os.environ.get("IMAGE_EXPLORER_LOCAL_IMPORT") == "1":
                folder = st.text_input("Local folder")
                if st.button("Import folder", disabled=not folder):
                    try:
                        st.write(ingest_directory(con, Path(folder), classifier, retry))
                    except Exception as exc:
                        st.error(str(exc))
                if st.button("Import NPZ dataset", disabled=not PATHS.dataset.exists()):
                    try:
                        st.write(ingest_npz(con, PATHS.dataset, classifier, retry))
                    except Exception as exc:
                        st.error(str(exc))

    data = add_outcomes(catalog(con))
    if data.empty:
        st.info("No images yet. Use Add images in the sidebar to get started.")
        st.stop()

    # --- Filters ------------------------------------------------------------
    with st.container(border=True):
        a, b, c, d, o, e = st.columns([3, 2, 2, 2, 2, 1], vertical_alignment="bottom")
        query = a.text_input("Search", key="search", placeholder="Filename or path")
        classes = b.multiselect("Class", sorted(data.predicted_class.dropna().unique()), key="classes")
        statuses = c.multiselect("Status", sorted(data.processing_status.unique()), key="statuses")
        splits = d.multiselect("Split", sorted(data.split.dropna().unique(), key=split_sort_key),
                               key="splits", help=SPLIT_HELP)
        outcomes = o.multiselect(
            "Outcome", OUTCOMES, key="outcomes",
            placeholder="All", format_func=str.capitalize,
            help="Compare the prediction with the ground-truth label. "
                 "False positive = predicted pneumonia, actually normal.",
        )
        e.button("Reset", on_click=clear_filters, **STRETCH)
        with st.expander("More filters"):
            a, b, c, d = st.columns(4)
            formats = a.multiselect("Format", sorted(data.format.dropna().unique()), key="formats")
            confidence = b.slider("Min. score", 0.0, 1.0, step=0.05, key="confidence")
            min_width = c.number_input("Min. width (px)", min_value=0, key="min_width")
            min_height = d.number_input("Min. height (px)", min_value=0, key="min_height")
            a, b, c, d = st.columns(4)
            max_mb = a.number_input("Max. size (MiB)", min_value=0.0, key="max_mb")
            sort = b.selectbox(
                "Sort by",
                ["filename", "processed_at", "confidence", "width", "height", "file_size"],
                key="sort",
            )
            descending = c.checkbox("Descending", key="descending")
            only_duplicates = c.checkbox("Duplicates only", key="duplicates")
            include_unclassified = d.checkbox("Include unclassified", key="unclassified")

    filtered = data.copy()
    if query:
        filtered = filtered[
            filtered.filename.str.contains(query, case=False, regex=False)
            | filtered.path.str.contains(query, case=False, regex=False)
        ]
    for column, values in [
        ("processing_status", statuses),
        ("predicted_class", classes),
        ("format", formats),
        ("split", splits),
        ("outcome", outcomes),
    ]:
        if values:
            filtered = filtered[filtered[column].isin(values)]
    filtered = filtered[
        (filtered.confidence >= confidence)
        | (include_unclassified & filtered.confidence.isna())
    ]
    filtered = filtered[
        (filtered.width.fillna(0) >= min_width) & (filtered.height.fillna(0) >= min_height)
    ]
    filtered = filtered[filtered.file_size.isna() | (filtered.file_size <= max_mb * 1024**2)]
    if only_duplicates:
        filtered = filtered[filtered.copies > 1]
    filtered = filtered.sort_values(sort, ascending=not descending, na_position="last")
    filtered = filtered.reset_index(drop=True)

    m = st.columns(4)
    m[0].metric("Images", f"{len(filtered):,}")
    m[1].metric("Unique", f"{filtered.image_id.nunique():,}")
    m[2].metric("Classified", f"{int(filtered.predicted_class.notna().sum()):,}")
    m[3].metric(
        "Need attention",
        int(filtered.processing_status.isin(["failed", "model_error", "unsupported"]).sum()),
    )

    if filtered.empty:
        st.info("No images match these filters. Try Reset.")
        st.stop()

    paths = filtered.path.tolist()
    if st.session_state.get("selected_path") not in paths:
        st.session_state["selected_path"] = paths[0]
    selected_path = st.session_state["selected_path"]
    sel_idx = paths.index(selected_path)
    if st.session_state.pop("jump_to_selected", False):
        choose(selected_path, sel_idx)
    record = filtered.iloc[sel_idx]

    # --- Sidebar: image list ----------------------------------------------------
    stamp = (len(data), str(data.processed_at.max()) if "processed_at" in data else "")
    icons = list_icons(str(database), tuple(filtered.image_id.unique()), stamp)
    list_pages = max(1, math.ceil(len(filtered) / LIST_SIZE))
    list_page = min(max(1, st.session_state["list_page"]), list_pages)
    st.session_state["list_page"] = list_page
    l0 = (list_page - 1) * LIST_SIZE
    l1 = min(l0 + LIST_SIZE, len(filtered))

    with st.sidebar:
        hdr = st.columns([1, 3, 1], vertical_alignment="center")
        hdr[0].button("‹", key="list_prev", disabled=list_page <= 1,
                      on_click=set_state, args=("list_page", list_page - 1), **STRETCH)
        hdr[1].markdown(
            f"<div style='text-align:center;font-size:.85rem;color:#475569'>"
            f"{l0 + 1:,}–{l1:,} of {len(filtered):,}</div>",
            unsafe_allow_html=True,
        )
        hdr[2].button("›", key="list_next", disabled=list_page >= list_pages,
                      on_click=set_state, args=("list_page", list_page + 1), **STRETCH)
        with st.container(key="sidebar_list"):
            for i in range(l0, l1):
                row = filtered.iloc[i]
                sel = row.path == selected_path
                with st.container(key=f"{'sbsel' if sel else 'sbrow'}_{path_key(row.path)}"):
                    cls = html.escape(str(row.predicted_class)) if pd.notna(row.predicted_class) else ""
                    st.markdown(
                        f"<div class='sb'><img src='{icons.get(row.image_id, BLANK)}'/>"
                        f"<span class='nm'>{html.escape(row.filename)}</span>"
                        f"<span class='cl'>{cls}</span></div>",
                        unsafe_allow_html=True,
                    )
                    st.button(row.filename, key=f"sb_{path_key(row.path)}",
                              on_click=choose, args=(row.path, i))

    # --- Tabs ------------------------------------------------------------------
    gallery, table, summary, metrics = st.tabs(["Images", "Table", "Distributions", "Model metrics"])

    with gallery:
        page_size = st.session_state["page_size"]
        pages = max(1, math.ceil(len(filtered) / page_size))
        page = min(max(1, st.session_state["page"]), pages)
        st.session_state["page"] = page
        start = (page - 1) * page_size
        end = min(start + page_size, len(filtered))

        nav = st.columns([1, 1, 3, 1, 1.4], vertical_alignment="center")
        nav[0].button("‹ Prev", key="page_prev", on_click=set_state, args=("page", page - 1),
                      disabled=page <= 1, **STRETCH)
        nav[1].button("Next ›", key="page_next", on_click=set_state, args=("page", page + 1),
                      disabled=page >= pages, **STRETCH)
        nav[2].markdown(
            f"<div style='text-align:center;color:#475569'>Page {page} of {pages} "
            f"<span class='muted'>({start + 1:,}–{end:,} of {len(filtered):,})</span></div>",
            unsafe_allow_html=True,
        )
        goto_key = f"goto_{page}_{pages}"
        nav[3].number_input("Go to page", 1, pages, value=page, key=goto_key,
                            label_visibility="collapsed",
                            on_change=lambda: set_state("page", st.session_state[goto_key]))
        nav[4].selectbox("Per page", PAGE_SIZES, key="page_size",
                         label_visibility="collapsed", format_func=lambda n: f"{n} per page",
                         on_change=set_state, args=("page", 1))

        thumbs = fetch_thumbnails(con, filtered.image_id.iloc[start:end].tolist())
        for r0 in range(start, end, GALLERY_COLS):
            cols = st.columns(GALLERY_COLS, gap="small")
            for col, i in zip(cols, range(r0, min(r0 + GALLERY_COLS, end))):
                row = filtered.iloc[i]
                sel = row.path == selected_path
                blob = thumbs.get(row.image_id)
                uri = to_uri(tile(blob, 160)) if blob else BLANK
                with col, st.container(key=f"{'gsel' if sel else 'gcard'}_{path_key(row.path)}"):
                    st.markdown(
                        f"<div class='gc'>{'<span class=miss title=Misclassified></span>' if row.misclassified else ''}"
                        f"<img src='{uri}'/>"
                        f"<div class='nm'>{html.escape(row.filename)}</div></div>",
                        unsafe_allow_html=True,
                    )
                    st.button(row.filename, key=f"g_{path_key(row.path)}",
                              on_click=choose, args=(row.path, i))

        selected_card(record, sel_idx, paths, con, classifier)

    with table:
        visible = [
            "filename", "split", "label", "format", "width", "height", "aspect_ratio",
            "file_size", "color_mode", "predicted_class", "confidence",
            "true_class", "outcome", "processing_status", "copies", "error",
        ]
        st.dataframe(filtered[visible], hide_index=True, **STRETCH)
        st.download_button(
            "Download CSV",
            filtered[visible + ["image_id", "processed_at", "model_version"]].to_csv(index=False),
            "image_results.csv",
            "text/csv",
        )

    with summary:
        left, right = st.columns(2)
        left.markdown("**Predicted class**")
        left.bar_chart(filtered.predicted_class.fillna("unclassified").value_counts(), color=ACCENT)
        right.markdown("**Processing status**")
        right.bar_chart(filtered.processing_status.value_counts(), color="#93b4f5")

    with metrics:
        use_filters = st.toggle("Apply current filters", key="metrics_filtered",
                                help="Off: metrics use every labeled image in the catalog.")
        metrics_tab(filtered if use_filters else data)
finally:
    con.close()