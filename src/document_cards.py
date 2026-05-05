"""Generate descriptive document cards for each cluster of California Housing data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import anthropic
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ─── Prompt constants ────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "Eres un analista de datos experto en el mercado inmobiliario de California. "
    "Tu tarea es leer estadísticas de un cluster de viviendas y generar una card "
    "descriptiva, concisa y útil para presentar ese segmento a un público no técnico. "
    "Responde siempre en JSON válido, sin texto adicional fuera del objeto JSON."
)

CARD_PROMPT_TEMPLATE = """\
A continuación se presentan las estadísticas agregadas de un cluster de viviendas \
del dataset California Housing.

## Cluster {cluster_id} — {n_samples:,} viviendas ({pct:.1f}% del total)

| Variable | Media | Mediana | Desv. estándar |
|---|---|---|---|
| Ingreso mediano (MedInc, ×$10k) | {MedInc_mean:.2f} | {MedInc_median:.2f} | {MedInc_std:.2f} |
| Edad mediana vivienda (HouseAge, años) | {HouseAge_mean:.1f} | {HouseAge_median:.1f} | {HouseAge_std:.1f} |
| Habitaciones por hogar (AveRooms) | {AveRooms_mean:.1f} | {AveRooms_median:.1f} | {AveRooms_std:.1f} |
| Dormitorios por hogar (AveBedrms) | {AveBedrms_mean:.1f} | {AveBedrms_median:.1f} | {AveBedrms_std:.1f} |
| Población del bloque | {Population_mean:.0f} | {Population_median:.0f} | {Population_std:.0f} |
| Ocupación media (AveOccup, personas/hogar) | {AveOccup_mean:.1f} | {AveOccup_median:.1f} | {AveOccup_std:.1f} |
| Latitud | {Latitude_mean:.2f} | — | — |
| Longitud | {Longitude_mean:.2f} | — | — |
| **Valor mediano vivienda (MedHouseVal, ×$100k)** | **{MedHouseVal_mean:.2f}** | **{MedHouseVal_median:.2f}** | **{MedHouseVal_std:.2f}** |

Genera una card con **exactamente** este esquema JSON:

{{
  "titulo": "<nombre descriptivo del segmento, máx. 8 palabras>",
  "descripcion": "<2-3 oraciones describiendo el perfil típico de las viviendas y residentes>",
  "caracteristicas_clave": [
    "<rasgo distintivo 1>",
    "<rasgo distintivo 2>",
    "<rasgo distintivo 3>"
  ],
  "zona_geografica": "<región aproximada de California inferida de lat/lon>",
  "nivel_precio": "<bajo | medio | alto | muy alto>",
  "perfil_demografico": "<breve descripción del perfil socioeconómico del segmento>"
}}
"""

# ─── Clustering ───────────────────────────────────────────────────────────────

FEATURES_FOR_CLUSTERING = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "AveOccup", "Latitude", "Longitude",
]

ALL_FEATURES = FEATURES_FOR_CLUSTERING + ["MedHouseVal"]


def cluster_data(df: pd.DataFrame, n_clusters: int = 5, random_state: int = 31415) -> pd.DataFrame:
    """Add a 'cluster' column to df using KMeans on housing features."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[FEATURES_FOR_CLUSTERING])
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    df = df.copy()
    df["cluster"] = km.fit_predict(X_scaled)
    return df


def compute_cluster_stats(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Return per-cluster aggregated statistics."""
    total = len(df)
    stats = []
    for cid, grp in df.groupby("cluster"):
        row: dict[str, Any] = {"cluster_id": int(cid), "n_samples": len(grp), "pct": len(grp) / total * 100}
        for col in ALL_FEATURES:
            row[f"{col}_mean"] = float(grp[col].mean())
            row[f"{col}_median"] = float(grp[col].median())
            row[f"{col}_std"] = float(grp[col].std())
        stats.append(row)
    return stats


# ─── Prompt builder ───────────────────────────────────────────────────────────

def build_card_prompt(cluster_stats: dict[str, Any]) -> str:
    """Fill CARD_PROMPT_TEMPLATE with per-cluster statistics.

    This is the exact user-turn content sent to Claude for each cluster.
    """
    return CARD_PROMPT_TEMPLATE.format(**cluster_stats)


# ─── Claude API call ──────────────────────────────────────────────────────────

def generate_card(client: anthropic.Anthropic, cluster_stats: dict[str, Any]) -> dict[str, Any]:
    """Call Claude to generate a card for a single cluster."""
    user_prompt = build_card_prompt(cluster_stats)

    with client.messages.stream(
        model="claude-opus-4-7",
        max_tokens=1024,
        thinking={"type": "adaptive"},
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    ) as stream:
        message = stream.get_final_message()

    text_block = next(b for b in message.content if b.type == "text")
    card = json.loads(text_block.text)
    card["cluster_id"] = cluster_stats["cluster_id"]
    card["n_samples"] = cluster_stats["n_samples"]
    card["pct"] = round(cluster_stats["pct"], 1)
    card["valor_mediano_mean"] = round(cluster_stats["MedHouseVal_mean"], 2)
    return card


def generate_all_cards(
    df: pd.DataFrame,
    n_clusters: int = 5,
    random_state: int = 31415,
    api_key: str | None = None,
) -> list[dict[str, Any]]:
    """Cluster df and generate one card per cluster via Claude."""
    client = anthropic.Anthropic(api_key=api_key)
    df_clustered = cluster_data(df, n_clusters=n_clusters, random_state=random_state)
    all_stats = compute_cluster_stats(df_clustered)
    cards = []
    for stats in sorted(all_stats, key=lambda s: s["cluster_id"]):
        print(f"  Generando card para cluster {stats['cluster_id']} ({stats['n_samples']:,} muestras)…")
        card = generate_card(client, stats)
        cards.append(card)
    return cards


# ─── Output ───────────────────────────────────────────────────────────────────

def save_cards_json(cards: list[dict], output_path: Path) -> None:
    output_path.write_text(json.dumps(cards, ensure_ascii=False, indent=2), encoding="utf-8")


def save_cards_html(cards: list[dict], output_path: Path) -> None:
    NIVEL_COLOR = {
        "bajo": "#4caf50",
        "medio": "#2196f3",
        "alto": "#ff9800",
        "muy alto": "#f44336",
    }

    card_html = ""
    for c in cards:
        color = NIVEL_COLOR.get(c.get("nivel_precio", "medio"), "#607d8b")
        items = "".join(f"<li>{f}</li>" for f in c.get("caracteristicas_clave", []))
        card_html += f"""
        <div class="card">
          <div class="card-header" style="border-left: 6px solid {color}">
            <span class="cluster-badge">Cluster {c["cluster_id"]}</span>
            <h2>{c["titulo"]}</h2>
            <span class="badge" style="background:{color}">{c.get("nivel_precio","").upper()}</span>
          </div>
          <p class="desc">{c["descripcion"]}</p>
          <ul>{items}</ul>
          <div class="meta">
            <span>📍 {c.get("zona_geografica","")}</span>
            <span>👥 {c.get("perfil_demografico","")}</span>
            <span>🏠 {c["n_samples"]:,} viviendas ({c["pct"]}%)</span>
            <span>💲 Valor medio: ${c["valor_mediano_mean"]*100:.0f}k</span>
          </div>
        </div>"""

    html = f"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <title>Document Cards — California Housing Clusters</title>
  <style>
    body{{font-family:Arial,sans-serif;background:#f5f5f5;margin:0;padding:24px}}
    h1{{color:#1f3b4d;text-align:center}}
    .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:20px;max-width:1400px;margin:0 auto}}
    .card{{background:#fff;border-radius:10px;padding:20px;box-shadow:0 2px 8px rgba(0,0,0,.1)}}
    .card-header{{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:12px}}
    .cluster-badge{{background:#37474f;color:#fff;border-radius:4px;padding:2px 8px;font-size:.8em;font-weight:bold}}
    h2{{margin:0;font-size:1.1em;color:#1f3b4d;flex:1}}
    .badge{{color:#fff;border-radius:4px;padding:2px 8px;font-size:.75em;font-weight:bold}}
    .desc{{color:#555;font-size:.95em;line-height:1.5;margin-bottom:10px}}
    ul{{padding-left:18px;color:#444;font-size:.9em;margin-bottom:12px}}
    .meta{{display:flex;flex-wrap:wrap;gap:8px;font-size:.8em;color:#666}}
    .meta span{{background:#f0f0f0;border-radius:4px;padding:3px 8px}}
  </style>
</head>
<body>
  <h1>California Housing — Document Cards por Cluster</h1>
  <div class="grid">{card_html}</div>
</body>
</html>"""
    output_path.write_text(html, encoding="utf-8")


# ─── CLI entry point ──────────────────────────────────────────────────────────

def main(csv_path: str = "california_housing.csv", n_clusters: int = 5, output_dir: str = ".") -> None:
    df = pd.read_csv(csv_path)
    print(f"Dataset cargado: {len(df):,} filas")
    print(f"Generando {n_clusters} clusters y cards via Claude…\n")

    cards = generate_all_cards(df, n_clusters=n_clusters)

    out = Path(output_dir)
    json_path = out / "document_cards.json"
    html_path = out / "document_cards.html"
    save_cards_json(cards, json_path)
    save_cards_html(cards, html_path)

    print(f"\n✓ JSON: {json_path}")
    print(f"✓ HTML: {html_path}")


if __name__ == "__main__":
    main()
