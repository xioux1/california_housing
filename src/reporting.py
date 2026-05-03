from __future__ import annotations

from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any


def _dict_to_rows(d: dict[str, Any], prefix: str = "") -> str:
    rows: list[str] = []
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            rows.append(_dict_to_rows(v, key))
        else:
            rows.append(f"<tr><td>{escape(key)}</td><td>{escape(str(v))}</td></tr>")
    return "".join(rows)


def generate_html_report(
    run_dir: Path,
    run_metadata: dict,
    dataset_summary: dict,
    config: dict,
    metrics: dict,
    plot_paths: list[Path],
    artifact_paths: list[Path],
    warnings: list[str] | None = None,
) -> Path:
    warnings = warnings or []
    rel_plots = [p.relative_to(run_dir) if p.exists() else p for p in plot_paths]
    missing_plots = [str(p) for p in rel_plots if not (run_dir / p).exists()]

    warning_items = "".join(f"<li>{escape(w)}</li>" for w in warnings + [f"Missing plot: {m}" for m in missing_plots])
    plots_html = "".join(
        f'<figure><img src="{escape(str(p))}" alt="{escape(p.stem)}"><figcaption>{escape(p.name)}</figcaption></figure>'
        for p in rel_plots if (run_dir / p).exists()
    )
    artifact_html = "".join(f"<li>{escape(str(p.relative_to(run_dir)))}</li>" for p in artifact_paths if p.exists())

    html = f"""<!doctype html>
<html><head><meta charset='utf-8'><title>Experiment Report {escape(str(run_metadata.get('run_id')))}</title>
<style>body{{font-family:Arial,sans-serif;background:#fafafa;color:#222}} .container{{max-width:1000px;margin:30px auto;background:#fff;padding:24px;border-radius:8px}} h2{{border-bottom:1px solid #ddd;padding-bottom:4px}} table{{width:100%;border-collapse:collapse}} td,th{{border:1px solid #ddd;padding:6px;text-align:left}} figure{{display:inline-block;width:31%;margin:1%}} img{{max-width:100%;border:1px solid #ddd}}</style></head>
<body><div class='container'>
<h1>Experiment Report</h1>
<p><b>Run ID:</b> {escape(str(run_metadata.get('run_id')))}<br><b>Date/Time:</b> {escape(run_metadata.get('timestamp', datetime.utcnow().isoformat()))}</p>
<h2>Dataset Summary</h2><table>{_dict_to_rows(dataset_summary)}</table>
<h2>Model / Config Summary</h2><table>{_dict_to_rows(config)}</table>
<h2>Training Summary</h2><table>{_dict_to_rows(run_metadata.get('training_summary', {}))}</table>
<h2>Metrics</h2><table>{_dict_to_rows(metrics)}</table>
<h2>Plots</h2><div>{plots_html}</div>
<h2>Notes / Warnings</h2><ul>{warning_items or '<li>None</li>'}</ul>
<h2>Artifacts</h2><ul>{artifact_html}</ul>
</div></body></html>"""

    report_path = run_dir / "report.html"
    report_path.write_text(html, encoding="utf-8")
    return report_path
