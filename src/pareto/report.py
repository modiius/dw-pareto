"""Render the ledger + front into one self-contained HTML file.

The template is a normal HTML file with `/*__PARETO_DATA__*/null` where the data
goes; nothing is fetched at view time, so the report can be emailed or committed.
"""
from __future__ import annotations

import html as html_mod
import json
import re
import time
from pathlib import Path

PLACEHOLDER = "/*__PARETO_DATA__*/null"
TITLE_PLACEHOLDER = "__PARETO_TITLE__"
TEMPLATE = Path(__file__).with_name("template.html")


def group_label(group_key: str) -> str:
    """Turn the JSON group key back into something a human reads."""
    try:
        ctx = json.loads(group_key)
    except json.JSONDecodeError:
        return group_key
    return ", ".join(f"{k}={v}" for k, v in ctx.items()) if ctx else "(no context)"


def payload(rows: list[dict], explained: dict, *, title: str, root: str) -> dict:
    info = explained["runs"]
    runs = []
    for r in rows:
        extra = info.get(r["run_id"], {})
        runs.append({**r, "status": extra.get("status", "incomplete"),
                     "dominated_by": extra.get("dominated_by", []),
                     "tied_with": extra.get("tied_with", []),
                     "group": extra.get("group", "")})
    groups = []
    for r in runs:
        if r["group"] and r["group"] not in [g["key"] for g in groups]:
            groups.append({"key": r["group"], "label": group_label(r["group"])})
    # every metric that appears anywhere, so extras can still be plotted
    known = dict(explained["metrics"])
    axes = [{"name": m, "direction": d, "scored": True} for m, d in known.items()]
    seen = set(known)
    for r in runs:
        for m in r["scores"]:
            if m not in seen:
                seen.add(m)
                axes.append({"name": m, "direction": None, "scored": False})
    return {"title": title, "root": root,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "metrics": known, "axes": axes, "groups": groups,
            "alpha": explained["alpha"], "eps": explained["eps"],
            "ignore": explained["ignore"], "runs": runs}


def fragment(html: str) -> str:
    """The same page without the document wrapper: <title>, <style>, then the body.

    That is the shape an Artifact host wants - it supplies its own <!doctype>,
    <head> and <body> - and it is also what you paste into an existing page.
    """
    parts = []
    for pattern in (r"<title>.*?</title>", r"<style>.*?</style>"):
        found = re.search(pattern, html, re.S)
        if found:
            parts.append(found.group(0))
    body = re.search(r"<body>(.*?)</body>", html, re.S)
    parts.append((body.group(1) if body else html).strip())
    return "\n".join(parts) + "\n"


def render(rows: list[dict], explained: dict, *, title: str, root: str,
           embed: bool = False) -> str:
    template = TEMPLATE.read_text(encoding="utf-8")
    if PLACEHOLDER not in template:
        raise RuntimeError(f"{TEMPLATE} has no {PLACEHOLDER} placeholder")
    data = json.dumps(payload(rows, explained, title=title, root=root), default=str)
    # </script> inside a string literal would close the tag early
    data = data.replace("</", "<\\/")
    out = template.replace(PLACEHOLDER, data).replace(TITLE_PLACEHOLDER, html_mod.escape(title))
    return fragment(out) if embed else out
