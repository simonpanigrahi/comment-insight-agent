"""Step 10: write the results. The HTML report is one self-contained file, so it works offline and can be emailed."""
from __future__ import annotations

import html
import json
import math
from datetime import datetime
from pathlib import Path

import pandas as pd

from .analyse import Analysis, pct
from .insights import ROUGH_BELOW
from .schemas import Insights
from .topics import GENERAL, OTHER

SENT_LABEL = {"negative": "Negative", "neutral": "Neutral", "positive": "Positive"}
URGENCY = {"today": ("Today", "critical"), "this_week": ("This week", "serious"), "this_month": ("This month", "quiet")}
SERIES = ["--s1", "--s2", "--s3"]


def esc(x) -> str:
    return html.escape(str(x), quote=True)


def seg_bar(counts: dict, tall: bool = True) -> str:
    total = sum(counts.values()) or 1
    parts = []
    for key in ("negative", "neutral", "positive"):
        n = counts.get(key, 0)
        if not n:
            continue
        share = 100 * n / total
        label = f"{share:.0f}%" if tall and share >= 7 else ""
        parts.append(f'<span class="seg seg-{key}" style="flex:{n} 1 0" data-tip="{SENT_LABEL[key]}: {n:,} ({share:.1f}%)" tabindex="0">{label}</span>')
    return f'<div class="bar {"tall" if tall else "thin"}" role="img" aria-label="' + ", ".join(f"{SENT_LABEL[k]} {counts.get(k, 0):,}" for k in SENT_LABEL) + f'">{"".join(parts)}</div>'


def nice_ticks(top: float, count: int = 4) -> list[int]:
    if top <= 0:
        return [0, 1]
    raw = top / count
    magnitude = 10 ** (len(str(int(raw))) - 1) if raw >= 1 else 1
    for step in (1, 2, 5, 10):
        if step * magnitude >= raw:
            s = step * magnitude
            break
    ticks = list(range(0, int(top) + s, int(s)))
    return ticks if ticks[-1] >= top else ticks + [ticks[-1] + int(s)]


def line_chart(a: Analysis, topics: list[str]) -> str:
    """Negative comments per day for the top topics. Inline SVG, so it stays sharp and follows the page's theme."""
    d = a.daily[topics]
    w, h, left, right, top, bottom = 960, 320, 44, 170, 18, 36
    plot_w, plot_h = w - left - right, h - top - bottom
    ticks = nice_ticks(float(d.values.max()))
    ymax = ticks[-1]
    x = lambda i: left + plot_w * i / max(len(d) - 1, 1)
    y = lambda v: top + plot_h * (1 - v / ymax)
    grid = "".join(f'<line class="grid" x1="{left}" x2="{w - right}" y1="{y(t):.1f}" y2="{y(t):.1f}"/><text class="tick" x="{left - 8}" y="{y(t) + 4:.1f}" text-anchor="end">{t:,}</text>' for t in ticks)
    every = max(1, round(len(d) / 7))
    xticks = "".join(f'<text class="tick" x="{x(i):.1f}" y="{h - 12}" text-anchor="middle">{day:%d %b}</text>' for i, day in enumerate(d.index) if i % every == 0)
    lines, ends = "", []
    for k, topic in enumerate(topics):
        pts = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(d[topic]))
        lines += f'<polyline class="series" style="stroke:var({SERIES[k]})" points="{pts}"/>'
        ends.append((float(y(d[topic].iloc[-1])), topic, k))
    ends.sort()
    apart = all(b[0] - a_[0] >= 16 for a_, b in zip(ends, ends[1:]))
    for ypos, topic, k in ends:
        lines += f'<circle class="dot" cx="{x(len(d) - 1):.1f}" cy="{ypos:.1f}" r="4" style="fill:var({SERIES[k]})"/>'
        if apart:
            lines += f'<text class="end" x="{x(len(d) - 1) + 10:.1f}" y="{ypos + 4:.1f}">{esc(topic[:22])}</text>'
    marks = ""
    for s in a.spikes:
        if s["topic"] in topics:
            i = list(d.index.strftime("%Y-%m-%d")).index(s["date"])
            name = esc(s["topic"][:24])
            marks += f'<line class="mark" x1="{x(i):.1f}" x2="{x(i):.1f}" y1="{top}" y2="{top + plot_h}"/><text class="tick" x="{x(i) + 6:.1f}" y="{top + 12}">Spike begins, {name}</text>'
    bands = ""
    band = plot_w / max(len(d) - 1, 1)
    for i, day in enumerate(d.index):
        tip = f"{day:%d %b}: " + ", ".join(f"{t} {int(d[t].iloc[i]):,}" for t in topics)
        bands += f'<rect class="hit" x="{x(i) - band / 2:.1f}" y="{top}" width="{band:.1f}" height="{plot_h}" data-tip="{esc(tip)}" data-x="{x(i):.1f}"/>'
    legend = "".join(f'<span class="key"><i style="background:var({SERIES[k]})"></i>{esc(t)}</span>' for k, t in enumerate(topics))
    rows = ""
    for i, day in enumerate(d.index):
        cells = "".join(f'<td class="num">{int(d[t].iloc[i]):,}</td>' for t in topics)
        rows += f"<tr><td>{day:%d %b %Y}</td>{cells}</tr>"
    heads = "".join(f'<th class="num">{esc(t)}</th>' for t in topics)
    table = f'<details><summary>See these numbers as a table</summary><div class="scroll"><table><thead><tr><th>Day</th>{heads}</tr></thead><tbody>{rows}</tbody></table></div></details>'
    aria = esc(", ".join(topics))
    return (f'<div class="legend">{legend}</div><div class="chart"><svg viewBox="0 0 {w} {h}" role="img" aria-label="Negative comments per day for {aria}">'
            f'{grid}{xticks}{marks}{lines}<line class="cross" x1="0" x2="0" y1="{top}" y2="{top + plot_h}" style="opacity:0"/>{bands}</svg></div>{table}')


def _action_rows(ins: Insights) -> str:
    rows = ""
    for x in sorted(ins.actions, key=lambda a: list(URGENCY).index(a.urgency)):
        label, cls = URGENCY[x.urgency]
        refs = " ".join(f'<a href="#{esc(f)}">{esc(f)}</a>' for f in x.fact_ids)
        rows += f'<tr><td><span class="badge {cls}"><i></i>{label}</span></td><td><strong>{esc(x.action)}</strong><div class="why">{esc(x.why)}</div></td><td>{esc(x.owner)}</td><td class="refs">{refs}</td></tr>'
    return rows


def _alert_cards(a: Analysis) -> str:
    out = ""
    for x in a.alerts:
        quotes = ""
        for e in x["examples"][:3]:
            reach = f", engagement {int(e['engagement']):,}" if a.has_engagement else ""
            quotes += f'<li>"{esc(str(e["clean_text"])[:180])}" <span class="meta">{esc(e["id"])}{reach}</span></li>'
        total = f"total engagement {int(x['engagement']):,}" if a.has_engagement else ""
        doubt = f'<p class="meta doubt">The LLM judged {x["doubted"]:,} of these false alarms. They stay here for a person to check, and the llm_doubted column of labelled_comments.csv marks them.</p>' if x.get("doubted") else ""
        out += f'<div class="alert"><h3>{esc(x["label"])}: {x["count"]:,} comments<span class="meta">{total}</span></h3><ul class="quotes">{quotes}</ul>{doubt}</div>'
    return out


def _topic_rows(a: Analysis) -> str:
    rows = ""
    arrows = {"rising": "Rising", "falling": "Falling", "steady": "Steady", "": ""}
    for r in a.topics.itertuples():
        counts = {"negative": r.negative, "neutral": r.neutral, "positive": r.positive}
        rows += f'<tr><td><strong>{esc(r.topic)}</strong></td><td class="num">{r.count:,}</td><td class="cellbar">{seg_bar(counts, tall=False)}</td><td class="num">{r.neg_share}%</td><td class="num">{r.pos_share}%</td>'
        rows += f'<td>{arrows.get(r.trend, "")}</td><td class="num">{r.impact:,.0f}</td></tr>' if a.has_time else f'<td class="num">{r.impact:,.0f}</td></tr>'
    return rows


def _reason_list(groups: list[dict]) -> str:
    items = "".join(f'<li>{esc(g["label"])} <span class="meta">{g["count"]:,}</span></li>' for g in groups[:3])
    return items or '<li class="meta">None</li>'


def _reason_rows(a: Analysis) -> str:
    rows = ""
    for t in [t for t in a.topics["topic"] if t != OTHER][:8]:
        entry = a.reasons.get(t, {})
        rows += f'<tr><td><strong>{esc(t)}</strong></td><td><ul class="plain">{_reason_list(entry.get("negative", []))}</ul></td><td><ul class="plain">{_reason_list(entry.get("positive", []))}</ul></td></tr>'
    return rows


def _group_list(groups: list[dict]) -> str:
    return "".join(f'<li><span class="count">{g["count"]:,}</span><span>{esc(g["label"][:150])}</span></li>' for g in groups) or '<li class="meta">None found in this data.</li>'


def render_html(title: str, a: Analysis, ins: Insights, usage: dict, stats: dict, removed: int, decisions: list[dict], info: dict) -> str:
    total = len(a.rows)
    share = a.coverage
    read = f"{math.floor(share * 1000) / 10:g}%"  # rounded down, so that 99.6% does not read as everything
    dated = a.rows["created_at"].dropna()
    period = f"{dated.min():%d %b %Y} to {dated.max():%d %b %Y}" if len(dated) else "dates not given"
    mode = "Rules only, no LLM" if share == 0 else f"LLM: {usage.get('model')} through {usage.get('provider')}, read {read} of comments"
    if share == 0:
        banner = ('<div class="banner"><strong>Rules-only run.</strong> These labels are rough. On real labelled tweets, rules reached about 61% accuracy on airline tweets and 53% on product tweets, '
                  'below always guessing the commonest class (63% and 60%), and they miss most sarcasm. This mode suits a quick look, alerts and spikes, not exact sentiment shares. '
                  'For better labels, run with an LLM and measure it with scripts/evaluate.py.</div>')
    elif share < ROUGH_BELOW:
        banner = (f'<div class="banner"><strong>Partial coverage.</strong> The LLM read {read} of comments. The rest use a small student model or rules, which are much less reliable, '
                  'so overall sentiment shares are estimates. See "Labels by source" at the end of this report.</div>')
    else:
        banner = ""
    sample_note = f" Random sample of {a.received:,} from {info['sampled_from']:,} rows." if info.get("sampled_from") else ""
    sources = stats.get("source_counts", {})
    source_text = ", ".join(f"{k}: {v:,}" for k, v in sources.items())
    agree = stats.get("rules_llm_agreement")
    agree_text = f"<li>Rules agreed with the LLM on {agree:.0%} of the sentiment labels the LLM produced.</li>" if agree is not None else ""
    review = int(a.rows["needs_review"].fillna(False).astype(bool).sum())
    verified = "The checker confirmed that each number, date and quote in the findings appears in the facts it cites. This does not prove each label is right, and it cannot catch a fact that is misread."
    if removed:
        verified += f" {removed} statements failed the check and were removed."
    points = ""
    for p in ins.points:
        refs = " ".join(f'<a href="#{esc(f)}">{esc(f)}</a>' for f in p.fact_ids)
        points += f'<li>{esc(p.text)} <span class="refs">{refs}</span></li>'
    tiles = [("Comments analysed", f"{total:,}", f"of {a.received:,} received"), ("Negative", f"{pct(a.overall['negative'], total)}%", f"{a.overall['negative']:,} comments"),
             ("Positive", f"{pct(a.overall['positive'], total)}%", f"{a.overall['positive']:,} comments"),
             ("Alerts", f"{len(a.alerts)}", f"{sum(x['count'] for x in a.alerts):,} comments"), ("Spikes", f"{len(a.spikes)}", "sudden rises in complaints")]
    tile_html = "".join(f'<div class="tile"><div class="label">{l}</div><div class="value">{v}</div><div class="meta">{m}</div></div>' for l, v, m in tiles)

    trend_html = ""
    if a.has_time and a.daily is not None and not a.daily.empty:
        named = [t for t in a.topics["topic"] if t not in (OTHER, GENERAL) and t in a.daily][:3]
        if named:
            drivers = "".join(f'<p class="note">{esc(f.text)}</p>' for f in a.book.of("driver"))
            trend_html = f'<section><h2>How complaints moved over time</h2><p class="lead">Negative comments per day for the three highest-impact topics.</p>{line_chart(a, named)}{drivers}</section>'

    platform_html = ""
    if a.platforms is not None and not a.platforms.empty:
        rows = ""
        for r in a.platforms.itertuples():
            bar = seg_bar({"negative": int(r.negative), "neutral": int(r.count - r.negative - r.positive), "positive": int(r.positive)}, tall=False)
            rows += f'<tr><td>{esc(r.platform)}</td><td class="num">{r.count:,}</td><td class="cellbar">{bar}</td><td class="num">{r.neg_share}%</td></tr>'
        platform_html = f'<div class="spaced"><h3>By platform</h3><div class="scroll"><table><thead><tr><th>Platform</th><th class="num">Comments</th><th>Sentiment</th><th class="num">Negative</th></tr></thead><tbody>{rows}</tbody></table></div></div>'
    lang_rows = "".join(f'<tr><td>{esc(k)}</td><td class="num">{v:,}</td><td class="num">{pct(v, total)}%</td></tr>' for k, v in a.languages.items())
    intent_rows = ""
    for k, v in a.intents.items():
        name = k.replace("_", " ").capitalize()
        intent_rows += f'<tr><td>{esc(name)}</td><td class="num">{v:,}</td><td class="num">{pct(v, total)}%</td></tr>'
    steps = "".join(f'<li><span class="step">{esc(d["step"])}</span>{esc(d["text"])}</li>' for d in decisions)
    facts = ""
    for f in a.book.facts:
        cited = ""
        if f.comment_ids:
            joined = esc(", ".join(f.comment_ids[:8]))
            cited = f'<div class="meta">Comments: {joined}</div>'
        facts += f'<tr id="{esc(f.id)}"><td class="refs">{esc(f.id)}</td><td>{esc(f.text)}{cited}</td></tr>'
    llm_line = ""
    if usage.get("calls") or usage.get("cached_calls"):
        llm_line = (f"<li>LLM use: {usage.get('calls', 0)} calls plus {usage.get('cached_calls', 0)} from cache, {usage.get('tokens_in', 0):,} tokens in, "
                    f"{usage.get('tokens_out', 0):,} tokens out, {usage.get('seconds_in_llm', 0):,.0f} seconds.</li>")
    alerts_html = ""
    if a.alerts:
        alerts_html = ('<section><h2>Alerts</h2><p class="lead">Rare but serious. Rules scan every comment for common wordings of these on every run, and an LLM can add to this list but cannot take anything off it. '
                       f'The word lists are narrow and can miss unusual wordings. A person should read these.</p><div class="alerts">{_alert_cards(a)}</div></section>')
    faq_lead = "Candidates for FAQ or chatbot knowledge base entries, for a person to review." if share >= ROUGH_BELOW else "Candidates for FAQ entries. Most were picked by rules from question marks and question words, so a person should review them first."
    trend_head = "<th>Trend</th>" if a.has_time else ""
    surge_html = "".join(f'<p class="note">{esc(f.text)}</p>' for f in a.book.of("surge"))
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)}: comment report</title><style>{CSS}</style></head><body><main>
<header><p class="eyebrow">Customer comment report</p><h1>{esc(title)}</h1><p class="meta">{a.received:,} comments, {period}. {esc(mode)}.{esc(sample_note)} Generated {datetime.now():%d %b %Y}.</p></header>
{banner}
<p class="headline">{esc(ins.headline)}</p>
<div class="tiles">{tile_html}</div>
<div class="sent">{seg_bar(a.overall)}<div class="legend"><span class="key"><i class="seg-negative"></i>Negative</span><span class="key"><i class="seg-neutral"></i>Neutral</span><span class="key"><i class="seg-positive"></i>Positive</span></div></div>

<section><h2>What to do first</h2><div class="scroll"><table class="actions"><thead><tr><th>When</th><th>Action</th><th>Owner</th><th>Evidence</th></tr></thead><tbody>{_action_rows(ins)}</tbody></table></div></section>
{alerts_html}
<section><h2>Key findings</h2><ul class="points">{points}</ul></section>
{trend_html}
{surge_html}<section><h2>Topics, ranked by impact</h2><p class="lead">Topics were found in the data, not set in advance. Impact is negative comments, scaled up when they get more engagement. It is a simple ranking aid, not a validated measure.</p>
<div class="scroll"><table><thead><tr><th>Topic</th><th class="num">Comments</th><th>Sentiment</th><th class="num">Negative</th><th class="num">Positive</th>{trend_head}<th class="num">Impact</th></tr></thead><tbody>{_topic_rows(a)}</tbody></table></div></section>
<section><h2>Why people feel this way</h2><div class="scroll"><table><thead><tr><th>Topic</th><th>Top reasons, negative</th><th>Top reasons, positive</th></tr></thead><tbody>{_reason_rows(a)}</tbody></table></div></section>
<section class="two"><div><h2>Common questions</h2><p class="lead">{faq_lead}</p><ol class="ranked">{_group_list(a.questions)}</ol></div><div><h2>Feature requests</h2><p class="lead">What customers ask for most often.</p><ol class="ranked">{_group_list(a.requests)}</ol></div></section>
<section><h2>Who is talking, and how</h2><div class="two"><div><h3>What people are doing</h3><div class="scroll"><table><thead><tr><th>Intent</th><th class="num">Comments</th><th class="num">Share</th></tr></thead><tbody>{intent_rows}</tbody></table></div></div>
<div><h3>Language</h3><div class="scroll"><table><thead><tr><th>Language</th><th class="num">Comments</th><th class="num">Share</th></tr></thead><tbody>{lang_rows}</tbody></table></div></div></div>{platform_html}</section>
<section><h2>How this report was made</h2><p class="verified"><strong>Traced to data.</strong> {verified}</p>
<ul class="plain facts-list"><li>Labels by source: {esc(source_text)}.</li>{agree_text}<li>{review:,} comments are flagged for human review in labelled_comments.csv.</li>{llm_line}<li>Phone numbers, long ID numbers and emails were masked before any model saw the text. Names and street addresses are not masked.</li></ul>
<h3>Decisions the agent made, in order</h3><ol class="steps">{steps}</ol>
<details><summary>All {len(a.book.facts)} facts behind this report</summary><div class="scroll"><table class="facts"><tbody>{facts}</tbody></table></div></details></section>
</main><div id="tip" role="status"></div><script>{JS}</script></body></html>"""


def md(text) -> str:
    """Comment text must not break a Markdown table or carry HTML into a page that renders the file."""
    return " ".join(str(text).split()).replace("|", "\\|").replace("<", "&lt;")


def csv_safe(column: pd.Series) -> pd.Series:
    """A cell starting with these characters runs as a formula in a spreadsheet, so a comment could carry one."""
    return column.astype(str).where(~column.astype(str).str.match(r"^[=+\-@]"), "'" + column.astype(str))


def render_markdown(title: str, a: Analysis, ins: Insights) -> str:
    total = len(a.rows)
    lines = [f"# {md(title)}: comment report", "", f"**{md(ins.headline)}**", "",
             f"{a.received:,} comments received, {total:,} analysed. Negative {pct(a.overall['negative'], total)}%, positive {pct(a.overall['positive'], total)}%, neutral {pct(a.overall['neutral'], total)}%.", "",
             "## What to do first", "", "| When | Action | Why | Owner | Evidence |", "| --- | --- | --- | --- | --- |"]
    for x in sorted(ins.actions, key=lambda z: list(URGENCY).index(z.urgency)):
        lines.append(f"| {URGENCY[x.urgency][0]} | {md(x.action)} | {md(x.why)} | {md(x.owner)} | {md(', '.join(x.fact_ids))} |")
    lines += ["", "## Key findings", ""] + [f"- {md(p.text)} ({md(', '.join(p.fact_ids))})" for p in ins.points]
    lines += ["", "## Topics", "", "| Topic | Comments | Negative | Positive | Impact |", "| --- | --- | --- | --- | --- |"]
    lines += [f"| {md(r.topic)} | {r.count:,} | {r.neg_share}% | {r.pos_share}% | {r.impact:,.0f} |" for r in a.topics.itertuples()]
    lines += ["", "## Common questions", ""] + [f"- ({g['count']:,}) {md(g['label'])}" for g in a.questions]
    lines += ["", "## Feature requests", ""] + [f"- ({g['count']:,}) {md(g['label'])}" for g in a.requests]
    lines += ["", "## Facts", ""] + [f"- **{f.id}**: {md(f.text)}" for f in a.book.facts]
    return "\n".join(lines) + "\n"


def write_all(out_dir, title, a: Analysis, ins: Insights, labelled, comments, log, usage, stats, removed, info, mermaid) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {"html": out / "report.html", "md": out / "report.md", "csv": out / "labelled_comments.csv", "json": out / "insights.json", "graph": out / "graph.mmd"}
    paths["html"].write_text(render_html(title, a, ins, usage, stats, removed, log.decisions, info), encoding="utf-8")
    paths["md"].write_text(render_markdown(title, a, ins), encoding="utf-8")
    # Only the masked text is saved, so personal data does not leak into the output files.
    labels = labelled[["key", "sentiment", "reason", "evidence", "intent", "topic", "risk", "llm_doubted", "source", "needs_review"]].reset_index(drop=True)
    table = comments[["id", "platform", "created_at", "clean_text", "lang", "status", "hold_from_llm", "engagement", "key"]].merge(labels, on="key", how="left").drop(columns="key")
    for column in ("id", "platform", "clean_text", "reason", "evidence", "topic"):
        table[column] = csv_safe(table[column].fillna(""))
    table.to_csv(paths["csv"], index=False)
    paths["json"].write_text(json.dumps({"title": title, "insights": ins.model_dump(), "facts": [vars(f) for f in a.book.facts], "usage": usage, "stats": stats, "removed_claims": removed, "input": info,
                                         "topics": a.topics.to_dict("records"), "alerts": [{k: v for k, v in x.items() if k != "examples"} for x in a.alerts], "llm_coverage": round(a.coverage, 4), "spikes": a.spikes}, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    paths["graph"].write_text(mermaid, encoding="utf-8")
    return {k: str(v) for k, v in paths.items()}


CSS = """
:root{color-scheme:light;--surface:#fcfcfb;--panel:#f4f4f1;--t1:#0b0b0b;--t2:#52514e;--t3:#73726c;--rule:#e2e1dc;--pos:#2a78d6;--neu:#a8a79f;--neg:#e34948;--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--critical:#d03b3b;--serious:#ec835a;--quiet:#a8a79f;--ink-on:#fff}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])){color-scheme:dark;--surface:#1a1a19;--panel:#242422;--t1:#f0efec;--t2:#c3c2b7;--t3:#9d9c92;--rule:#383835;--pos:#3987e5;--neu:#6f6e68;--neg:#e66767;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--quiet:#6f6e68}}
*{box-sizing:border-box}body{margin:0;background:var(--surface);color:var(--t1);font:16px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,"Noto Sans","Noto Sans Devanagari","Noto Sans Oriya",sans-serif;-webkit-text-size-adjust:100%}
main{max-width:1040px;margin:0 auto;padding:48px 24px 80px}
h1{font-size:34px;line-height:1.15;margin:4px 0 8px;letter-spacing:-.01em}h2{font-size:22px;line-height:1.25;margin:0 0 6px;letter-spacing:-.005em}h3{font-size:16px;margin:0 0 8px}
.eyebrow{margin:0;font-size:13px;font-weight:600;color:var(--t2);letter-spacing:.02em}.meta{color:var(--t3);font-size:13px;font-weight:400}.lead{color:var(--t2);margin:0 0 16px;max-width:72ch}.note{color:var(--t2);font-size:14px;margin:12px 0 0}
.headline{font-size:22px;line-height:1.35;font-weight:600;margin:28px 0 24px;max-width:36em;border-left:3px solid var(--t1);padding-left:16px}
.banner{background:var(--panel);border:1px solid var(--rule);border-radius:8px;padding:12px 16px;margin:20px 0 0;font-size:14px;color:var(--t2)}.banner strong{color:var(--t1)}
section{margin-top:48px}section.two,.two{display:grid;grid-template-columns:1fr 1fr;gap:32px}section.two{align-items:start}
.tiles{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin-bottom:16px}.tile{background:var(--panel);border-radius:8px;padding:14px 16px}.tile .label{font-size:13px;color:var(--t2)}.tile .value{font-size:28px;font-weight:600;line-height:1.2;margin:2px 0}
.bar{display:flex;gap:2px;width:100%}.bar.tall{height:28px}.bar.thin{height:8px;min-width:120px}.seg{display:flex;align-items:center;padding-left:8px;font-size:13px;font-weight:600;min-width:2px;color:#fff}
.bar.tall .seg{border-radius:4px}.bar.thin .seg:first-child{border-radius:4px 0 0 4px}.bar.thin .seg:last-child{border-radius:0 4px 4px 0}.seg-negative{background:var(--neg)}.seg-neutral{background:var(--neu)}.seg-neutral.seg{color:#0b0b0b}.seg-positive{background:var(--pos)}
.legend{display:flex;flex-wrap:wrap;gap:16px;margin:10px 0 0;font-size:13px;color:var(--t2)}.key{display:inline-flex;align-items:center;gap:6px}.key i{width:12px;height:12px;border-radius:3px;display:inline-block}
.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:14px}th{font-size:13px;font-weight:600;color:var(--t2);text-align:left;padding:8px 12px 8px 0;border-bottom:1px solid var(--rule);white-space:nowrap}td{padding:10px 12px 10px 0;border-bottom:1px solid var(--rule);vertical-align:top}.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}.cellbar{min-width:140px;vertical-align:middle}
.actions strong{font-weight:600}.why{color:var(--t2);font-size:13px;margin-top:2px}.badge{display:inline-flex;align-items:center;gap:6px;font-weight:600;font-size:13px;white-space:nowrap}.badge i{width:9px;height:9px;border-radius:50%;display:inline-block}.badge.critical i{background:var(--critical)}.badge.serious i{background:var(--serious)}.badge.quiet i{background:var(--quiet)}
.refs{font-size:12px;white-space:nowrap}.refs a{color:var(--t2);text-underline-offset:2px;margin-right:4px}
.alerts{display:grid;gap:12px}.alert{border:1px solid var(--rule);border-left:3px solid var(--critical);border-radius:8px;padding:14px 16px}.alert .meta{margin-left:8px}.quotes{margin:0;padding:0 0 0 18px;font-size:14px}.quotes li{margin:4px 0}.alert .doubt{margin:8px 0 0}
.points{padding-left:20px;margin:0;max-width:78ch}.points li{margin:8px 0}.plain{list-style:none;margin:0;padding:0}.plain li{margin:3px 0}
.ranked{list-style:none;padding:0;margin:0}.ranked li{display:flex;gap:12px;padding:8px 0;border-bottom:1px solid var(--rule);font-size:14px}.count{font-variant-numeric:tabular-nums;font-weight:600;min-width:32px;text-align:right}
.chart{margin-top:8px}.chart svg{width:100%;height:auto;display:block}.grid{stroke:var(--rule);stroke-width:1}.tick{fill:var(--t3);font-size:12px}.end{fill:var(--t2);font-size:12px}.series{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}.dot{stroke:var(--surface);stroke-width:2}.mark{stroke:var(--t3);stroke-width:1}.hit{fill:transparent}.cross{stroke:var(--t3);stroke-width:1}
details{margin-top:12px}summary{cursor:pointer;color:var(--t2);font-size:14px}details table{margin-top:8px}
.spaced{margin-top:28px}.verified{background:var(--panel);border-radius:8px;padding:12px 16px;margin:0 0 16px;font-size:14px}.facts-list{font-size:14px;color:var(--t2);margin-bottom:20px}.steps{padding-left:22px;font-size:14px;margin:0}.steps li{margin:6px 0}.step{display:inline-block;font-weight:600;margin-right:8px}
#tip{position:fixed;pointer-events:none;background:var(--t1);color:var(--surface);font-size:13px;padding:6px 10px;border-radius:6px;max-width:320px;opacity:0;transition:opacity .08s;z-index:5}
@media (max-width:760px){main{padding:24px 16px 56px}h1{font-size:28px}.tiles{grid-template-columns:repeat(2,1fr)}section.two,.two{grid-template-columns:1fr}.headline{font-size:19px}}
"""

JS = """
const tip=document.getElementById('tip');
document.addEventListener('pointermove',e=>{const t=e.target.closest&&e.target.closest('[data-tip]');if(!t){tip.style.opacity=0;document.querySelectorAll('.cross').forEach(c=>c.style.opacity=0);return}
tip.textContent=t.dataset.tip;tip.style.opacity=1;tip.style.left=Math.min(e.clientX+14,innerWidth-tip.offsetWidth-8)+'px';tip.style.top=(e.clientY+14)+'px';
if(t.dataset.x){const c=t.ownerSVGElement.querySelector('.cross');c.setAttribute('x1',t.dataset.x);c.setAttribute('x2',t.dataset.x);c.style.opacity=1}});
document.addEventListener('focusin',e=>{const t=e.target;if(t.dataset&&t.dataset.tip){const r=t.getBoundingClientRect();tip.textContent=t.dataset.tip;tip.style.opacity=1;tip.style.left=r.left+'px';tip.style.top=(r.bottom+6)+'px'}});
document.addEventListener('focusout',()=>{tip.style.opacity=0});
function openFact(hash){const row=hash&&hash.length>1&&document.getElementById(hash.slice(1));const box=row&&row.closest('details');if(box)box.open=true}
document.addEventListener('click',e=>{const a=e.target.closest&&e.target.closest('a[href^="#F"]');if(a)openFact(a.getAttribute('href'))});
openFact(location.hash);
"""
