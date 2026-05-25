import os
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

DATA_DIR = Path(os.getenv("KURU_DATA_DIR", "data"))
TRADES_JSONL = DATA_DIR / "trades.jsonl"
TRADES_CSV = DATA_DIR / "trades.csv"
SIGNALS_JSONL = DATA_DIR / "signals.jsonl"

app = FastAPI(title="Project Kuru API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return items


def parse_timestamp(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except Exception:
        return datetime.min


def compute_stats(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(trades)
    wins = len([t for t in trades if t.get("pnl", 0) > 0])
    losses = len([t for t in trades if t.get("pnl", 0) < 0])
    pnl_total = sum(t.get("pnl", 0) for t in trades)
    avg_pnl = pnl_total / total if total else 0
    avg_win = (
        sum(t.get("pnl", 0) for t in trades if t.get("pnl", 0) > 0) / wins
        if wins
        else 0
    )
    avg_loss = (
        sum(t.get("pnl", 0) for t in trades if t.get("pnl", 0) < 0) / losses
        if losses
        else 0
    )
    starting_balance = float(os.getenv("KURU_STARTING_BALANCE", "0") or 0)
    roi_pct = (pnl_total / starting_balance * 100) if starting_balance else 0
    win_rate = (wins / total * 100) if total else 0

    return {
        "total_trades": total,
        "wins": wins,
        "losses": losses,
        "win_rate": round(win_rate, 2),
        "total_pnl": round(pnl_total, 8),
        "avg_pnl": round(avg_pnl, 8),
        "avg_win": round(avg_win, 8),
        "avg_loss": round(avg_loss, 8),
        "roi_pct": round(roi_pct, 4),
        "starting_balance": starting_balance,
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/trades")
def trades():
    items = read_jsonl(TRADES_JSONL)
    items.sort(key=lambda x: parse_timestamp(x.get("exit_time", "")))
    return {"trades": items}


@app.get("/trades.csv", response_class=PlainTextResponse)
def trades_csv():
    if not TRADES_CSV.exists():
        return ""
    return TRADES_CSV.read_text(encoding="utf-8")


@app.get("/signals")
def signals():
    items = read_jsonl(SIGNALS_JSONL)
    items.sort(key=lambda x: parse_timestamp(x.get("timestamp", "")))
    return {"signals": items}


@app.get("/stats")
def stats():
    items = read_jsonl(TRADES_JSONL)
    return compute_stats(items)


@app.get("/equity")
def equity():
    items = read_jsonl(TRADES_JSONL)
    items.sort(key=lambda x: parse_timestamp(x.get("exit_time", "")))
    equity_curve = []
    cumulative = 0.0
    for trade in items:
        pnl = trade.get("pnl", 0)
        cumulative += pnl
        equity_curve.append({
            "timestamp": trade.get("exit_time"),
            "equity": round(cumulative, 8)
        })
    return {"equity": equity_curve}
