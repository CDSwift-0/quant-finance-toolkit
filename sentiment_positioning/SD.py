# -*- coding: utf-8 -*-
from __future__ import annotations

import contextlib
import html
import importlib.util
import io
import math
import queue
import re
import threading
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import pandas as pd
import requests
import tkinter as tk
from tkinter import ttk


DESKTOP = Path(__file__).resolve().parent
HOME = Path.home()
AAII_PATH = DESKTOP / "aaii.py"
CFTC_PATH = DESKTOP / "CFTC.py"
CACHE_DIR = DESKTOP / ".sentiment_cache"
AAII_CACHE = CACHE_DIR / "aaii_daily.csv"
CFTC_CACHE = CACHE_DIR / "cftc_daily.csv"
CFTC_BASE_URL = "https://publicreporting.cftc.gov/resource/gpe5-46if.csv"
CFTC_START_DATE = "1995-01-01"
CFTC_MARKET_LIKE = "%S&P 500%"
CFTC_CHUNK_SIZE = 50000
CFTC_PREFERRED_MARKETS = [
    "E-MINI S&P 500",
    "S&P 500 STOCK INDEX",
    "S&P 500",
]
AAII_LOCAL_FILES = [
    DESKTOP / "aaii_sentiment_from_html.csv",
    DESKTOP / "aaii_sentiment.csv",
    DESKTOP / "AAII.csv",
    DESKTOP / "sentiment.xls",
    DESKTOP / "sentiment.xlsx",
    HOME / "Downloads" / "sentiment.xls",
    HOME / "Downloads" / "sentiment.xlsx",
]

AAII_URLS = [
    "https://www.aaii.com/sentimentsurvey/sent_results",
    "https://www.aaii.com/sentimentsurvey",
]
AAII_INSIGHTS_FEED = "https://insights.aaii.com/feed"
AAII_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7",
    "Referer": "https://www.aaii.com/sentimentsurvey",
}


def load_module(path: Path) -> Any:
    module_name = f"dashboard_source_{path.stem}_{uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Impossible de charger {path.name}.")
    module = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        spec.loader.exec_module(module)
    return module


def is_today_cache(path: Path) -> bool:
    return path.exists() and pd.Timestamp(path.stat().st_mtime, unit="s").date() == pd.Timestamp.today().date()


def read_cache(path: Path) -> Optional[pd.DataFrame]:
    if not is_today_cache(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def save_cache(df: pd.DataFrame, path: Path) -> None:
    CACHE_DIR.mkdir(exist_ok=True)
    df.to_csv(path, index=False)


def first_existing(columns: list[str], needles: list[str]) -> str:
    for needle in needles:
        needle = needle.lower()
        for col in columns:
            if needle in str(col).lower():
                return col
    raise KeyError(f"Colonne introuvable: {', '.join(needles)}")


def clean_number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str)
        .str.replace("%", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.replace("\u2212", "-", regex=False)
        .str.strip(),
        errors="coerce",
    )


def fmt_int(value: Any) -> str:
    try:
        if pd.isna(value):
            return "n.d."
        return f"{float(value):,.0f}".replace(",", " ")
    except Exception:
        return "n.d."


def fmt_pct(value: Any, digits: int = 1, signed: bool = True) -> str:
    try:
        if pd.isna(value):
            return "n.d."
        prefix = "+" if signed and float(value) > 0 else ""
        return f"{prefix}{float(value):.{digits}f}%"
    except Exception:
        return "n.d."


def fmt_float(value: Any, digits: int = 2) -> str:
    try:
        if pd.isna(value):
            return "n.d."
        return f"{float(value):.{digits}f}"
    except Exception:
        return "n.d."


def percentile_rank(series: pd.Series, value: Any) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty or pd.isna(value):
        return float("nan")
    return float((clean <= float(value)).mean() * 100)


def normalize_aaii(raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw.copy()
    raw.columns = [str(col).strip() for col in raw.columns]
    date_col = first_existing(raw.columns.tolist(), ["date", "week"])
    bull_col = first_existing(raw.columns.tolist(), ["bullish", "bull"])
    bear_col = first_existing(raw.columns.tolist(), ["bearish", "bear"])
    neutral_col = first_existing(raw.columns.tolist(), ["neutral"])
    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(raw[date_col], errors="coerce"),
            "Bullish": clean_number(raw[bull_col]),
            "Neutral": clean_number(raw[neutral_col]),
            "Bearish": clean_number(raw[bear_col]),
        }
    ).dropna(subset=["Date", "Bullish", "Bearish"])
    if df.empty:
        raise RuntimeError("Les données AAII sont vides après nettoyage.")
    df = df.sort_values("Date").reset_index(drop=True)
    df["Spread"] = df["Bullish"] - df["Bearish"]
    return df


def read_local_aaii() -> Optional[pd.DataFrame]:
    for path in AAII_LOCAL_FILES:
        if not path.exists():
            continue
        try:
            if path.suffix.lower() in {".xls", ".xlsx"}:
                tables = pd.read_excel(path, sheet_name=None)
                for table in tables.values():
                    try:
                        normalize_aaii(table)
                        return table
                    except Exception:
                        continue
            else:
                return pd.read_csv(path)
        except Exception:
            continue
    return None


def text_from_html(value: str) -> str:
    value = html.unescape(value or "")
    value = re.sub(r"<[^>]+>", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def fetch_aaii_insights() -> Optional[pd.DataFrame]:
    try:
        response = requests.get(AAII_INSIGHTS_FEED, headers=AAII_HEADERS, timeout=45)
        response.raise_for_status()
        root = ET.fromstring(response.content)
    except Exception:
        return None

    rows: list[dict[str, Any]] = []
    for item in root.findall(".//item"):
        title = text_from_html(item.findtext("title", ""))
        if "sentiment" not in title.lower():
            continue
        body_parts = [
            item.findtext("description", ""),
            item.findtext("{http://purl.org/rss/1.0/modules/content/}encoded", ""),
        ]
        text = text_from_html(" ".join(body_parts))
        bullish = re.search(r"Bullish:\s*([0-9]+(?:\.[0-9]+)?)%", text, re.I)
        neutral = re.search(r"Neutral:\s*([0-9]+(?:\.[0-9]+)?)%", text, re.I)
        bearish = re.search(r"Bearish:\s*([0-9]+(?:\.[0-9]+)?)%", text, re.I)
        if not (bullish and neutral and bearish):
            continue
        rows.append(
            {
                "Date": pd.to_datetime(item.findtext("pubDate", ""), errors="coerce"),
                "Bullish": float(bullish.group(1)),
                "Neutral": float(neutral.group(1)),
                "Bearish": float(bearish.group(1)),
            }
        )

    if not rows:
        return None
    return pd.DataFrame(rows).dropna(subset=["Date"])


def fetch_aaii_table() -> pd.DataFrame:
    local = read_local_aaii()
    if local is not None:
        return local

    session = requests.Session()
    session.headers.update(AAII_HEADERS)
    last_error: Optional[Exception] = None

    for url in AAII_URLS:
        try:
            response = session.get(url, timeout=45)
            response.raise_for_status()
            tables = pd.read_html(io.StringIO(response.text))
            for table in tables:
                joined = " ".join(str(col).lower() for col in table.columns)
                if "bull" in joined and "bear" in joined:
                    return table
        except Exception as exc:
            last_error = exc

    insights = fetch_aaii_insights()
    if insights is not None:
        return insights

    if last_error is not None:
        raise RuntimeError(
            "AAII bloque temporairement l'accès au site et aucun CSV local "
            "n'a été trouvé sur le Bureau."
        ) from last_error
    raise RuntimeError("Impossible de trouver les données AAII.")


def compute_aaii() -> dict[str, Any]:
    raw = read_cache(AAII_CACHE)
    if raw is None:
        raw = fetch_aaii_table()
        save_cache(raw, AAII_CACHE)

    df = normalize_aaii(raw)
    recent = df.tail(52).copy()
    latest = df.iloc[-1]
    spread = float(latest["Spread"])

    if spread > 15:
        regime = "Optimisme"
        note = "Les particuliers sont nettement plus haussiers que baissiers."
    elif spread < -15:
        regime = "Pessimisme"
        note = "La prudence domine nettement le sondage AAII."
    else:
        regime = "Équilibre"
        note = "Le sentiment reste proche d'une zone neutre."

    table = df.tail(12).sort_values("Date", ascending=False).copy()
    table["Date"] = table["Date"].dt.strftime("%Y-%m-%d")
    return {
        "available": True,
        "kpis": {
            "AAII Bullish": fmt_pct(latest["Bullish"], signed=False),
            "AAII Bearish": fmt_pct(latest["Bearish"], signed=False),
            "AAII Spread": fmt_pct(latest["Spread"]),
            "AAII Régime": regime,
        },
        "note": note,
        "chart": recent[["Date", "Spread"]],
        "table": table[["Date", "Bullish", "Neutral", "Bearish", "Spread"]],
    }


def unavailable_aaii(message: str) -> dict[str, Any]:
    empty = pd.DataFrame(columns=["Date", "Bullish", "Neutral", "Bearish", "Spread"])
    return {
        "available": False,
        "kpis": {
            "AAII Bullish": "n.d.",
            "AAII Bearish": "n.d.",
            "AAII Spread": "n.d.",
            "AAII Régime": "Source bloquée",
        },
        "note": message,
        "chart": pd.DataFrame(columns=["Date", "Spread"]),
        "table": empty,
    }


def request_cftc(params: dict[str, Any]) -> pd.DataFrame:
    response = requests.get(CFTC_BASE_URL, params=params, timeout=60)
    response.raise_for_status()
    if not response.text.strip():
        return pd.DataFrame()
    return pd.read_csv(io.StringIO(response.text))


def fetch_cftc_all(
    select: str,
    where: Optional[str] = None,
    group: Optional[str] = None,
    order: Optional[str] = None,
    limit: int = CFTC_CHUNK_SIZE,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    offset = 0
    while True:
        params: dict[str, Any] = {
            "$select": select,
            "$limit": limit,
            "$offset": offset,
        }
        if where:
            params["$where"] = where
        if group:
            params["$group"] = group
        if order:
            params["$order"] = order

        chunk = request_cftc(params)
        if chunk.empty:
            break
        frames.append(chunk)
        if len(chunk) < limit:
            break
        offset += limit

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def pick_cftc_market(markets: list[str]) -> str:
    if not markets:
        raise RuntimeError("Aucun marché CFTC disponible.")

    upper_markets = [(market, market.upper()) for market in markets]
    for preferred in CFTC_PREFERRED_MARKETS:
        preferred_upper = preferred.upper()
        for market, market_upper in upper_markets:
            if preferred_upper in market_upper:
                return market
    return sorted(markets, key=len)[0]


def compute_cftc() -> dict[str, Any]:
    df = read_cache(CFTC_CACHE)
    market_name = "S&P 500"
    if df is None:
        markets = fetch_cftc_all(
            select="market_and_exchange_names",
            where=f"market_and_exchange_names like '{CFTC_MARKET_LIKE}'",
            group="market_and_exchange_names",
            order="market_and_exchange_names",
            limit=50000,
        )
        if markets.empty:
            raise RuntimeError("Aucun marché CFTC n'a été trouvé pour S&P 500.")
        market_name = pick_cftc_market(markets["market_and_exchange_names"].dropna().astype(str).unique().tolist())
        escaped_market = market_name.replace("'", "''")
        where = (
            f"market_and_exchange_names = '{escaped_market}' "
            f"and report_date_as_yyyy_mm_dd >= '{CFTC_START_DATE}'"
        )
        df = fetch_cftc_all(
            select=(
                "report_date_as_yyyy_mm_dd,asset_mgr_positions_long,"
                "asset_mgr_positions_short,open_interest_all"
            ),
            where=where,
            order="report_date_as_yyyy_mm_dd",
            limit=CFTC_CHUNK_SIZE,
        )
        if df.empty:
            raise RuntimeError("Les données CFTC sont vides pour le marché choisi.")
        df["market_and_exchange_names"] = market_name
        save_cache(df, CFTC_CACHE)
    elif "market_and_exchange_names" in df.columns and not df["market_and_exchange_names"].dropna().empty:
        market_name = str(df["market_and_exchange_names"].dropna().iloc[-1])

    cftc = pd.DataFrame(
        {
            "Date": pd.to_datetime(df["report_date_as_yyyy_mm_dd"], errors="coerce"),
            "Long": pd.to_numeric(df["asset_mgr_positions_long"], errors="coerce"),
            "Short": pd.to_numeric(df["asset_mgr_positions_short"], errors="coerce"),
            "Open Interest": pd.to_numeric(df["open_interest_all"], errors="coerce"),
        }
    ).dropna(subset=["Date", "Long", "Short"])
    if cftc.empty:
        raise RuntimeError("Les colonnes CFTC attendues n'ont pas pu être lues.")

    cftc = cftc.sort_values("Date").reset_index(drop=True)
    cftc["Net"] = cftc["Long"] - cftc["Short"]
    cftc["Net % OI"] = cftc["Net"] / cftc["Open Interest"].replace(0, pd.NA) * 100
    latest = cftc.iloc[-1]
    recent = cftc.tail(104).copy()
    year = cftc.tail(52)
    three_years = cftc.tail(156)
    net_pct = latest["Net % OI"]
    net_4w_change = latest["Net"] - cftc.iloc[-5]["Net"] if len(cftc) >= 5 else pd.NA
    net_12w_change = latest["Net"] - cftc.iloc[-13]["Net"] if len(cftc) >= 13 else pd.NA
    net_pct_4w_change = latest["Net % OI"] - cftc.iloc[-5]["Net % OI"] if len(cftc) >= 5 else pd.NA
    mean_52 = year["Net % OI"].mean()
    median_52 = year["Net % OI"].median()
    std_52 = year["Net % OI"].std()
    zscore_52 = (latest["Net % OI"] - mean_52) / std_52 if pd.notna(std_52) and std_52 else pd.NA
    pct_3y = percentile_rank(three_years["Net % OI"], latest["Net % OI"])
    long_share = latest["Long"] / (latest["Long"] + latest["Short"]) * 100
    short_share = latest["Short"] / (latest["Long"] + latest["Short"]) * 100

    if pd.notna(net_pct) and net_pct > 12:
        regime = "Long marqué"
        note = "Les asset managers gardent une exposition nette clairement positive."
    elif pd.notna(net_pct) and net_pct < -5:
        regime = "Short net"
        note = "Le positionnement des asset managers est défensif."
    else:
        regime = "Modéré"
        note = "Le positionnement reste lisible mais sans excès extrême."

    table = cftc.tail(12).sort_values("Date", ascending=False).copy()
    table["Date"] = table["Date"].dt.strftime("%Y-%m-%d")
    direction = "se renforce" if pd.notna(net_4w_change) and net_4w_change > 0 else "se réduit"
    crowding = "élevé" if pd.notna(pct_3y) and pct_3y >= 80 else "faible" if pd.notna(pct_3y) and pct_3y <= 20 else "intermédiaire"
    trend_note = (
        f"Le net Asset Managers {direction} sur 4 semaines "
        f"({fmt_int(net_4w_change)} contrats)."
    )
    benchmark_note = (
        f"Le Net % OI est au percentile {fmt_float(pct_3y, 0)} sur trois ans, "
        f"un niveau de concentration {crowding}."
    )
    balance_note = (
        f"La structure long/short est à {fmt_float(long_share, 1)}% long "
        f"contre {fmt_float(short_share, 1)}% short."
    )
    return {
        "available": True,
        "market": market_name,
        "kpis": {
            "CFTC Net AM": fmt_int(latest["Net"]),
            "CFTC Net % OI": fmt_pct(latest["Net % OI"]),
            "CFTC Long / Short": f"{fmt_float(latest['Long'] / latest['Short'])}x",
            "CFTC Régime": regime,
        },
        "note": note,
        "chart": recent[["Date", "Net % OI"]],
        "table": table[["Date", "Long", "Short", "Net", "Net % OI"]],
        "analytics": [
            ("Moyenne 52 sem.", fmt_pct(mean_52), "Référence récente du Net % OI."),
            ("Médiane 52 sem.", fmt_pct(median_52), "Point central du positionnement récent."),
            ("Percentile 3 ans", f"{fmt_float(pct_3y, 0)}%", "Niveau actuel versus historique récent."),
            ("Z-score 52 sem.", fmt_float(zscore_52, 2), "Écart à la moyenne en écarts-types."),
            ("Variation 4 sem.", fmt_int(net_4w_change), "Accélération ou détente récente du net."),
            ("Variation 12 sem.", fmt_int(net_12w_change), "Mouvement de fond sur un trimestre."),
            ("Delta Net % OI 4 sem.", fmt_pct(net_pct_4w_change), "Variation relative à l'open interest."),
            ("Open Interest", fmt_int(latest["Open Interest"]), "Taille totale du marché futures retenu."),
        ],
        "conclusions": [trend_note, benchmark_note, balance_note],
    }


def unavailable_cftc(message: str) -> dict[str, Any]:
    empty = pd.DataFrame(columns=["Date", "Long", "Short", "Net", "Net % OI"])
    return {
        "available": False,
        "market": "Source indisponible",
        "kpis": {
            "CFTC Net AM": "n.d.",
            "CFTC Net % OI": "n.d.",
            "CFTC Long / Short": "n.d.",
            "CFTC Régime": "Source bloquée",
        },
        "note": message,
        "chart": pd.DataFrame(columns=["Date", "Net % OI"]),
        "table": empty,
        "analytics": [],
        "conclusions": [],
    }


class Dashboard(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("AAII / CFTC")
        self.geometry("1460x980")
        self.minsize(1180, 780)
        self.colors = {
            "bg": "#f3f1ec",
            "panel": "#fbfaf7",
            "panel_alt": "#f6f4ef",
            "ink": "#202126",
            "muted": "#6f7482",
            "line": "#d8d3ca",
            "gold": "#a98f52",
            "green": "#16806d",
            "red": "#bf4646",
            "blue": "#4b6f8f",
        }
        self.configure(bg=self.colors["bg"])
        self.queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.spinner_job: Optional[str] = None
        self.spinner_angle = 0
        self.status = tk.StringVar(value="Prêt")
        self._setup_styles()
        self._build_shell()
        self._bind_scrolling()
        self.after(200, self.refresh)

    def _setup_styles(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(
            "Dashboard.Treeview",
            background=self.colors["panel"],
            fieldbackground=self.colors["panel"],
            foreground=self.colors["ink"],
            borderwidth=0,
            rowheight=34,
            font=("Avenir Next", 12),
        )
        style.configure(
            "Dashboard.Treeview.Heading",
            background=self.colors["panel_alt"],
            foreground=self.colors["muted"],
            borderwidth=0,
            font=("Avenir Next", 11, "bold"),
        )

    def _build_shell(self) -> None:
        self.canvas = tk.Canvas(self, bg=self.colors["bg"], highlightthickness=0, borderwidth=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.page = tk.Frame(self.canvas, bg=self.colors["bg"])
        self.window_id = self.canvas.create_window((0, 0), window=self.page, anchor="nw")
        self.page.bind("<Configure>", self._sync_scroll_region)
        self.canvas.bind("<Configure>", self._sync_canvas_width)
        self._header()
        self.kpi_grid = tk.Frame(self.page, bg=self.colors["bg"])
        self.kpi_grid.pack(fill="x", padx=34, pady=(0, 22))
        self.body = tk.Frame(self.page, bg=self.colors["bg"])
        self.body.pack(fill="both", expand=True, padx=34, pady=(0, 34))

    def _header(self) -> None:
        head = tk.Frame(self.page, bg=self.colors["bg"])
        head.pack(fill="x", padx=34, pady=(28, 20))
        left = tk.Frame(head, bg=self.colors["bg"])
        left.pack(side="left", fill="x", expand=True)
        tk.Label(left, text="SENTIMENT & POSITIONNEMENT", bg=self.colors["bg"], fg=self.colors["gold"], font=("Avenir Next", 11, "bold")).pack(anchor="w")
        tk.Label(left, text="AAII / CFTC", bg=self.colors["bg"], fg=self.colors["ink"], font=("Avenir Next", 34, "bold")).pack(anchor="w", pady=(3, 1))
        tk.Label(left, text="Lecture croisée du sentiment des investisseurs particuliers et du positionnement institutionnel sur futures.", bg=self.colors["bg"], fg=self.colors["muted"], font=("Avenir Next", 13)).pack(anchor="w")
        right = tk.Frame(head, bg=self.colors["bg"])
        right.pack(side="right", anchor="ne")
        self.spinner = tk.Canvas(right, width=28, height=28, bg=self.colors["bg"], highlightthickness=0)
        self.spinner.pack(side="left", padx=(0, 12), pady=(6, 0))
        tk.Label(right, textvariable=self.status, bg=self.colors["bg"], fg=self.colors["muted"], font=("Avenir Next", 12)).pack(side="left", padx=(0, 14), pady=(6, 0))
        tk.Button(right, text="Actualiser", command=self.refresh, bg=self.colors["ink"], fg="white", activebackground="#34363c", activeforeground="white", borderwidth=0, padx=16, pady=9, cursor="hand2", font=("Avenir Next", 12, "bold")).pack(side="left")

    def _sync_scroll_region(self, _event: Optional[tk.Event] = None) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _sync_canvas_width(self, event: tk.Event) -> None:
        self.canvas.itemconfigure(self.window_id, width=event.width)

    def _bind_scrolling(self) -> None:
        def wheel(event: tk.Event) -> None:
            if event.num == 4:
                self.canvas.yview_scroll(-3, "units")
            elif event.num == 5:
                self.canvas.yview_scroll(3, "units")
            else:
                delta = getattr(event, "delta", 0)
                if delta:
                    step = -3 if delta > 0 else 3
                    self.canvas.yview_scroll(step, "units")
        self.bind_all("<MouseWheel>", wheel, add="+")
        self.bind_all("<Button-4>", wheel, add="+")
        self.bind_all("<Button-5>", wheel, add="+")

    def refresh(self) -> None:
        self.status.set("Calcul en cours")
        self._start_spinner()
        self._clear(self.body)
        self._clear(self.kpi_grid)
        self._loading_panel()
        threading.Thread(target=self._worker, daemon=True).start()
        self.after(150, self._poll)

    def _worker(self) -> None:
        try:
            try:
                aaii = compute_aaii()
            except Exception as exc:
                aaii = unavailable_aaii(str(exc))
            try:
                cftc = compute_cftc()
            except Exception as exc:
                cftc = unavailable_cftc(str(exc))
            self.queue.put(("ok", {"aaii": aaii, "cftc": cftc}))
        except Exception as exc:
            self.queue.put(("error", exc))

    def _poll(self) -> None:
        try:
            state, payload = self.queue.get_nowait()
        except queue.Empty:
            self.after(150, self._poll)
            return
        self._stop_spinner()
        if state == "error":
            self.status.set("Erreur")
            self._clear(self.body)
            self._error_panel(str(payload))
            return
        self.status.set("Données à jour")
        self._render(payload)

    def _start_spinner(self) -> None:
        self.spinner_angle = 0
        def animate() -> None:
            self.spinner.delete("all")
            for i in range(9):
                color = self._blend("#d8d3ca", "#202126", i / 8)
                self.spinner.create_arc(5, 5, 23, 23, start=self.spinner_angle + i * 28, extent=16, style="arc", width=2, outline=color)
            self.spinner_angle = (self.spinner_angle + 18) % 360
            self.spinner_job = self.after(55, animate)
        animate()

    def _stop_spinner(self) -> None:
        if self.spinner_job is not None:
            self.after_cancel(self.spinner_job)
            self.spinner_job = None
        self.spinner.delete("all")

    def _blend(self, a: str, b: str, t: float) -> str:
        def rgb(value: str) -> tuple[int, int, int]:
            value = value.lstrip("#")
            return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))
        ar, ag, ab = rgb(a)
        br, bg, bb = rgb(b)
        return f"#{int(ar + (br - ar) * t):02x}{int(ag + (bg - ag) * t):02x}{int(ab + (bb - ab) * t):02x}"

    def _render(self, data: dict[str, Any]) -> None:
        self._clear(self.body)
        self._clear(self.kpi_grid)
        aaii = data["aaii"]
        cftc = data["cftc"]
        for index, (label, value) in enumerate({**aaii["kpis"], **cftc["kpis"]}.items()):
            self._kpi_card(index, label, value)
        self._summary_band(aaii, cftc)
        aaii_section = self._section("AAII", "Sentiment des investisseurs particuliers, avec spread Bullish - Bearish.")
        if aaii["available"]:
            self._chart_and_table(aaii_section, aaii["chart"], "Spread AAII", "Spread", self.colors["blue"], aaii["table"], ["Date", "Bullish", "Neutral", "Bearish", "Spread"], "aaii")
        else:
            self._notice(aaii_section, aaii["note"])
        cftc_section = self._section("CFTC", "Positionnement Asset Managers sur le contrat retenu.", meta=cftc["market"])
        if cftc["available"]:
            self._chart_and_table(cftc_section, cftc["chart"], "Net % Open Interest", "Net % OI", self.colors["green"], cftc["table"], ["Date", "Long", "Short", "Net", "Net % OI"], "cftc")
            self._cftc_analysis(cftc_section, cftc)
        else:
            self._notice(cftc_section, cftc["note"])

    def _kpi_card(self, index: int, label: str, value: str) -> None:
        cols = 4
        card = tk.Frame(self.kpi_grid, bg=self.colors["panel"])
        card.grid(row=index // cols, column=index % cols, sticky="ew", padx=7, pady=7)
        self.kpi_grid.grid_columnconfigure(index % cols, weight=1)
        lower = label.lower()
        accent = self.colors["red"] if "bear" in lower or "short" in lower else self.colors["green"] if "bull" in lower or "long" in lower or "net" in lower else self.colors["gold"]
        tk.Frame(card, bg=accent, height=3).pack(fill="x")
        inner = tk.Frame(card, bg=self.colors["panel"])
        inner.pack(fill="both", expand=True, padx=18, pady=16)
        tk.Label(inner, text=label.upper(), bg=self.colors["panel"], fg=self.colors["muted"], font=("Avenir Next", 11, "bold")).pack(anchor="w")
        tk.Label(inner, text=value, bg=self.colors["panel"], fg=self.colors["ink"], font=("Avenir Next", 25, "bold")).pack(anchor="w", pady=(8, 0))

    def _summary_band(self, aaii: dict[str, Any], cftc: dict[str, Any]) -> None:
        band = tk.Frame(self.body, bg=self.colors["panel"])
        band.pack(fill="x", pady=(0, 18))
        tk.Frame(band, bg=self.colors["gold"], width=4).pack(side="left", fill="y")
        text = tk.Frame(band, bg=self.colors["panel"])
        text.pack(side="left", fill="both", expand=True, padx=18, pady=16)
        tk.Label(text, text="Lecture rapide", bg=self.colors["panel"], fg=self.colors["ink"], font=("Avenir Next", 17, "bold")).pack(anchor="w")
        tk.Label(text, text=f"{aaii['note']} {cftc['note']}", bg=self.colors["panel"], fg=self.colors["muted"], justify="left", wraplength=1120, font=("Avenir Next", 13)).pack(anchor="w", pady=(6, 0))

    def _section(self, title: str, subtitle: str, meta: Optional[str] = None) -> tk.Frame:
        section = tk.Frame(self.body, bg=self.colors["panel"])
        section.pack(fill="x", pady=(0, 22))
        header = tk.Frame(section, bg=self.colors["panel"])
        header.pack(fill="x", padx=20, pady=(20, 12))
        left = tk.Frame(header, bg=self.colors["panel"])
        left.pack(side="left", fill="x", expand=True)
        tk.Label(left, text=title, bg=self.colors["panel"], fg=self.colors["ink"], font=("Avenir Next", 23, "bold")).pack(anchor="w")
        tk.Label(left, text=subtitle, bg=self.colors["panel"], fg=self.colors["muted"], font=("Avenir Next", 12)).pack(anchor="w", pady=(3, 0))
        if meta:
            tk.Label(header, text=meta, bg=self.colors["panel_alt"], fg=self.colors["muted"], padx=12, pady=7, font=("Avenir Next", 11, "bold")).pack(side="right", anchor="ne")
        body = tk.Frame(section, bg=self.colors["panel"])
        body.pack(fill="x", padx=20, pady=(0, 20))
        return body

    def _cftc_analysis(self, parent: tk.Frame, cftc: dict[str, Any]) -> None:
        analysis = tk.Frame(parent, bg=self.colors["panel"])
        analysis.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(18, 0))
        analysis.grid_columnconfigure(0, weight=1)
        analysis.grid_columnconfigure(1, weight=1)

        conclusion = tk.Frame(analysis, bg=self.colors["panel_alt"])
        conclusion.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 14))
        tk.Label(
            conclusion,
            text="Lecture CFTC",
            bg=self.colors["panel_alt"],
            fg=self.colors["ink"],
            font=("Avenir Next", 18, "bold"),
        ).pack(anchor="w", padx=18, pady=(16, 6))
        for line in cftc["conclusions"]:
            tk.Label(
                conclusion,
                text=line,
                bg=self.colors["panel_alt"],
                fg=self.colors["muted"],
                justify="left",
                wraplength=1180,
                font=("Avenir Next", 13),
            ).pack(anchor="w", padx=18, pady=(0, 6))
        tk.Label(
            conclusion,
            text=(
                "Interprétation : un percentile élevé signale un positionnement déjà concentré ; "
                "un percentile faible signale une exposition institutionnelle plus légère ou défensive."
            ),
            bg=self.colors["panel_alt"],
            fg=self.colors["muted"],
            justify="left",
            wraplength=1180,
            font=("Avenir Next", 12),
        ).pack(anchor="w", padx=18, pady=(4, 16))

        metrics = tk.Frame(analysis, bg=self.colors["panel"])
        metrics.grid(row=1, column=0, columnspan=2, sticky="ew")
        for col in range(4):
            metrics.grid_columnconfigure(col, weight=1)

        for index, (label, value, detail) in enumerate(cftc["analytics"]):
            card = tk.Frame(metrics, bg=self.colors["panel_alt"])
            card.grid(row=index // 4, column=index % 4, sticky="nsew", padx=6, pady=6)
            tk.Label(
                card,
                text=label.upper(),
                bg=self.colors["panel_alt"],
                fg=self.colors["muted"],
                font=("Avenir Next", 10, "bold"),
            ).pack(anchor="w", padx=14, pady=(13, 0))
            tk.Label(
                card,
                text=value,
                bg=self.colors["panel_alt"],
                fg=self.colors["ink"],
                font=("Avenir Next", 20, "bold"),
            ).pack(anchor="w", padx=14, pady=(5, 0))
            tk.Label(
                card,
                text=detail,
                bg=self.colors["panel_alt"],
                fg=self.colors["muted"],
                wraplength=250,
                justify="left",
                font=("Avenir Next", 11),
            ).pack(anchor="w", padx=14, pady=(5, 14))

    def _chart_and_table(self, parent: tk.Frame, chart_df: pd.DataFrame, chart_title: str, value_col: str, color: str, table_df: pd.DataFrame, columns: list[str], value_format: str) -> None:
        parent.grid_columnconfigure(0, weight=3)
        parent.grid_columnconfigure(1, weight=2)
        chart_box = tk.Frame(parent, bg=self.colors["panel_alt"])
        chart_box.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        tk.Label(chart_box, text=chart_title, bg=self.colors["panel_alt"], fg=self.colors["muted"], font=("Avenir Next", 12, "bold")).pack(anchor="w", padx=16, pady=(14, 0))
        chart = tk.Canvas(chart_box, height=275, bg=self.colors["panel_alt"], highlightthickness=0)
        chart.pack(fill="x", expand=True, padx=14, pady=12)
        chart.bind("<Configure>", lambda event, c=chart, df=chart_df, col=value_col, line=color: self._draw_chart(c, df, col, line))
        table_box = tk.Frame(parent, bg=self.colors["panel_alt"])
        table_box.grid(row=0, column=1, sticky="nsew", padx=(12, 0))
        tk.Label(table_box, text="Dernières observations", bg=self.colors["panel_alt"], fg=self.colors["muted"], font=("Avenir Next", 12, "bold")).pack(anchor="w", padx=16, pady=(14, 0))
        self._table(table_box, table_df, columns, value_format)

    def _draw_chart(self, canvas: tk.Canvas, df: pd.DataFrame, value_col: str, color: str) -> None:
        canvas.delete("all")
        width = max(canvas.winfo_width(), 320)
        height = max(canvas.winfo_height(), 220)
        pad_x = 42
        pad_y = 28
        series = pd.to_numeric(df[value_col], errors="coerce").dropna() if value_col in df else pd.Series(dtype=float)
        if series.empty:
            return
        values = series.tolist()
        low, high = min(values), max(values)
        if math.isclose(low, high):
            low -= 1
            high += 1
        zero = height - pad_y - ((0 - low) / (high - low)) * (height - 2 * pad_y) if low < 0 < high else None
        for i in range(5):
            y = pad_y + i * (height - 2 * pad_y) / 4
            canvas.create_line(pad_x, y, width - pad_x, y, fill="#e2ded6", width=1)
        if zero is not None:
            canvas.create_line(pad_x, zero, width - pad_x, zero, fill="#c9c1b5", width=1)
        if value_col == "Net % OI":
            for label, stat, line_color in [
                ("moy.", pd.Series(values).mean(), "#a98f52"),
                ("méd.", pd.Series(values).median(), "#8d929d"),
            ]:
                y_stat = height - pad_y - ((stat - low) / (high - low)) * (height - 2 * pad_y)
                canvas.create_line(pad_x, y_stat, width - pad_x, y_stat, fill=line_color, width=1, dash=(5, 5))
                canvas.create_text(
                    width - pad_x,
                    y_stat - 8,
                    text=f"{label} {fmt_pct(stat)}",
                    anchor="e",
                    fill=line_color,
                    font=("Avenir Next", 10, "bold"),
                )
        points: list[float] = []
        for index, value in enumerate(values):
            x = pad_x + index * (width - 2 * pad_x) / max(len(values) - 1, 1)
            y = height - pad_y - ((value - low) / (high - low)) * (height - 2 * pad_y)
            points.extend([x, y])
        if len(points) >= 4:
            canvas.create_line(points, fill=color, width=3, smooth=True)
        x, y = points[-2], points[-1]
        canvas.create_oval(x - 4, y - 4, x + 4, y + 4, fill=color, outline=color)
        canvas.create_text(width - pad_x, pad_y - 8, text=fmt_pct(values[-1]), anchor="e", fill=color, font=("Avenir Next", 12, "bold"))

    def _table(self, parent: tk.Frame, df: pd.DataFrame, columns: list[str], value_format: str) -> None:
        tree = ttk.Treeview(parent, columns=columns, show="headings", height=12, style="Dashboard.Treeview")
        tree.pack(fill="both", expand=True, padx=12, pady=12)
        widths = {"Date": 112, "Bullish": 86, "Neutral": 86, "Bearish": 86, "Spread": 88, "Long": 104, "Short": 104, "Net": 104, "Net % OI": 92}
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=widths.get(col, 90), anchor="e" if col != "Date" else "w", stretch=True)
        tree.tag_configure("even", background=self.colors["panel"])
        tree.tag_configure("odd", background=self.colors["panel_alt"])
        tree.tag_configure("positive", foreground=self.colors["green"])
        tree.tag_configure("negative", foreground=self.colors["red"])
        for index, (_, row) in enumerate(df.iterrows()):
            values = [self._format_cell(row[col], col) for col in columns]
            tags = ["even" if index % 2 == 0 else "odd"]
            marker = row["Spread"] if "Spread" in row else row.get("Net % OI", 0)
            try:
                tags.append("positive" if float(marker) >= 0 else "negative")
            except Exception:
                pass
            tree.insert("", "end", values=values, tags=tuple(tags))

    def _format_cell(self, value: Any, col: str) -> str:
        if col == "Date":
            return str(value)
        if col in {"Bullish", "Neutral", "Bearish", "Spread", "Net % OI"}:
            return fmt_pct(value, signed=col in {"Spread", "Net % OI"})
        if col in {"Long", "Short", "Net"}:
            return fmt_int(value)
        return str(value)

    def _notice(self, parent: tk.Frame, message: str) -> None:
        panel = tk.Frame(parent, bg=self.colors["panel_alt"])
        panel.pack(fill="x")
        tk.Label(panel, text="Source AAII indisponible", bg=self.colors["panel_alt"], fg=self.colors["red"], font=("Avenir Next", 18, "bold")).pack(anchor="w", padx=18, pady=(18, 4))
        tk.Label(panel, text=message, bg=self.colors["panel_alt"], fg=self.colors["muted"], wraplength=1100, justify="left", font=("Avenir Next", 13)).pack(anchor="w", padx=18, pady=(0, 18))

    def _loading_panel(self) -> None:
        panel = tk.Frame(self.body, bg=self.colors["panel"])
        panel.pack(fill="x", pady=(8, 18))
        tk.Label(panel, text="Chargement des données", bg=self.colors["panel"], fg=self.colors["ink"], font=("Avenir Next", 22, "bold")).pack(anchor="w", padx=22, pady=(22, 4))
        tk.Label(panel, text="Récupération AAII, lecture CFTC et préparation des tableaux.", bg=self.colors["panel"], fg=self.colors["muted"], font=("Avenir Next", 13)).pack(anchor="w", padx=22, pady=(0, 22))

    def _error_panel(self, message: str) -> None:
        panel = tk.Frame(self.body, bg=self.colors["panel"])
        panel.pack(fill="x", pady=(8, 18))
        tk.Label(panel, text="Impossible de construire le tableau de bord", bg=self.colors["panel"], fg=self.colors["red"], font=("Avenir Next", 22, "bold")).pack(anchor="w", padx=22, pady=(22, 4))
        tk.Label(panel, text=message, bg=self.colors["panel"], fg=self.colors["muted"], wraplength=1100, justify="left", font=("Avenir Next", 13)).pack(anchor="w", padx=22, pady=(0, 22))

    def _clear(self, widget: tk.Widget) -> None:
        for child in widget.winfo_children():
            child.destroy()


def check_files() -> None:
    missing = [str(path) for path in [AAII_PATH, CFTC_PATH] if not path.exists()]
    if missing:
        raise FileNotFoundError("Fichier introuvable: " + ", ".join(missing))


def main() -> None:
    check_files()
    app = Dashboard()
    app.mainloop()


if __name__ == "__main__":
    import sys
    if "--check" in sys.argv:
        check_files()
        print("OK - fichiers sources trouvés.")
    else:
        main()
