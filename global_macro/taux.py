#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Macro Intelligence — reconstruction complète.

Objectifs :
- interface Tkinter fluide, sans animation de scroll artificielle ;
- une seule source réseau principale : FRED ;
- aucune clé API ;
- cache local immédiat ;
- chaque série est téléchargée indépendamment : une panne ne bloque pas les autres ;
- aucun téléchargement lors d'un changement d'horizon ;
- tous les graphiques ont la même taille, sauf le dernier en pleine largeur ;
- dernier graphique volontairement simple : variation des grands indicateurs sur 12 mois.

Dépendance :
    python3 -m pip install matplotlib

Le réseau n'est jamais exécuté dans le thread Tkinter.
"""

from __future__ import annotations

import csv
import io
import json
import math
import queue
import ssl
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import tkinter as tk
from tkinter import ttk

# Matplotlib est importé après l'ouverture de la fenêtre.
mdates: Any = None
Figure: Any = None
FigureCanvasTkAgg: Any = None


def load_matplotlib() -> None:
    global mdates, Figure, FigureCanvasTkAgg
    if Figure is not None:
        return
    import matplotlib.dates as _mdates
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg as _FigureCanvasTkAgg
    from matplotlib.figure import Figure as _Figure

    mdates = _mdates
    Figure = _Figure
    FigureCanvasTkAgg = _FigureCanvasTkAgg


# =============================================================================
# Configuration
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / ".cache_macro_intelligence_v2"

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd=1989-01-01"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125 Safari/537.36"
)
NETWORK_TIMEOUT = 7.0
NETWORK_ATTEMPTS = 2

# Toutes les données viennent du même fournisseur. C'est volontaire : le dashboard
# reste cohérent et beaucoup moins fragile que lorsqu'il dépend de plusieurs sites.
SERIES = {
    "fed_funds": {"id": "DFF", "transform": "raw", "ttl": 2 * 3600},
    "cpi": {"id": "CPIAUCSL", "transform": "yoy", "ttl": 12 * 3600},
    "core_cpi": {"id": "CPILFESL", "transform": "yoy", "ttl": 12 * 3600},
    "pce": {"id": "PCEPI", "transform": "yoy", "ttl": 12 * 3600},
    "yield_curve": {"id": "T10Y2Y", "transform": "raw", "ttl": 2 * 3600},
    "recession_probability": {"id": "RECPROUSM156N", "transform": "raw", "ttl": 12 * 3600},
}

PERIODS: dict[str, int | None] = {
    "1 an": 365,
    "3 ans": 1095,
    "5 ans": 1825,
    "10 ans": 3650,
    "20 ans": 7300,
    "Depuis 1990": None,
}
DEFAULT_PERIOD = "10 ans"
UNAVAILABLE_MESSAGE = "Bientôt mis à jour"

FONT = "Avenir Next"

COLORS = {
    "bg": "#f3f5f7",
    "panel": "#ffffff",
    "panel_alt": "#f8fafc",
    "panel_soft": "#f1f5f9",
    "ink": "#0f172a",
    "ink_soft": "#1e293b",
    "muted": "#64748b",
    "muted_light": "#94a3b8",
    "line": "#cbd5e1",
    "line_soft": "#e2e8f0",
    "grid": "#e7edf3",
    "gold": "#b7791f",
    "gold_bg": "#f7ecd5",
    "green": "#0f9f6e",
    "green_bg": "#ddf3ea",
    "red": "#d64545",
    "red_bg": "#f8e3e3",
    "blue": "#2563eb",
    "blue_bg": "#e3ecff",
    "purple": "#7c5ba7",
    "teal": "#0f8b78",
    "recession": "#cbd5e1",
}


# =============================================================================
# Modèles / utilitaires
# =============================================================================

@dataclass(frozen=True)
class SeriesPoint:
    day: date
    value: float


@dataclass
class ChartRef:
    fig: Any
    ax: Any
    canvas: Any


def ensure_cache() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def cache_file(key: str) -> Path:
    return CACHE_DIR / f"{key}.json"


def atomic_json_write(path: Path, payload: Any) -> None:
    try:
        ensure_cache()
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
    except Exception:
        pass


def json_read(path: Path) -> Any:
    try:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_series(key: str, points: list[SeriesPoint]) -> None:
    atomic_json_write(
        cache_file(key),
        {
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "points": [[p.day.isoformat(), p.value] for p in points],
        },
    )


def load_series(key: str) -> list[SeriesPoint]:
    payload = json_read(cache_file(key))
    if not isinstance(payload, dict):
        return []
    output: list[SeriesPoint] = []
    for row in payload.get("points", []):
        try:
            output.append(SeriesPoint(date.fromisoformat(row[0]), float(row[1])))
        except Exception:
            continue
    return sorted(output, key=lambda p: p.day)


def cache_age_seconds(key: str) -> float | None:
    payload = json_read(cache_file(key))
    if not isinstance(payload, dict):
        return None
    raw = payload.get("updated_at")
    if not isinstance(raw, str):
        return None
    try:
        stamp = datetime.fromisoformat(raw)
        return max(0.0, (datetime.now() - stamp).total_seconds())
    except Exception:
        return None


def cache_is_fresh(key: str) -> bool:
    age = cache_age_seconds(key)
    ttl = int(SERIES[key]["ttl"])
    return age is not None and age <= ttl


def fmt_pct(value: float | None, digits: int = 2, signed: bool = False) -> str:
    if value is None or not math.isfinite(value):
        return UNAVAILABLE_MESSAGE
    prefix = "+" if signed and value > 0 else ""
    return f"{prefix}{value:.{digits}f}%"


def latest(points: list[SeriesPoint]) -> float | None:
    return points[-1].value if points else None


def select_start(period: str) -> date:
    days = PERIODS.get(period)
    return date(1990, 1, 1) if days is None else date.today() - timedelta(days=days)


def visible_points(points: list[SeriesPoint], start: date, end: date) -> list[SeriesPoint]:
    return [p for p in points if start <= p.day <= end]


def downsample(points: list[SeriesPoint], max_points: int = 1600) -> list[SeriesPoint]:
    if len(points) <= max_points:
        return points
    step = max(1, math.ceil(len(points) / max_points))
    output = points[::step]
    if output[-1] != points[-1]:
        output.append(points[-1])
    return output


def value_on_or_before(points: list[SeriesPoint], target: date) -> float | None:
    for p in reversed(points):
        if p.day <= target:
            return p.value
    return None


def change_12m(points: list[SeriesPoint]) -> tuple[float | None, float | None, float | None]:
    if not points:
        return None, None, None
    current = points[-1].value
    previous = value_on_or_before(points, points[-1].day - timedelta(days=365))
    if previous is None:
        return current, None, None
    return current, previous, current - previous


def monthly_yoy(levels: list[SeriesPoint]) -> list[SeriesPoint]:
    by_month = {(p.day.year, p.day.month): p for p in levels}
    output: list[SeriesPoint] = []
    for p in levels:
        previous = by_month.get((p.day.year - 1, p.day.month))
        if previous is None or previous.value == 0 or p.day.year < 1990:
            continue
        output.append(SeriesPoint(p.day, (p.value / previous.value - 1.0) * 100.0))
    return output


def build_policy_spread(fed: list[SeriesPoint], cpi: list[SeriesPoint]) -> list[SeriesPoint]:
    if not fed or not cpi:
        return []
    fed = sorted(fed, key=lambda p: p.day)
    cpi = sorted(cpi, key=lambda p: p.day)
    output: list[SeriesPoint] = []
    i = 0
    current_fed: float | None = None
    for cp in cpi:
        while i < len(fed) and fed[i].day <= cp.day:
            current_fed = fed[i].value
            i += 1
        if current_fed is not None:
            output.append(SeriesPoint(cp.day, current_fed - cp.value))
    return output


def recession_bands() -> list[tuple[date, date]]:
    return [
        (date(1990, 7, 1), date(1991, 4, 1)),
        (date(2001, 3, 1), date(2001, 12, 1)),
        (date(2007, 12, 1), date(2009, 7, 1)),
        (date(2020, 2, 1), date(2020, 5, 1)),
    ]


# =============================================================================
# Réseau FRED
# =============================================================================


def _urllib_text(url: str, timeout: float, context: ssl.SSLContext | None = None) -> str:
    request = Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/csv,text/plain,application/json,*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "close",
        },
    )
    with urlopen(request, timeout=timeout, context=context) as response:
        raw = response.read()
    return raw.decode("utf-8", errors="replace")


def _curl_text(url: str, timeout: float) -> str:
    """Repli très utile sur macOS quand le Python local a un problème de certificats."""
    curl = "curl"
    result = subprocess.run(
        [
            curl, "-L", "--fail", "--silent", "--show-error",
            "--connect-timeout", str(max(2, int(timeout // 2))),
            "--max-time", str(max(3, int(timeout))),
            "-A", USER_AGENT, url,
        ],
        capture_output=True,
        text=True,
        timeout=timeout + 2.0,
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError((result.stderr or "curl sans réponse").strip())
    return result.stdout


def http_text(url: str, timeout: float = NETWORK_TIMEOUT) -> str:
    """Téléchargement robuste sans dépendance obligatoire.

    Ordre : urllib normal -> urllib avec certifi si présent -> curl système.
    Cette combinaison règle la plupart des erreurs SSL observées avec Python sur macOS.
    """
    errors: list[str] = []

    try:
        return _urllib_text(url, timeout)
    except Exception as exc:
        errors.append(f"urllib: {exc}")

    try:
        import certifi  # type: ignore
        ctx = ssl.create_default_context(cafile=certifi.where())
        return _urllib_text(url, timeout, context=ctx)
    except Exception as exc:
        errors.append(f"certifi: {exc}")

    try:
        return _curl_text(url, timeout)
    except Exception as exc:
        errors.append(f"curl: {exc}")

    raise RuntimeError("Téléchargement impossible · " + " | ".join(errors[-3:]))


def _parse_fred_csv(text: str, series_id: str) -> list[SeriesPoint]:
    reader = csv.DictReader(io.StringIO(text))
    output: list[SeriesPoint] = []
    for row in reader:
        raw_day = row.get("observation_date") or row.get("DATE") or row.get("date")
        raw_value = row.get(series_id) or row.get(series_id.upper()) or row.get("VALUE") or row.get("value")
        if not raw_day or raw_value in (None, "", ".", "NA"):
            continue
        try:
            day = date.fromisoformat(str(raw_day).strip()[:10])
            value = float(str(raw_value).strip())
        except Exception:
            continue
        if day >= date(1989, 1, 1) and math.isfinite(value):
            output.append(SeriesPoint(day, value))
    return sorted(output, key=lambda p: p.day)


def _parse_fred_txt(text: str) -> list[SeriesPoint]:
    output: list[SeriesPoint] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "DATE", "Title", "Series", "Source", "Release", "Frequency", "Units")):
            continue
        parts = line.replace(",", " ").split()
        if len(parts) < 2 or len(parts[0]) < 10:
            continue
        try:
            day = date.fromisoformat(parts[0][:10])
            value = float(parts[1])
        except Exception:
            continue
        if day >= date(1989, 1, 1) and math.isfinite(value):
            output.append(SeriesPoint(day, value))
    return sorted(output, key=lambda p: p.day)


def _parse_dbnomics(text: str) -> list[SeriesPoint]:
    payload = json.loads(text)
    docs = (((payload or {}).get("series") or {}).get("docs") or [])
    if not docs:
        docs = ((((payload or {}).get("dataset") or {}).get("series") or {}).get("docs") or [])
    if not docs:
        return []
    doc = docs[0]
    periods = doc.get("period") or doc.get("periods") or []
    values = doc.get("value") or doc.get("values") or []
    output: list[SeriesPoint] = []
    for raw_day, raw_value in zip(periods, values):
        if raw_value is None:
            continue
        text_day = str(raw_day)
        try:
            if len(text_day) == 4:
                day = date(int(text_day), 1, 1)
            elif len(text_day) == 7:
                day = date.fromisoformat(text_day + "-01")
            else:
                day = date.fromisoformat(text_day[:10])
            value = float(raw_value)
        except Exception:
            continue
        if day >= date(1989, 1, 1) and math.isfinite(value):
            output.append(SeriesPoint(day, value))
    return sorted(output, key=lambda p: p.day)


def fetch_fred_raw(series_id: str) -> list[SeriesPoint]:
    """Essaie plusieurs routes publiques avant d'abandonner."""
    errors: list[str] = []

    csv_url = FRED_URL.format(series_id=series_id)
    try:
        points = _parse_fred_csv(http_text(csv_url), series_id)
        if points:
            return points
        errors.append("CSV vide")
    except Exception as exc:
        errors.append(f"CSV: {exc}")

    txt_url = f"https://fred.stlouisfed.org/data/{series_id}.txt"
    try:
        points = _parse_fred_txt(http_text(txt_url))
        if points:
            return points
        errors.append("TXT vide")
    except Exception as exc:
        errors.append(f"TXT: {exc}")

    mirror_url = f"https://api.db.nomics.world/v22/series/FRED/{series_id}?observations=1"
    try:
        points = _parse_dbnomics(http_text(mirror_url))
        if points:
            return points
        errors.append("DBnomics vide")
    except Exception as exc:
        errors.append(f"DBnomics: {exc}")

    raise RuntimeError(f"{series_id}: aucune source disponible · " + " | ".join(errors[-3:]))


def fetch_series(key: str) -> list[SeriesPoint]:
    cfg = SERIES[key]
    raw = fetch_fred_raw(str(cfg["id"]))
    if cfg["transform"] == "yoy":
        points = monthly_yoy(raw)
    else:
        points = [p for p in raw if p.day >= date(1990, 1, 1)]
    if not points:
        raise RuntimeError(f"{key} : aucune donnée exploitable")
    save_series(key, points)
    return points


# =============================================================================
# Application
# =============================================================================

class MacroDashboard(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Macro Intelligence — Taux et inflation")
        self.configure(bg=COLORS["bg"])

        screen_w = max(self.winfo_screenwidth(), 1280)
        screen_h = max(self.winfo_screenheight(), 820)
        width = min(1920, max(1280, int(screen_w * 0.96)))
        height = min(1120, max(820, int(screen_h * 0.93)))
        x = max(0, (screen_w - width) // 2)
        y = max(0, (screen_h - height) // 2)
        self.geometry(f"{width}x{height}+{x}+{y}")
        self.minsize(1220, 780)

        self.period = tk.StringVar(value=DEFAULT_PERIOD)
        self.status = tk.StringVar(value="Ouverture…")
        self.range_text = tk.StringVar(value="—")
        self.session_text = tk.StringVar(value="")

        self.data: dict[str, list[SeriesPoint]] = {key: [] for key in SERIES}
        self.source_state: dict[str, str] = {key: "—" for key in SERIES}
        self.source_error: dict[str, str] = {}
        self.updated_at: datetime | None = None

        self.queue: queue.Queue[tuple[int, str, str, Any]] = queue.Queue()
        self.executor = ThreadPoolExecutor(max_workers=len(SERIES), thread_name_prefix="macro")
        self.generation = 0
        self.pending: set[str] = set()
        self.render_job: str | None = None
        self.dirty: set[str] = set()

        self.charts: dict[str, ChartRef] = {}
        self.chart_holders: dict[str, tk.Frame] = {}
        self.kpi_value_labels: dict[str, tk.Label] = {}

        self._styles()
        self._build_shell()

        self.after(10, self._boot)
        self.after(80, self._poll_queue)
        self.protocol("WM_DELETE_WINDOW", self._close)

    # ------------------------------------------------------------------ UI --

    def _styles(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass

    def _build_shell(self) -> None:
        self.page = tk.Frame(self, bg=COLORS["bg"])
        self.page.pack(fill="both", expand=True)

        self._header()
        self._period_control()
        self._kpi_strip()

        self.footer = tk.Label(
            self.page,
            text="",
            bg=COLORS["bg"],
            fg=COLORS["muted_light"],
            font=(FONT, 8),
            anchor="w",
        )
        self.footer.pack(side="bottom", fill="x", padx=20, pady=(3, 7))

        wrap = tk.Frame(self.page, bg=COLORS["bg"])
        wrap.pack(fill="both", expand=True, padx=(18, 10), pady=(0, 2))

        self.body_canvas = tk.Canvas(
            wrap,
            bg=COLORS["bg"],
            highlightthickness=0,
            borderwidth=0,
            yscrollincrement=1,
        )
        scrollbar = tk.Scrollbar(wrap, orient="vertical", command=self.body_canvas.yview, width=9)
        self.body_canvas.configure(yscrollcommand=scrollbar.set)
        self.body_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y", padx=(5, 0))

        self.body = tk.Frame(self.body_canvas, bg=COLORS["bg"])
        self.body_window = self.body_canvas.create_window((0, 0), window=self.body, anchor="nw")
        self.body.grid_columnconfigure(0, weight=1, uniform="main")
        self.body.grid_columnconfigure(1, weight=1, uniform="main")

        self.body.bind("<Configure>", self._sync_scroll_region)
        self.body_canvas.bind("<Configure>", self._fit_body_width)
        self.bind_all("<MouseWheel>", self._on_mousewheel)
        self.bind_all("<Button-4>", self._on_mousewheel)
        self.bind_all("<Button-5>", self._on_mousewheel)

        self._build_panels()

    def _header(self) -> None:
        head = tk.Frame(self.page, bg=COLORS["bg"])
        head.pack(fill="x", padx=20, pady=(16, 10))

        left = tk.Frame(head, bg=COLORS["bg"])
        left.pack(side="left", fill="x", expand=True)
        tk.Label(
            left,
            text="MACRO INTELLIGENCE",
            bg=COLORS["bg"],
            fg=COLORS["blue"],
            font=(FONT, 10, "bold"),
        ).pack(anchor="w")
        tk.Label(
            left,
            text="Taux et inflation",
            bg=COLORS["bg"],
            fg=COLORS["ink"],
            font=(FONT, 27, "bold"),
        ).pack(anchor="w")
        tk.Label(
            left,
            text="Fed Funds · CPI · core CPI · PCE · courbe des taux · récession · taux réel",
            bg=COLORS["bg"],
            fg=COLORS["muted"],
            font=(FONT, 11),
        ).pack(anchor="w")

        right = tk.Frame(head, bg=COLORS["bg"])
        right.pack(side="right", anchor="ne", pady=(8, 0))
        self.status_label = tk.Label(
            right,
            textvariable=self.status,
            bg=COLORS["bg"],
            fg=COLORS["muted"],
            font=(FONT, 10, "bold"),
        )
        self.status_label.pack(side="left", padx=(0, 12), pady=(5, 0))
        self.refresh_button = tk.Button(
            right,
            text="Actualiser les données",
            command=lambda: self.refresh_live(force=True),
            bg=COLORS["ink"],
            fg="white",
            activebackground=COLORS["ink_soft"],
            activeforeground="white",
            borderwidth=0,
            padx=18,
            pady=10,
            cursor="hand2",
            font=(FONT, 10, "bold"),
        )
        self.refresh_button.pack(side="left")

    def _period_control(self) -> None:
        outer = tk.Frame(
            self.page,
            bg=COLORS["panel"],
            highlightbackground=COLORS["line_soft"],
            highlightthickness=1,
            height=92,
        )
        outer.pack(fill="x", padx=20, pady=(0, 12))
        outer.pack_propagate(False)

        left = tk.Frame(outer, bg=COLORS["panel"], width=220)
        left.pack(side="left", fill="y", padx=(18, 8), pady=14)
        left.pack_propagate(False)
        tk.Label(
            left,
            text="HORIZON D’ANALYSE",
            bg=COLORS["panel"],
            fg=COLORS["ink"],
            font=(FONT, 11, "bold"),
        ).pack(anchor="w")
        tk.Label(
            left,
            text="Le changement est instantané : aucun réseau",
            bg=COLORS["panel"],
            fg=COLORS["muted"],
            font=(FONT, 9),
        ).pack(anchor="w", pady=(4, 0))

        segment = tk.Frame(
            outer,
            bg=COLORS["panel_soft"],
            highlightbackground=COLORS["line_soft"],
            highlightthickness=1,
        )
        segment.pack(side="left", fill="both", expand=True, padx=8, pady=15)

        self.period_buttons: dict[str, tk.Button] = {}
        for label in PERIODS:
            button = tk.Button(
                segment,
                text=label,
                command=lambda v=label: self._set_period(v),
                borderwidth=0,
                padx=14,
                pady=10,
                cursor="hand2",
                font=(FONT, 10, "bold"),
            )
            button.pack(side="left", fill="both", expand=True, padx=2, pady=2)
            self.period_buttons[label] = button
        self._style_period_buttons()

        info = tk.Frame(outer, bg=COLORS["panel"], width=245)
        info.pack(side="right", fill="y", padx=(10, 18), pady=14)
        info.pack_propagate(False)
        tk.Label(
            info,
            text="PLAGE AFFICHÉE",
            bg=COLORS["panel"],
            fg=COLORS["muted_light"],
            font=(FONT, 10, "bold"),
        ).pack(anchor="e")
        tk.Label(
            info,
            textvariable=self.range_text,
            bg=COLORS["panel"],
            fg=COLORS["ink"],
            font=(FONT, 10, "bold"),
        ).pack(anchor="e", pady=(3, 0))
        tk.Label(
            info,
            textvariable=self.session_text,
            bg=COLORS["panel"],
            fg=COLORS["muted"],
            font=(FONT, 8),
        ).pack(anchor="e", pady=(2, 0))

    def _kpi_strip(self) -> None:
        self.kpi_grid = tk.Frame(self.page, bg=COLORS["bg"])
        self.kpi_grid.pack(fill="x", padx=20, pady=(0, 11))
        for col in range(7):
            self.kpi_grid.grid_columnconfigure(col, weight=1, uniform="kpi")

        specs = [
            ("fed", "FED FUNDS", COLORS["blue"]),
            ("cpi", "CPI", COLORS["gold"]),
            ("core", "CORE CPI", COLORS["purple"]),
            ("pce", "PCE", COLORS["red"]),
            ("curve", "10Y – 2Y", COLORS["teal"]),
            ("recession", "RÉCESSION", COLORS["green"]),
            ("real", "TAUX RÉEL", COLORS["blue"]),
        ]

        for idx, (key, label, accent) in enumerate(specs):
            card = tk.Frame(
                self.kpi_grid,
                bg=COLORS["panel"],
                highlightbackground=COLORS["line_soft"],
                highlightthickness=1,
            )
            card.grid(row=0, column=idx, sticky="nsew", padx=3)
            tk.Frame(card, bg=accent, width=3).pack(side="left", fill="y")
            inner = tk.Frame(card, bg=COLORS["panel"])
            inner.pack(fill="both", expand=True, padx=9, pady=7)
            tk.Label(
                inner,
                text=label,
                bg=COLORS["panel"],
                fg=COLORS["muted"],
                font=(FONT, 9, "bold"),
            ).pack(anchor="w")
            value = tk.Label(
                inner,
                text="—",
                bg=COLORS["panel"],
                fg=COLORS["ink"],
                font=(FONT, 15, "bold"),
                wraplength=170,
                justify="left",
            )
            value.pack(anchor="w", pady=(2, 0))
            self.kpi_value_labels[key] = value

    def _build_panels(self) -> None:
        self.body.grid_rowconfigure(0, minsize=455)
        self.body.grid_rowconfigure(1, minsize=455)
        self.body.grid_rowconfigure(2, minsize=455)
        self.body.grid_rowconfigure(3, minsize=585)

        self.summary_panel = self._panel("Lecture macro", "Synthèse des signaux", 0, 0, 1, 443)
        self._build_summary_content()

        self.chart_holders["inflation"] = self._plot_panel(
            "Inflation US", "CPI headline, core CPI et PCE", 0, 1, 1, 443
        )
        self.chart_holders["fed_rate"] = self._plot_panel(
            "Fed Funds", "Taux effectif quotidien", 1, 0, 1, 443
        )
        self.chart_holders["yield_curve"] = self._plot_panel(
            "Courbe des taux 10Y – 2Y", "Inversion, désinversion et récessions", 1, 1, 1, 443
        )
        self.chart_holders["recession"] = self._plot_panel(
            "Probabilité de récession", "Modèle historique Chauvet–Piger", 2, 0, 1, 443
        )
        self.chart_holders["policy"] = self._plot_panel(
            "Taux réel ex-post", "Fed Funds – inflation CPI", 2, 1, 1, 443
        )
        self.chart_holders["momentum"] = self._plot_panel(
            "Dynamique macro — 12 mois",
            "À gauche = baisse · à droite = hausse · variation en points de pourcentage",
            3,
            0,
            2,
            573,
        )

    def _panel(self, title: str, subtitle: str, row: int, col: int, colspan: int, height: int) -> tk.Frame:
        panel = tk.Frame(
            self.body,
            bg=COLORS["panel"],
            highlightbackground=COLORS["line_soft"],
            highlightthickness=1,
            height=height,
        )
        panel.grid(row=row, column=col, columnspan=colspan, sticky="nsew", padx=6, pady=6)
        panel.grid_propagate(False)
        panel.pack_propagate(False)

        head = tk.Frame(panel, bg=COLORS["panel"], height=46)
        head.pack(fill="x", padx=14, pady=(9, 4))
        head.pack_propagate(False)
        tk.Label(
            head,
            text=title,
            bg=COLORS["panel"],
            fg=COLORS["ink"],
            font=(FONT, 13, "bold"),
        ).pack(side="left")
        tk.Label(
            head,
            text=subtitle,
            bg=COLORS["panel"],
            fg=COLORS["muted"],
            font=(FONT, 9),
        ).pack(side="right")
        return panel

    def _plot_panel(self, title: str, subtitle: str, row: int, col: int, colspan: int, height: int) -> tk.Frame:
        panel = self._panel(title, subtitle, row, col, colspan, height)
        holder = tk.Frame(panel, bg=COLORS["panel_alt"])
        holder.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        return holder

    def _build_summary_content(self) -> None:
        self.summary_hero = tk.Frame(
            self.summary_panel,
            bg=COLORS["blue_bg"],
            highlightbackground=COLORS["line_soft"],
            highlightthickness=1,
        )
        self.summary_hero.pack(fill="x", padx=12, pady=(4, 8))
        self.summary_accent = tk.Frame(self.summary_hero, bg=COLORS["blue"], width=5)
        self.summary_accent.pack(side="left", fill="y")

        inner = tk.Frame(self.summary_hero, bg=COLORS["blue_bg"])
        inner.pack(fill="both", expand=True, padx=13, pady=10)
        self.regime_badge = tk.Label(
            inner,
            text="RÉGIME ACTUEL",
            bg=COLORS["blue"],
            fg="white",
            font=(FONT, 8, "bold"),
            padx=9,
            pady=4,
        )
        self.regime_badge.pack(anchor="w")
        self.regime_label = tk.Label(
            inner,
            text="Chargement des signaux",
            bg=COLORS["blue_bg"],
            fg=COLORS["ink"],
            font=(FONT, 20, "bold"),
        )
        self.regime_label.pack(anchor="w", pady=(6, 1))
        self.note_label = tk.Label(
            inner,
            text="Le cache local est affiché avant toute requête réseau.",
            bg=COLORS["blue_bg"],
            fg=COLORS["muted"],
            font=(FONT, 9),
            wraplength=520,
            justify="left",
        )
        self.note_label.pack(anchor="w")

        signals = tk.Frame(self.summary_panel, bg=COLORS["panel"])
        signals.pack(fill="x", padx=10, pady=(0, 8))
        for col in range(3):
            signals.grid_columnconfigure(col, weight=1, uniform="signals")

        self.signal_values: dict[str, tk.Label] = {}
        specs = [
            ("policy", "TAUX RÉEL APPROX.", "Fed Funds – CPI"),
            ("curve", "COURBE 10Y–2Y", "Négatif = inversion"),
            ("inflation", "CPI SUR 12 MOIS", "Variation de l'inflation"),
        ]
        for idx, (key, label, detail) in enumerate(specs):
            card = tk.Frame(
                signals,
                bg=COLORS["panel_alt"],
                highlightbackground=COLORS["line_soft"],
                highlightthickness=1,
            )
            card.grid(row=0, column=idx, sticky="nsew", padx=3)
            tk.Label(
                card,
                text=label,
                bg=COLORS["panel_alt"],
                fg=COLORS["muted_light"],
                font=(FONT, 9, "bold"),
            ).pack(anchor="w", padx=8, pady=(7, 1))
            value = tk.Label(
                card,
                text="—",
                bg=COLORS["panel_alt"],
                fg=COLORS["ink"],
                font=(FONT, 12, "bold"),
            )
            value.pack(anchor="w", padx=8)
            tk.Label(
                card,
                text=detail,
                bg=COLORS["panel_alt"],
                fg=COLORS["muted"],
                font=(FONT, 8),
            ).pack(anchor="w", padx=8, pady=(1, 7))
            self.signal_values[key] = value

        box = tk.Frame(
            self.summary_panel,
            bg=COLORS["panel_alt"],
            highlightbackground=COLORS["line_soft"],
            highlightthickness=1,
        )
        box.pack(fill="both", expand=True, padx=12, pady=(0, 10))
        tk.Label(
            box,
            text="LECTURE DES 12 DERNIERS MOIS",
            bg=COLORS["panel_alt"],
            fg=COLORS["muted"],
            font=(FONT, 9, "bold"),
        ).pack(anchor="w", padx=11, pady=(9, 4))
        self.momentum_title = tk.Label(
            box,
            text="En attente des données",
            bg=COLORS["panel_alt"],
            fg=COLORS["ink"],
            font=(FONT, 14, "bold"),
        )
        self.momentum_title.pack(anchor="w", padx=11)
        self.momentum_detail = tk.Label(
            box,
            text="Les séries sont chargées indépendamment pour éviter qu'une source bloque le tableau.",
            bg=COLORS["panel_alt"],
            fg=COLORS["muted"],
            font=(FONT, 9),
            wraplength=520,
            justify="left",
        )
        self.momentum_detail.pack(anchor="w", padx=11, pady=(3, 8))

    # -------------------------------------------------------------- lifecycle --

    def _boot(self) -> None:
        self.status.set("Initialisation des graphiques")
        try:
            load_matplotlib()
            self._create_charts()
        except Exception as exc:
            self.status.set("Matplotlib indisponible")
            self._show_matplotlib_error(str(exc))
            return

        self._load_cache_into_memory()
        self._render_all()

        stale = [key for key in SERIES if not cache_is_fresh(key)]
        if stale:
            self.refresh_live(keys=stale, force=False)
        else:
            self.status.set("Prêt · cache à jour")

    def _show_matplotlib_error(self, detail: str) -> None:
        for child in self.body.winfo_children():
            child.destroy()
        msg = (
            f"{UNAVAILABLE_MESSAGE}\n\n"
            "Matplotlib n'est pas disponible.\n\n"
            "Installez-le avec :\npython3 -m pip install matplotlib\n\n"
            f"Détail : {detail}"
        )
        tk.Label(
            self.body,
            text=msg,
            bg=COLORS["panel"],
            fg=COLORS["ink"],
            font=(FONT, 13),
            justify="left",
            padx=30,
            pady=30,
        ).grid(row=0, column=0, columnspan=2, sticky="nsew", padx=10, pady=10)

    def _load_cache_into_memory(self) -> None:
        for key in SERIES:
            cached = load_series(key)
            if cached:
                self.data[key] = cached
                self.source_state[key] = "cache"

    def refresh_live(self, keys: list[str] | None = None, force: bool = True) -> None:
        if self.pending:
            return
        keys = list(SERIES) if keys is None else list(dict.fromkeys(keys))
        keys = [key for key in keys if key in SERIES]
        if not keys:
            return

        self.generation += 1
        generation = self.generation
        self.pending = set(keys)
        self.refresh_button.configure(state="disabled")
        self.status.set(f"Actualisation · {len(keys)} série(s)")

        for key in keys:
            def task(k: str = key) -> None:
                try:
                    points = fetch_series(k)
                    self.queue.put((generation, k, "ok", points))
                except Exception as exc:
                    self.queue.put((generation, k, "error", str(exc)))
            self.executor.submit(task)

    def _poll_queue(self) -> None:
        changed: set[str] = set()
        any_state_change = False

        while True:
            try:
                generation, key, kind, payload = self.queue.get_nowait()
            except queue.Empty:
                break

            if generation != self.generation:
                continue

            self.pending.discard(key)
            any_state_change = True

            if kind == "ok":
                self.data[key] = payload
                self.source_state[key] = "live"
                self.source_error.pop(key, None)
                self.updated_at = datetime.now()
                changed.add(key)
            else:
                self.source_error[key] = str(payload)
                self.source_state[key] = "cache" if self.data.get(key) else "indisponible"

        if changed:
            self._schedule_render(changed)
        elif any_state_change:
            self._render_footer()

        if self.pending:
            self.status.set(f"Actualisation · {len(self.pending)} série(s)")
        elif str(self.refresh_button.cget("state")) == "disabled":
            self.refresh_button.configure(state="normal")
            unavailable = [k for k in SERIES if not self.data.get(k)]
            if unavailable:
                self.status.set(f"Prêt · {len(unavailable)} série(s) indisponible(s)")
            else:
                self.status.set("À jour")
            self._render_footer()

        self.after(80, self._poll_queue)

    def _close(self) -> None:
        try:
            self.executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        self.destroy()

    # --------------------------------------------------------------- scroll --

    def _sync_scroll_region(self, _event: tk.Event | None = None) -> None:
        bbox = self.body_canvas.bbox("all")
        if bbox:
            self.body_canvas.configure(scrollregion=bbox)

    def _fit_body_width(self, event: tk.Event) -> None:
        self.body_canvas.itemconfigure(self.body_window, width=max(1, int(event.width)))

    def _on_mousewheel(self, event: tk.Event) -> str | None:
        bbox = self.body_canvas.bbox("all")
        if not bbox:
            return None

        total_h = float(bbox[3] - bbox[1])
        view_h = float(self.body_canvas.winfo_height())
        max_top = max(0.0, total_h - view_h)
        if max_top <= 0 or total_h <= 0:
            return None

        if getattr(event, "num", None) == 4:
            pixels = -55.0
        elif getattr(event, "num", None) == 5:
            pixels = 55.0
        else:
            delta = float(getattr(event, "delta", 0) or 0)
            if delta == 0:
                return None
            if sys.platform == "darwin":
                # Sur trackpad macOS, delta est petit et fréquent : mouvement direct,
                # aucun after(), aucune inertie artificielle, aucun conflit avec Tk.
                pixels = -delta * 2.6
            else:
                pixels = -(delta / 120.0) * 60.0

        current_top = float(self.body_canvas.yview()[0]) * total_h
        new_top = min(max(current_top + pixels, 0.0), max_top)
        self.body_canvas.yview_moveto(new_top / total_h)
        return "break"

    # --------------------------------------------------------------- charts --

    def _create_charts(self) -> None:
        specs = {
            "inflation": (6.5, 3.65),
            "fed_rate": (6.5, 3.65),
            "yield_curve": (6.5, 3.65),
            "recession": (6.5, 3.65),
            "policy": (6.5, 3.65),
            "momentum": (13.5, 5.05),
        }
        for key, size in specs.items():
            fig = Figure(figsize=size, dpi=86, facecolor=COLORS["panel_alt"])
            ax = fig.add_subplot(111, facecolor=COLORS["panel_alt"])
            if key == "momentum":
                fig.subplots_adjust(left=0.13, right=0.975, top=0.93, bottom=0.16)
            else:
                fig.subplots_adjust(left=0.10, right=0.985, top=0.93, bottom=0.17)
            canvas = FigureCanvasTkAgg(fig, master=self.chart_holders[key])
            canvas.get_tk_widget().pack(fill="both", expand=True)
            self.charts[key] = ChartRef(fig, ax, canvas)
            self._empty_axis(ax, "Chargement…")
            canvas.draw_idle()

    def _empty_axis(self, ax: Any, text: str) -> None:
        ax.clear()
        ax.set_facecolor(COLORS["panel_alt"])
        ax.text(
            0.5,
            0.52,
            text,
            transform=ax.transAxes,
            ha="center",
            va="center",
            color=COLORS["muted"],
            fontsize=10,
        )
        ax.set_axis_off()

    def _base_axis(self, ax: Any, ylabel: str = "", recession_shading: bool = True) -> None:
        ax.set_facecolor(COLORS["panel_alt"])
        ax.set_axis_on()
        ax.tick_params(colors=COLORS["muted"], labelsize=9)
        ax.grid(True, color=COLORS["grid"], linewidth=0.75, alpha=0.9)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color(COLORS["line"])
        ax.spines["bottom"].set_color(COLORS["line"])
        if ylabel:
            ax.set_ylabel(ylabel, color=COLORS["muted"], fontsize=9.5, fontweight="bold")

        locator = mdates.AutoDateLocator(minticks=4, maxticks=7)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

        if recession_shading:
            for lo, hi in recession_bands():
                ax.axvspan(lo, hi, color=COLORS["recession"], alpha=0.28, linewidth=0, zorder=0)

    def _set_period(self, value: str) -> None:
        if value == self.period.get():
            return
        self.period.set(value)
        self._style_period_buttons()
        self._schedule_render({"period"})

    def _style_period_buttons(self) -> None:
        current = self.period.get()
        for label, button in self.period_buttons.items():
            selected = label == current
            button.configure(
                bg=COLORS["ink"] if selected else COLORS["panel_soft"],
                fg="white" if selected else COLORS["ink_soft"],
                activebackground=COLORS["ink_soft"] if selected else COLORS["line_soft"],
                activeforeground="white" if selected else COLORS["ink"],
            )

    def _schedule_render(self, sources: set[str]) -> None:
        self.dirty.update(sources)
        if self.render_job is not None:
            try:
                self.after_cancel(self.render_job)
            except Exception:
                pass
        self.render_job = self.after(55, self._render_dirty)

    def _render_dirty(self) -> None:
        self.render_job = None
        dirty = set(self.dirty)
        self.dirty.clear()

        if "period" in dirty:
            self._render_all()
            return

        start, end = self._current_window()
        self._update_range_labels(start, end)
        self._render_kpis_and_summary()

        if dirty & {"cpi", "core_cpi", "pce"}:
            self._render_inflation(start, end)
        if "fed_funds" in dirty:
            self._render_fed_rate(start, end)
        if "yield_curve" in dirty:
            self._render_yield_curve(start, end)
        if "recession_probability" in dirty:
            self._render_recession(start, end)
        if dirty & {"fed_funds", "cpi"}:
            self._render_policy(start, end)
        if dirty & {"fed_funds", "cpi", "core_cpi", "yield_curve"}:
            self._render_momentum()
        self._render_footer()

    def _current_window(self) -> tuple[date, date]:
        start = select_start(self.period.get())
        end = date.today()
        for points in self.data.values():
            if points:
                end = max(end, points[-1].day)
        return start, end

    def _update_range_labels(self, start: date, end: date) -> None:
        self.range_text.set(f"{start.strftime('%d.%m.%Y')}  →  {end.strftime('%d.%m.%Y')}")
        count = len(visible_points(self.data["yield_curve"], start, end))
        self.session_text.set(f"{count:,} observations 10Y–2Y".replace(",", " "))

    def _render_all(self) -> None:
        start, end = self._current_window()
        self._update_range_labels(start, end)
        self._render_kpis_and_summary()
        self._render_inflation(start, end)
        self._render_fed_rate(start, end)
        self._render_yield_curve(start, end)
        self._render_recession(start, end)
        self._render_policy(start, end)
        self._render_momentum()
        self._render_footer()

    def _summary(self) -> dict[str, Any]:
        fed = latest(self.data["fed_funds"])
        cpi = latest(self.data["cpi"])
        core = latest(self.data["core_cpi"])
        pce = latest(self.data["pce"])
        curve = latest(self.data["yield_curve"])
        recession = latest(self.data["recession_probability"])
        policy = fed - cpi if fed is not None and cpi is not None else None
        _, _, cpi_change = change_12m(self.data["cpi"])
        _, _, fed_change = change_12m(self.data["fed_funds"])
        _, _, curve_change = change_12m(self.data["yield_curve"])

        if all(value is None for value in (fed, cpi, core, pce, curve, recession)):
            regime, tone = UNAVAILABLE_MESSAGE, "blue"
            note = "Les données nécessaires à cette synthèse ne sont pas encore disponibles."
        elif recession is not None and recession >= 35:
            regime, tone = "Risque macro élevé", "red"
            note = "Le modèle de récession est élevé ; la croissance devient le signal prioritaire."
        elif cpi is not None and fed is not None and cpi >= 3.0 and fed >= 4.0:
            regime, tone = "Politique restrictive", "gold"
            note = "Les Fed Funds restent élevés alors que l'inflation reste au-dessus de la cible de 2 %."
        elif cpi is not None and cpi_change is not None and cpi < 3.0 and cpi_change < 0:
            regime, tone = "Désinflation", "green"
            note = "L'inflation est sous 3 % et recule sur un an ; la pression monétaire se détend."
        elif curve is not None and curve < 0:
            regime, tone = "Courbe inversée", "gold"
            note = "Le spread 10Y–2Y reste négatif, signal classique d'une politique monétaire restrictive."
        else:
            regime, tone = "Transition", "blue"
            note = "Les indicateurs ne donnent pas encore un régime macro unique ; la tendance à 12 mois est prioritaire."

        return {
            "fed": fed,
            "cpi": cpi,
            "core": core,
            "pce": pce,
            "curve": curve,
            "recession": recession,
            "policy": policy,
            "cpi_change": cpi_change,
            "fed_change": fed_change,
            "curve_change": curve_change,
            "regime": regime,
            "tone": tone,
            "note": note,
        }

    def _render_kpis_and_summary(self) -> None:
        s = self._summary()
        self.kpi_value_labels["fed"].configure(text=fmt_pct(s["fed"], 2))
        self.kpi_value_labels["cpi"].configure(text=fmt_pct(s["cpi"], 2))
        self.kpi_value_labels["core"].configure(text=fmt_pct(s["core"], 2))
        self.kpi_value_labels["pce"].configure(text=fmt_pct(s["pce"], 2))
        self.kpi_value_labels["curve"].configure(text=fmt_pct(s["curve"], 2, True))
        self.kpi_value_labels["recession"].configure(text=fmt_pct(s["recession"], 1))
        self.kpi_value_labels["real"].configure(text=fmt_pct(s["policy"], 2, True))

        tone = COLORS.get(s["tone"], COLORS["blue"])
        tone_bg = COLORS.get(f"{s['tone']}_bg", COLORS["blue_bg"])
        self.summary_hero.configure(bg=tone_bg)
        self.summary_accent.configure(bg=tone)
        self.regime_badge.configure(bg=tone)
        self.regime_label.configure(text=s["regime"], bg=tone_bg)
        self.note_label.configure(text=s["note"], bg=tone_bg)

        self.signal_values["policy"].configure(
            text=fmt_pct(s["policy"], 2, True),
            fg=COLORS["green"] if (s["policy"] or 0) > 0 else COLORS["gold"],
        )
        self.signal_values["curve"].configure(
            text=fmt_pct(s["curve"], 2, True),
            fg=COLORS["green"] if (s["curve"] or -1) >= 0 else COLORS["red"],
        )
        self.signal_values["inflation"].configure(
            text=(UNAVAILABLE_MESSAGE if s["cpi_change"] is None else f"{s['cpi_change']:+.2f} pp"),
            fg=COLORS["green"] if (s["cpi_change"] or 0) < 0 else COLORS["gold"],
        )

        cpi_change = s["cpi_change"]
        fed_change = s["fed_change"]
        curve_change = s["curve_change"]
        if cpi_change is None and fed_change is None and curve_change is None:
            self.momentum_title.configure(text=UNAVAILABLE_MESSAGE)
            self.momentum_detail.configure(
                text="Le cache est utilisé immédiatement ; les séries absentes arrivent ensuite en parallèle.",
                fg=COLORS["muted"],
            )
        else:
            inflation_word = "baisse" if (cpi_change or 0) < -0.10 else "hausse" if (cpi_change or 0) > 0.10 else "stable"
            fed_word = "baisse" if (fed_change or 0) < -0.10 else "hausse" if (fed_change or 0) > 0.10 else "stable"
            curve_word = "se pentifie" if (curve_change or 0) > 0.10 else "s'aplatit" if (curve_change or 0) < -0.10 else "stable"
            self.momentum_title.configure(text=f"Inflation en {inflation_word} · Fed en {fed_word}")
            self.momentum_detail.configure(
                text=f"Sur 12 mois : la courbe {curve_word}. Le dernier graphique montre ces variations directement en points de pourcentage.",
                fg=COLORS["blue"],
            )

    def _visible_ylim(
        self,
        point_sets: list[list[SeriesPoint]],
        start: date,
        end: date,
        floor_margin: float = 0.4,
    ) -> tuple[float, float] | None:
        vals: list[float] = []
        for points in point_sets:
            vals.extend(p.value for p in points if start <= p.day <= end and math.isfinite(p.value))
        if not vals:
            return None
        lo, hi = min(vals), max(vals)
        margin = max(floor_margin, abs(lo) * 0.08) if math.isclose(lo, hi) else max(floor_margin, (hi - lo) * 0.10)
        return lo - margin, hi + margin

    def _annotate_latest(self, ax: Any, points: list[SeriesPoint], color: str, label: str) -> None:
        if not points:
            return
        p = points[-1]
        ax.scatter([p.day], [p.value], s=28, color=color, edgecolors="white", linewidths=0.7, zorder=6)
        ax.annotate(
            f"{label} {p.value:.2f}%",
            (p.day, p.value),
            xytext=(-7, 8),
            textcoords="offset points",
            ha="right",
            va="bottom",
            color=color,
            fontsize=8.2,
            fontweight="bold",
        )

    def _render_inflation(self, start: date, end: date) -> None:
        ref = self.charts["inflation"]
        ax = ref.ax
        ax.clear()
        self._base_axis(ax, "Inflation YoY (%)")

        series = [
            ("cpi", "CPI", COLORS["gold"], 2.0),
            ("core_cpi", "Core CPI", COLORS["purple"], 1.8),
            ("pce", "PCE", COLORS["red"], 1.6),
        ]
        visible_sets: list[list[SeriesPoint]] = []
        has_data = False
        for key, label, color, width in series:
            points = visible_points(self.data[key], start, end)
            visible_sets.append(points)
            if points:
                has_data = True
                ax.plot(
                    [p.day for p in points],
                    [p.value for p in points],
                    color=color,
                    linewidth=width,
                    label=label,
                )

        ax.axhline(2.0, color=COLORS["green"], linewidth=1.0, linestyle=(0, (2, 4)), alpha=0.75, label="Cible 2 %")
        ax.set_xlim(start, end)
        ylim = self._visible_ylim(visible_sets, start, end, 0.7)
        if ylim:
            ax.set_ylim(*ylim)
        if has_data:
            ax.legend(loc="upper left", frameon=False, fontsize=8.3, ncol=2)
        else:
            self._empty_axis(ax, UNAVAILABLE_MESSAGE)
        ref.canvas.draw_idle()

    def _render_fed_rate(self, start: date, end: date) -> None:
        ref = self.charts["fed_rate"]
        ax = ref.ax
        ax.clear()
        self._base_axis(ax, "Taux (%)")

        points = downsample(visible_points(self.data["fed_funds"], start, end), 1800)
        if points:
            xs = [p.day for p in points]
            ys = [p.value for p in points]
            ax.plot(xs, ys, color=COLORS["blue"], linewidth=2.0)
            ax.fill_between(xs, ys, 0, color=COLORS["blue"], alpha=0.06)
            ax.set_xlim(start, end)
            lo, hi = min(ys), max(ys)
            margin = max(0.35, (hi - lo) * 0.10)
            ax.set_ylim(max(-0.2, lo - margin), hi + margin)
            self._annotate_latest(ax, points, COLORS["blue"], "Actuel")
        else:
            self._empty_axis(ax, UNAVAILABLE_MESSAGE)
        ref.canvas.draw_idle()

    def _render_yield_curve(self, start: date, end: date) -> None:
        ref = self.charts["yield_curve"]
        ax = ref.ax
        ax.clear()
        self._base_axis(ax, "Spread (%)")
        ax.axhline(0, color=COLORS["ink"], linewidth=1.0, alpha=0.85)

        points = downsample(visible_points(self.data["yield_curve"], start, end), 1700)
        if points:
            xs = [p.day for p in points]
            ys = [p.value for p in points]
            ax.fill_between(xs, ys, 0, where=[v < 0 for v in ys], color=COLORS["red"], alpha=0.14, interpolate=True)
            ax.fill_between(xs, ys, 0, where=[v >= 0 for v in ys], color=COLORS["green"], alpha=0.10, interpolate=True)
            ax.plot(xs, ys, color=COLORS["teal"], linewidth=1.8)
            lo, hi = min(min(ys), 0), max(max(ys), 0)
            margin = max(0.25, (hi - lo) * 0.12)
            ax.set_ylim(lo - margin, hi + margin)
            ax.set_xlim(start, end)
            self._annotate_latest(ax, points, COLORS["teal"], "Actuel")
        else:
            self._empty_axis(ax, UNAVAILABLE_MESSAGE)
        ref.canvas.draw_idle()

    def _render_recession(self, start: date, end: date) -> None:
        ref = self.charts["recession"]
        ax = ref.ax
        ax.clear()
        self._base_axis(ax, "Probabilité (%)", recession_shading=False)

        points = visible_points(self.data["recession_probability"], start, end)
        if points:
            xs = [p.day for p in points]
            ys = [p.value for p in points]
            max_y = max(ys)
            top = min(100.0, max(10.0, max_y * 1.18 + 2.0))
            if max_y >= 30:
                top = max(50.0, top)
            if top > 35:
                ax.axhspan(35, top, color=COLORS["red"], alpha=0.05)
                ax.axhline(35, color=COLORS["red"], linewidth=0.9, linestyle=(0, (4, 4)), alpha=0.7)
            if top > 20:
                ax.axhline(20, color=COLORS["gold"], linewidth=0.9, linestyle=(0, (3, 5)), alpha=0.65)
            ax.fill_between(xs, ys, 0, color=COLORS["teal"], alpha=0.14)
            ax.plot(xs, ys, color=COLORS["teal"], linewidth=1.9)
            ax.set_xlim(start, end)
            ax.set_ylim(0, top)
            self._annotate_latest(ax, points, COLORS["teal"], "Actuel")
        else:
            self._empty_axis(ax, UNAVAILABLE_MESSAGE)
        ref.canvas.draw_idle()

    def _render_policy(self, start: date, end: date) -> None:
        ref = self.charts["policy"]
        ax = ref.ax
        ax.clear()
        self._base_axis(ax, "Fed Funds – CPI (pp)")
        ax.axhline(0, color=COLORS["ink"], linewidth=1.0, alpha=0.85)

        spread = visible_points(build_policy_spread(self.data["fed_funds"], self.data["cpi"]), start, end)
        if spread:
            xs = [p.day for p in spread]
            ys = [p.value for p in spread]
            ax.fill_between(xs, ys, 0, where=[v >= 0 for v in ys], color=COLORS["green"], alpha=0.11, interpolate=True)
            ax.fill_between(xs, ys, 0, where=[v < 0 for v in ys], color=COLORS["red"], alpha=0.11, interpolate=True)
            ax.plot(xs, ys, color=COLORS["blue"], linewidth=1.9)
            lo, hi = min(min(ys), 0), max(max(ys), 0)
            margin = max(0.5, (hi - lo) * 0.10)
            ax.set_ylim(lo - margin, hi + margin)
            ax.set_xlim(start, end)
            self._annotate_latest(ax, spread, COLORS["blue"], "Actuel")
        else:
            self._empty_axis(ax, UNAVAILABLE_MESSAGE)
        ref.canvas.draw_idle()

    def _render_momentum(self) -> None:
        ref = self.charts["momentum"]
        ax = ref.ax
        ax.clear()
        ax.set_facecolor(COLORS["panel_alt"])

        real_rate = build_policy_spread(self.data["fed_funds"], self.data["cpi"])
        metrics = [
            ("Fed Funds", self.data["fed_funds"], COLORS["blue"]),
            ("CPI", self.data["cpi"], COLORS["gold"]),
            ("Core CPI", self.data["core_cpi"], COLORS["purple"]),
            ("10Y – 2Y", self.data["yield_curve"], COLORS["teal"]),
            ("Taux réel", real_rate, COLORS["green"]),
        ]

        rows: list[tuple[str, float, float, float, str]] = []
        for label, points, color in metrics:
            current, previous, change = change_12m(points)
            if current is None or previous is None or change is None:
                continue
            rows.append((label, current, previous, change, color))

        if not rows:
            self._empty_axis(ax, UNAVAILABLE_MESSAGE)
            ref.canvas.draw_idle()
            return

        labels = [r[0] for r in rows]
        changes = [r[3] for r in rows]
        colors = [r[4] for r in rows]
        y = list(range(len(rows)))

        ax.barh(y, changes, color=colors, alpha=0.84, height=0.54, zorder=3)
        ax.axvline(0, color=COLORS["ink"], linewidth=1.1, zorder=4)
        ax.grid(axis="x", color=COLORS["grid"], linewidth=0.8, alpha=0.95)
        ax.grid(axis="y", visible=False)
        ax.set_axisbelow(True)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, color=COLORS["ink_soft"], fontsize=10, fontweight="bold")
        ax.invert_yaxis()
        ax.set_xlabel("Variation sur 12 mois (points de pourcentage)", color=COLORS["muted"], fontsize=9.5, fontweight="bold")
        ax.tick_params(axis="x", colors=COLORS["muted"], labelsize=9)

        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(COLORS["line"])

        max_abs = max(abs(v) for v in changes) if changes else 1.0
        x_margin = max(0.30, max_abs * 0.28)
        x_limit = max(0.75, max_abs + x_margin)
        ax.set_xlim(-x_limit, x_limit)

        for idx, (label, current, previous, change, color) in enumerate(rows):
            sign = "+" if change > 0 else ""
            text = f"{previous:.2f}% → {current:.2f}%   ({sign}{change:.2f} pp)"
            if change >= 0:
                ax.text(
                    min(change + x_limit * 0.035, x_limit * 0.96),
                    idx,
                    text,
                    va="center",
                    ha="left",
                    fontsize=9.2,
                    color=COLORS["ink_soft"],
                    fontweight="bold",
                )
            else:
                ax.text(
                    max(change - x_limit * 0.035, -x_limit * 0.96),
                    idx,
                    text,
                    va="center",
                    ha="right",
                    fontsize=9.2,
                    color=COLORS["ink_soft"],
                    fontweight="bold",
                )

        ax.text(
            0.01,
            0.98,
            "Lecture directe : la longueur = ampleur du changement. Aucun score composite, aucune normalisation.",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.7,
            color=COLORS["muted"],
        )
        ref.canvas.draw_idle()

    def _render_footer(self) -> None:
        names = {
            "fed_funds": "Fed Funds",
            "cpi": "CPI",
            "core_cpi": "Core CPI",
            "pce": "PCE",
            "yield_curve": "10Y–2Y",
            "recession_probability": "Récession",
        }
        states = "  ·  ".join(f"{names[key]} {self.source_state.get(key, '—')}" for key in names)
        if self.updated_at:
            stamp = self.updated_at.strftime("%Y-%m-%d %H:%M")
            prefix = f"Dernière actualisation {stamp}"
        else:
            prefix = "Session locale"
        self.footer.configure(text=f"{prefix}  ·  {states}  ·  Source : FRED  ·  Cache : {CACHE_DIR}")


def main() -> None:
    app = MacroDashboard()
    app.mainloop()


if __name__ == "__main__":
    main()
