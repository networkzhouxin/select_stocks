# -*- coding: utf-8 -*-
"""Build segmented, offline K-line reviews for the cross-signal strategy.

JoinQuant fills are the execution authority. Local unadjusted daily bars are
used only as the chart background and are never rewritten by this module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime
from html import escape
import json
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Iterable, Mapping, Sequence

import pandas as pd


@dataclass(frozen=True)
class PeriodSpec:
    key: str
    start: str
    end: str
    log_path: str = ""


@dataclass(frozen=True)
class TradeMarker:
    timestamp: datetime
    code: str
    side: str
    price: float
    amount: int
    commission: float
    reason: str = ""
    buy_score: float | None = None
    reversal_score: float | None = None
    location_score: float | None = None
    trend_score: float | None = None
    volume_score: float | None = None
    realized_pnl: float | None = None
    return_pct: float | None = None
    hold_days: int | None = None


DEFAULT_PERIODS = (
    PeriodSpec(
        "2010-2014", "2010-01-01", "2014-12-31",
        r"C:\Users\xin\.codex\attachments\c2e99faa-710c-48be-816d-30fbaa54ce92\pasted-text.txt",
    ),
    PeriodSpec(
        "2015-2018", "2015-01-01", "2018-12-31",
        r"C:\Users\xin\.codex\attachments\f37bb66f-d763-4dfd-a0e4-10cc19c7da19\pasted-text.txt",
    ),
    PeriodSpec(
        "2019-2021", "2019-01-01", "2021-12-31",
        r"C:\Users\xin\.codex\attachments\ff4cb7e7-7a39-46e5-a0cf-1759fafb2459\pasted-text.txt",
    ),
    PeriodSpec(
        "2022-2023", "2022-01-01", "2023-12-31",
        r"C:\Users\xin\.codex\attachments\a5ef7f67-09e3-41bf-95a7-1d6195231c47\pasted-text.txt",
    ),
    PeriodSpec(
        "2024-latest", "2024-01-01", "2026-07-08",
        r"C:\Users\xin\.codex\attachments\a0a5f354-c098-48d5-bace-cf60d72a52cd\pasted-text.txt",
    ),
)

ETF_NAMES = {
    "159915": "创业板ETF",
    "159920": "恒生ETF",
    "159928": "消费ETF",
    "159985": "豆粕ETF",
    "510300": "沪深300ETF",
    "510880": "红利ETF",
    "512100": "中证1000ETF",
    "513050": "中概互联网ETF",
    "513100": "纳指ETF",
    "513500": "标普500ETF",
    "513880": "日经ETF",
    "518880": "黄金ETF",
}
ETF_CODES = tuple(ETF_NAMES)

DEFAULT_DAILY_ROOT = Path(r"G:\financial\history_data\按年份合并\日级")
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "reports" / "trade_charts"
DEFAULT_SEVEN_ZIP = Path(r"C:\Program Files\NVIDIA Corporation\NVIDIA App\7z.exe")

_PREFIX = r"(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+-\s+\w+\s+-\s+"
_BUY_RE = re.compile(
    _PREFIX
    + r"\[buy\]\s+(?P<code>\d{6})\.(?:XSHG|XSHE)\s+"
      r"buy=(?P<buy>-?[\d.]+)\s+rev=(?P<rev>-?[\d.]+)\s+"
      r"loc=(?P<loc>-?[\d.]+)\s+trend=(?P<trend>-?[\d.]+)\s+"
      r"vol=(?P<vol>-?[\d.]+)"
)
_SELL_RE = re.compile(
    _PREFIX
    + r"\[sell\]\s+(?P<code>\d{6})\.(?:XSHG|XSHE)\s+"
      r"reason=(?P<reason>.*?)\s+amount="
)
_FILL_RE = re.compile(
    _PREFIX
    + r".*?security=(?P<code>\d{6})\.(?:XSHG|XSHE).*?"
      r"action=(?P<action>open|close).*?\)\s+trade price:\s*(?P<price>[\d.]+),\s*"
      r"amount:(?P<amount>-?\d+),\s*commission:\s*(?P<commission>[\d.]+)"
)


def parse_joinquant_trade_log(text: str) -> list[TradeMarker]:
    """Extract actual filled orders and attach same-day strategy context."""
    buys: dict[tuple[str, str], dict[str, float]] = {}
    sells: dict[tuple[str, str], str] = {}
    fills: list[TradeMarker] = []
    pending_order_line = ""

    for raw_line in text.splitlines():
        line = raw_line
        if pending_order_line and "trade price:" in line and not re.match(r"^\d{4}-\d{2}-\d{2}", line):
            line = pending_order_line + line
            pending_order_line = ""
        elif (
            re.match(r"^\d{4}-\d{2}-\d{2}", line)
            and " - order StockOrder(" in line
            and "trade price:" not in line
        ):
            pending_order_line = line
            continue
        elif re.match(r"^\d{4}-\d{2}-\d{2}", line):
            pending_order_line = ""

        buy_match = _BUY_RE.search(line)
        if buy_match:
            key = (buy_match.group("timestamp")[:10], buy_match.group("code"))
            buys[key] = {
                "buy_score": float(buy_match.group("buy")),
                "reversal_score": float(buy_match.group("rev")),
                "location_score": float(buy_match.group("loc")),
                "trend_score": float(buy_match.group("trend")),
                "volume_score": float(buy_match.group("vol")),
            }
            continue

        sell_match = _SELL_RE.search(line)
        if sell_match:
            key = (sell_match.group("timestamp")[:10], sell_match.group("code"))
            sells[key] = sell_match.group("reason").strip()
            continue

        fill_match = _FILL_RE.search(line)
        if not fill_match:
            continue
        timestamp = datetime.strptime(fill_match.group("timestamp"), "%Y-%m-%d %H:%M:%S")
        code = fill_match.group("code")
        side = "buy" if fill_match.group("action") == "open" else "sell"
        key = (timestamp.strftime("%Y-%m-%d"), code)
        context = buys.get(key, {}) if side == "buy" else {}
        fills.append(TradeMarker(
            timestamp=timestamp,
            code=code,
            side=side,
            price=float(fill_match.group("price")),
            amount=abs(int(fill_match.group("amount"))),
            commission=float(fill_match.group("commission")),
            reason=sells.get(key, "") if side == "sell" else "",
            **context,
        ))

    return sorted(fills, key=lambda item: item.timestamp)


def pair_trade_outcomes(markers: Sequence[TradeMarker]) -> list[TradeMarker]:
    """Attach FIFO realized outcome to sell markers when a buy can be proven."""
    open_lots: dict[str, list[TradeMarker]] = {}
    result: list[TradeMarker] = []
    for marker in sorted(markers, key=lambda item: item.timestamp):
        if marker.side == "buy":
            open_lots.setdefault(marker.code, []).append(marker)
            result.append(marker)
            continue

        lots = open_lots.get(marker.code, [])
        if not lots:
            result.append(marker)
            continue
        buy = lots.pop(0)
        amount = min(buy.amount, marker.amount)
        pnl = amount * (marker.price - buy.price) - buy.commission - marker.commission
        outcome = replace(
            marker,
            realized_pnl=pnl,
            return_pct=(marker.price / buy.price - 1.0) * 100.0,
            hold_days=(marker.timestamp.date() - buy.timestamp.date()).days,
        )
        result.append(outcome)
    return result


def _serial_number(value: object) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def _marker_payload(marker: TradeMarker) -> dict[str, object]:
    payload = asdict(marker)
    payload.pop("timestamp")
    payload["date"] = marker.timestamp.strftime("%Y-%m-%d")
    return payload


def build_symbol_dataset(
    frame: pd.DataFrame,
    markers: Sequence[TradeMarker],
    period_start: str | None = None,
    period_end: str | None = None,
) -> dict[str, object]:
    """Convert one ETF's immutable daily bars and fills into JSON-safe data."""
    required = {"code", "date", "open", "high", "low", "close", "volume"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"daily frame missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("daily frame is empty")

    data = frame.copy()
    data["date"] = pd.to_datetime(data["date"], errors="raise")
    data = data.sort_values("date").drop_duplicates("date", keep="last")
    lower = pd.Timestamp(period_start) if period_start else None
    upper = pd.Timestamp(period_end) if period_end else None
    if lower is not None and data["date"].min() < lower:
        raise ValueError("daily frame contains dates outside period")
    if upper is not None and data["date"].max() > upper:
        raise ValueError("daily frame contains dates outside period")

    for column in ("open", "high", "low", "close", "volume"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["open", "high", "low", "close"])
    for window in (5, 10, 20, 60):
        data[f"ma{window}"] = data["close"].rolling(window, min_periods=window).mean()

    code = str(data["code"].iloc[0]).split(".")[0].zfill(6)
    raw_name = str(data["symbol"].iloc[0]) if "symbol" in data.columns else ""
    name = raw_name if raw_name and raw_name.lower() != "nan" else ETF_NAMES.get(code, code)
    symbol_markers = [item for item in markers if item.code == code]
    return {
        "code": code,
        "name": name,
        "dates": data["date"].dt.strftime("%Y-%m-%d").tolist(),
        "open": [_serial_number(value) for value in data["open"]],
        "high": [_serial_number(value) for value in data["high"]],
        "low": [_serial_number(value) for value in data["low"]],
        "close": [_serial_number(value) for value in data["close"]],
        "volume": [_serial_number(value) for value in data["volume"]],
        "ma5": [_serial_number(value) for value in data["ma5"]],
        "ma10": [_serial_number(value) for value in data["ma10"]],
        "ma20": [_serial_number(value) for value in data["ma20"]],
        "ma60": [_serial_number(value) for value in data["ma60"]],
        "buys": [_marker_payload(item) for item in symbol_markers if item.side == "buy"],
        "sells": [_marker_payload(item) for item in symbol_markers if item.side == "sell"],
    }


def render_period_page(period_key: str, datasets: Mapping[str, Mapping[str, object]]) -> str:
    """Render a standalone period page that references the shared Plotly bundle."""
    if not datasets:
        raise ValueError("cannot render a period without datasets")
    payload = json.dumps(datasets, ensure_ascii=False, separators=(",", ":"))
    options = "".join(
        f'<option value="{escape(code)}">{escape(code)} {escape(str(data.get("name", "")))}</option>'
        for code, data in datasets.items()
    )
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escape(period_key)} 交易 K 线复盘</title><script src="plotly.min.js"></script>
<style>
:root{{--bg:#f4f6f8;--panel:#fff;--ink:#18212b;--muted:#687381;--line:#d9dee5;--blue:#1769aa;--orange:#e26d21}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,"Microsoft YaHei",sans-serif}}
header{{background:#fff;border-bottom:1px solid var(--line);padding:18px 24px}} h1{{font-size:22px;margin:0 0 4px;letter-spacing:0}}
.sub{{color:var(--muted)}} main{{padding:16px 24px 26px}} .toolbar{{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-bottom:12px}}
select{{height:36px;min-width:230px;border:1px solid #b9c1cb;background:#fff;padding:0 10px;font:inherit}} .stats{{color:var(--muted)}}
#chart{{height:calc(100vh - 210px);min-height:560px;background:var(--panel);border:1px solid var(--line)}}
.note{{margin-top:10px;color:var(--muted);font-size:13px}} .legend{{font-weight:600;color:var(--ink)}}
@media(max-width:700px){{header,main{{padding-left:12px;padding-right:12px}} #chart{{height:720px;min-height:720px}} select{{width:100%}}}}
</style></head><body>
<header><h1>{escape(period_key)} 日 K 交易复盘</h1><div class="sub">cross-v0.3.2（由 combo-candidate 同逻辑回测日志生成）</div></header>
<main><div class="toolbar"><label for="etf-select">ETF</label><select id="etf-select">{options}</select><span id="stats" class="stats"></span></div>
<div id="chart"></div>
<div class="note"><span class="legend">数据口径：</span>买卖标记采用<strong>聚宽成交</strong>日志中的实际成交价；K 线采用<strong>本地日线</strong>原始行情。两套数据仅叠加展示，不修改、不反推、不强行对齐。页面仅用于复盘观察，不用于验证期调参。</div></main>
<script>
const datasets={payload};
const select=document.getElementById('etf-select'); const stats=document.getElementById('stats');
const fmt=n=>n==null?'--':Number(n).toFixed(2);
function markerText(items, side){{return items.map(m=>{{
  const base=`${{side==='buy'?'买入':'卖出'}} ${{m.date}}<br>聚宽成交价：${{m.price}}<br>数量：${{m.amount}}<br>佣金：${{m.commission}}`;
  if(side==='buy') return base+`<br>buy/rev/loc：${{fmt(m.buy_score)}}/${{fmt(m.reversal_score)}}/${{fmt(m.location_score)}}<br>trend/vol：${{fmt(m.trend_score)}}/${{fmt(m.volume_score)}}`;
  return base+`<br>原因：${{m.reason||'日志未记录'}}`+(m.realized_pnl==null?'':`<br>配对盈亏：${{fmt(m.realized_pnl)}} 元<br>区间收益：${{fmt(m.return_pct)}}%<br>持有：${{m.hold_days}} 天`);
}})}}
function draw(code){{const d=datasets[code]; const buy=d.buys||[], sell=d.sells||[];
 const traces=[
  {{type:'candlestick',name:'日 K',x:d.dates,open:d.open,high:d.high,low:d.low,close:d.close,increasing:{{line:{{color:'#d64545'}},fillcolor:'#d64545'}},decreasing:{{line:{{color:'#19945a'}},fillcolor:'#19945a'}},xaxis:'x',yaxis:'y'}},
  ...[[d.ma5,'MA5','#7a8794'],[d.ma10,'MA10','#9270ca'],[d.ma20,'MA20','#1677a6'],[d.ma60,'MA60','#c4861a']].map(([y,name,color])=>({{type:'scatter',mode:'lines',name,x:d.dates,y,line:{{width:name==='MA60'?1.8:1.2,color}},hoverinfo:'skip',xaxis:'x',yaxis:'y'}})),
  {{type:'bar',name:'成交量',x:d.dates,y:d.volume,marker:{{color:'#aeb8c2'}},opacity:.65,xaxis:'x2',yaxis:'y2',hovertemplate:'%{{x}}<br>成交量 %{{y:,.0f}}<extra></extra>'}},
  {{type:'scatter',mode:'markers',name:'买入',x:buy.map(m=>m.date),y:buy.map(m=>m.price),text:markerText(buy,'buy'),hovertemplate:'%{{text}}<extra></extra>',marker:{{symbol:'triangle-up',size:13,color:'#1769aa',line:{{color:'#fff',width:1}}}},xaxis:'x',yaxis:'y'}},
  {{type:'scatter',mode:'markers',name:'卖出',x:sell.map(m=>m.date),y:sell.map(m=>m.price),text:markerText(sell,'sell'),hovertemplate:'%{{text}}<extra></extra>',marker:{{symbol:'triangle-down',size:13,color:'#e26d21',line:{{color:'#fff',width:1}}}},xaxis:'x',yaxis:'y'}}
 ];
 const layout={{margin:{{l:62,r:28,t:48,b:45}},paper_bgcolor:'#fff',plot_bgcolor:'#fff',hovermode:'x unified',dragmode:'zoom',showlegend:true,
  legend:{{orientation:'h',x:0,y:1.08}},xaxis:{{domain:[0,1],anchor:'y',rangeslider:{{visible:false}},showgrid:true,gridcolor:'#edf0f3',rangebreaks:[{{bounds:['sat','mon']}}]}},
  yaxis:{{domain:[.26,1],title:'价格',fixedrange:false,gridcolor:'#e5e9ee'}},xaxis2:{{domain:[0,1],anchor:'y2',matches:'x',showticklabels:false,rangebreaks:[{{bounds:['sat','mon']}}]}},
  yaxis2:{{domain:[0,.17],title:'成交量',gridcolor:'#eef1f4'}}}};
 Plotly.react('chart',traces,layout,{{responsive:true,displaylogo:false,scrollZoom:true}});
 stats.textContent=`${{d.dates[0]}} 至 ${{d.dates[d.dates.length-1]}} · 买入 ${{buy.length}} 次 · 卖出 ${{sell.length}} 次`;
}}
select.addEventListener('change',()=>draw(select.value)); draw(select.value);
</script></body></html>"""


def render_index_page(summary: Mapping[str, Mapping[str, object]]) -> str:
    cards = "".join(
        f'<a class="card" href="{escape(key)}.html"><strong>{escape(key)}</strong>'
        f'<span>{int(item["fills"])} 笔成交 · {int(item["symbols"])} 只 ETF</span>'
        f'<small>日线截至 {escape(str(item["last_kline"]))}</small></a>'
        for key, item in summary.items()
    )
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Cross Signal 分段交易复盘</title><style>
*{{box-sizing:border-box}} body{{margin:0;background:#f4f6f8;color:#18212b;font:15px/1.5 system-ui,"Microsoft YaHei",sans-serif}}
main{{max-width:980px;margin:auto;padding:36px 24px}} h1{{font-size:28px;margin:0 0 6px;letter-spacing:0}} p{{color:#687381;margin:0 0 24px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px}} .card{{display:flex;flex-direction:column;gap:7px;background:#fff;color:inherit;text-decoration:none;border:1px solid #d9dee5;border-left:4px solid #1769aa;padding:17px}}
.card:hover{{border-color:#1769aa}} .card strong{{font-size:19px}} .card span{{color:#364250}} .card small{{color:#7a8591}} .note{{margin-top:22px;padding-top:16px;border-top:1px solid #d9dee5;font-size:13px;color:#687381}}
</style></head><body><main><h1>Cross Signal 分段交易复盘</h1><p>选择一个回测区间，逐只 ETF 查看每日 K 线与聚宽真实买卖成交点。</p><div class="grid">{cards}</div>
<div class="note">这是观察报告，不参与策略评分、参数选择或验证期调优。聚宽成交与本地日线保留各自原始口径。</div></main></body></html>"""


def _archive_path(daily_root: Path, year: int) -> Path:
    return daily_root / str(year) / f"{year}_日K.7z"


def load_period_daily_bars(
    period: PeriodSpec,
    daily_root: Path = DEFAULT_DAILY_ROOT,
    seven_zip: Path = DEFAULT_SEVEN_ZIP,
    codes: Iterable[str] = ETF_CODES,
) -> dict[str, pd.DataFrame]:
    """Read selected members from yearly archives without touching source data."""
    if not seven_zip.is_file():
        raise FileNotFoundError(f"7z executable not found: {seven_zip}")
    start_year, end_year = int(period.start[:4]), int(period.end[:4])
    requested = tuple(codes)
    chunks: dict[str, list[pd.DataFrame]] = {code: [] for code in requested}
    with tempfile.TemporaryDirectory(prefix="cross_signal_kline_") as temp_name:
        temp_root = Path(temp_name)
        for year in range(start_year, end_year + 1):
            archive = _archive_path(daily_root, year)
            if not archive.is_file():
                raise FileNotFoundError(f"daily archive not found: {archive}")
            year_dir = temp_root / str(year)
            year_dir.mkdir()
            members = [f"{code}.csv" for code in requested]
            completed = subprocess.run(
                [str(seven_zip), "e", str(archive), *members, f"-o{year_dir}", "-y"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
            if completed.returncode not in (0, 1):
                raise RuntimeError(f"7z extraction failed for {archive}: {completed.stderr.strip()}")
            for code in requested:
                csv_path = year_dir / f"{code}.csv"
                if csv_path.is_file():
                    chunks[code].append(pd.read_csv(csv_path, dtype={"code": str}))

    lower, upper = pd.Timestamp(period.start), pd.Timestamp(period.end)
    result: dict[str, pd.DataFrame] = {}
    for code, frames in chunks.items():
        if not frames:
            continue
        frame = pd.concat(frames, ignore_index=True)
        dates = pd.to_datetime(frame["date"], errors="coerce")
        frame = frame.loc[(dates >= lower) & (dates <= upper)].copy()
        if frame.empty:
            continue
        frame["code"] = code
        frame["symbol"] = ETF_NAMES.get(code, code)
        result[code] = frame
    return result


def generate_all_reports(
    periods: Sequence[PeriodSpec] = DEFAULT_PERIODS,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, dict[str, object]]:
    """Generate the five offline HTML reports and their index."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict[str, object]] = {}
    for period in periods:
        log_path = Path(period.log_path)
        if not log_path.is_file():
            raise FileNotFoundError(f"JoinQuant log not found: {log_path}")
        markers = pair_trade_outcomes(parse_joinquant_trade_log(log_path.read_text(encoding="utf-8")))
        frames = load_period_daily_bars(period)
        datasets = {
            code: build_symbol_dataset(
                frame,
                markers,
                period_start=period.start,
                period_end=period.end,
            )
            for code, frame in frames.items()
        }
        (output_dir / f"{period.key}.html").write_text(
            render_period_page(period.key, datasets), encoding="utf-8"
        )
        last_kline = max(str(data["dates"][-1]) for data in datasets.values())
        summary[period.key] = {
            "fills": len(markers),
            "symbols": len(datasets),
            "last_kline": last_kline,
        }

    try:
        from plotly.offline import get_plotlyjs
    except ImportError as exc:
        raise RuntimeError("plotly is required to generate the offline JavaScript bundle") from exc
    (output_dir / "plotly.min.js").write_text(get_plotlyjs(), encoding="utf-8")
    (output_dir / "index.html").write_text(render_index_page(summary), encoding="utf-8")
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


if __name__ == "__main__":
    generated = generate_all_reports()
    print(json.dumps(generated, ensure_ascii=False, indent=2))
