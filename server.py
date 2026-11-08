"""Football Analyst MCP server for Claude, powered by API-Football (API-Sports)."""
from __future__ import annotations

import asyncio
import json
import math
import os
import re
import time
from datetime import date, datetime
from typing import Any, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx
from mcp.server.fastmcp import FastMCP

API_BASE = "https://v3.football.api-sports.io"
API_KEY = os.getenv("API_FOOTBALL_KEY", "").strip()
DEFAULT_TZ = os.getenv("FOOTBALL_TIMEZONE", "Africa/Cairo").strip() or "Africa/Cairo"

mcp = FastMCP(
    "football-analyst",
    instructions=(
        "أنت محلل كرة قدم يعتمد على بيانات فعلية من API-Football. "
        "استخدم list_matches لعرض كل مباريات التاريخ المطلوب، ثم analyze_match أو "
        "analyze_today_matches للتحليل. افصل بين بيانات المصدر والحسابات الإحصائية "
        "والاستنتاجات. اعرض نسب الاحتمالات لا كضمانات. لا تخترع إحصائيات ناقصة. "
        "في التسديدات داخل المنطقة والركنيات، اذكر عدد المباريات التي بُني عليها المتوسط. "
        "لا تقدم توصية رهان قطعية، واذكر عندما تكون البيانات أو التغطية ضعيفة."
    ),
)

_client: Optional[httpx.AsyncClient] = None
_cache: dict[str, tuple[float, Any]] = {}
_cache_lock = asyncio.Lock()
_api_semaphore = asyncio.Semaphore(5)
_last_rate_info: dict[str, Any] = {}


def _json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, default=str)


def _valid_timezone(tz_name: str) -> str:
    try:
        ZoneInfo(tz_name)
        return tz_name
    except (ZoneInfoNotFoundError, ValueError):
        raise ValueError(f"منطقة زمنية غير معروفة: {tz_name}")


def _date_string(date_value: Optional[str], tz_name: str) -> str:
    if not date_value:
        return datetime.now(ZoneInfo(tz_name)).date().isoformat()
    try:
        return date.fromisoformat(date_value).isoformat()
    except ValueError:
        raise ValueError("التاريخ يجب أن يكون بصيغة YYYY-MM-DD")


def _cache_ttl(endpoint: str, params: dict[str, Any]) -> int:
    if endpoint == "fixtures" and "date" in params:
        return 60
    if endpoint == "fixtures" and "id" in params:
        return 300
    if endpoint == "fixtures" and "team" in params:
        return 6 * 3600
    if endpoint == "fixtures/statistics":
        return 60 if params.get("live") else 24 * 3600
    if endpoint == "predictions":
        return 45 * 60
    if endpoint == "injuries":
        return 3 * 3600
    if endpoint == "fixtures/headtohead":
        return 12 * 3600
    return 30 * 60


async def _api_get(endpoint: str, params: Optional[dict[str, Any]] = None) -> list[dict[str, Any]]:
    global _client, _last_rate_info
    if not API_KEY:
        raise RuntimeError(
            "مفتاح API غير مضبوط. ثبّت الإضافة وأدخل API-Football API Key في إعداداتها."
        )
    params = {k: v for k, v in (params or {}).items() if v is not None and v != ""}
    key = endpoint + "?" + json.dumps(params, sort_keys=True, ensure_ascii=False)
    now = time.monotonic()
    async with _cache_lock:
        cached = _cache.get(key)
        if cached and now - cached[0] < _cache_ttl(endpoint, params):
            return cached[1]

    if _client is None:
        _client = httpx.AsyncClient(
            base_url=API_BASE,
            headers={"x-apisports-key": API_KEY, "Accept": "application/json"},
            timeout=httpx.Timeout(25.0, connect=10.0),
        )

    async with _api_semaphore:
        try:
            response = await _client.get("/" + endpoint.lstrip("/"), params=params)
            _last_rate_info = {
                "remaining": response.headers.get("x-ratelimit-requests-remaining"),
                "daily_limit": response.headers.get("x-ratelimit-requests-limit"),
            }
            response.raise_for_status()
            if response.status_code == 204 or not response.content:
                payload = {"response": [], "errors": {}}
            else:
                payload = response.json()
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status == 401 or status == 403:
                raise RuntimeError("مفتاح API غير صالح أو لا يملك صلاحية لهذا الطلب.") from None
            if status == 429:
                raise RuntimeError("تم بلوغ حد الطلبات المسموح به من مزود البيانات (Rate Limit).") from None
            raise RuntimeError(f"خطأ من مزود البيانات: HTTP {status}") from None
        except (httpx.HTTPError, ValueError):
            raise RuntimeError("تعذر الاتصال بمزود بيانات المباريات أو قراءة رده.") from None

    errors = payload.get("errors")
    if errors:
        raise RuntimeError(f"API-Football أعاد خطأ: {errors}")
    result = payload.get("response") or []
    if not isinstance(result, list):
        result = [result]
    async with _cache_lock:
        _cache[key] = (time.monotonic(), result)
    return result


def _fixture_status(fx: dict[str, Any]) -> str:
    return str((fx.get("fixture") or {}).get("status", {}).get("short") or "UNK")


def _finished(status: str) -> bool:
    return status in {"FT", "AET", "PEN"}


def _live(status: str) -> bool:
    return status in {"1H", "HT", "2H", "ET", "P", "LIVE", "BT", "INT", "SUSP", "AWD", "ABD"}


def _fixture_to_brief(item: dict[str, Any], tz_name: str) -> dict[str, Any]:
    fx = item.get("fixture") or {}
    teams = item.get("teams") or {}
    league = item.get("league") or {}
    goals = item.get("goals") or {}
    home = teams.get("home") or {}
    away = teams.get("away") or {}
    kickoff = fx.get("date")
    local_kickoff = kickoff
    if kickoff:
        try:
            dt = datetime.fromisoformat(kickoff.replace("Z", "+00:00"))
            local_kickoff = dt.astimezone(ZoneInfo(tz_name)).strftime("%Y-%m-%d %H:%M")
        except (ValueError, ZoneInfoNotFoundError):
            pass
    return {
        "fixture_id": fx.get("id"),
        "kickoff_local": local_kickoff,
        "status": _fixture_status(item),
        "status_long": (fx.get("status") or {}).get("long"),
        "league": league.get("name"),
        "country": league.get("country"),
        "season": league.get("season"),
        "round": league.get("round"),
        "home": home.get("name"),
        "away": away.get("name"),
        "home_id": home.get("id"),
        "away_id": away.get("id"),
        "score": {"home": goals.get("home"), "away": goals.get("away")},
        "venue": (fx.get("venue") or {}).get("name"),
    }


async def _fetch_day_matches(
    date_value: Optional[str], timezone: str, league_filter: Optional[str],
    country_filter: Optional[str], team_filter: Optional[str], status_filter: Optional[str],
    max_results: int,
) -> dict[str, Any]:
    tz_name = _valid_timezone(timezone or DEFAULT_TZ)
    target_date = _date_string(date_value, tz_name)
    rows = await _api_get("fixtures", {"date": target_date, "timezone": tz_name})
    lf = (league_filter or "").strip().casefold()
    cf = (country_filter or "").strip().casefold()
    tf = (team_filter or "").strip().casefold()
    sf = (status_filter or "").strip().casefold()
    selected: list[dict[str, Any]] = []
    for row in rows:
        brief = _fixture_to_brief(row, tz_name)
        if lf and lf not in str(brief.get("league") or "").casefold():
            continue
        if cf and cf not in str(brief.get("country") or "").casefold():
            continue
        if tf and tf not in (str(brief.get("home") or "") + " " + str(brief.get("away") or "")).casefold():
            continue
        status = brief["status"]
        if sf == "upcoming" and status not in {"NS", "TBD", "PST"}:
            continue
        if sf == "live" and not _live(status):
            continue
        if sf == "finished" and not _finished(status):
            continue
        selected.append(brief)

    selected.sort(key=lambda x: str(x.get("kickoff_local") or ""))
    limit = max(1, min(int(max_results), 250))
    return {
        "date": target_date,
        "timezone": tz_name,
        "matches_returned": min(len(selected), limit),
        "matches_found_before_limit": len(selected),
        "matches": selected[:limit],
        "data_provider": "API-Football / API-Sports",
        "rate_limit": _last_rate_info or None,
        "coverage_note": "هذه كل المباريات التي يعرضها مزود البيانات لهذا التاريخ، وليست ضمانًا بأن كل مباراة في العالم مغطاة.",
    }


@mcp.tool()
async def list_matches(
    date: Optional[str] = None,
    timezone: str = DEFAULT_TZ,
    league: Optional[str] = None,
    country: Optional[str] = None,
    team: Optional[str] = None,
    status: Optional[str] = None,
    max_results: int = 150,
) -> str:
    """اعرض جدول مباريات كرة القدم في تاريخ محدد، مع إمكانية التصفية بالدوري أو الدولة أو الفريق أو الحالة. استخدمه أولاً لجلب كل مباريات اليوم."""
    try:
        result = await _fetch_day_matches(date, timezone, league, country, team, status, max_results)
        return _json(result)
    except Exception as exc:
        return _json({"error": str(exc)})


def _num(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip().replace("%", "")
        if value in {"", "-", "null"}:
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _weighted_average(values: list[Optional[float]]) -> Optional[float]:
    usable = [v for v in values if v is not None]
    if len(usable) < 2:
        return None
    # Values are ordered newest first, so the newest observations receive more weight.
    weights = [1.0 / (i + 1) for i in range(len(values)) if values[i] is not None]
    return sum(v * w for v, w in zip(usable, weights)) / sum(weights)


def _team_stat_map(stat_rows: list[dict[str, Any]]) -> dict[int, dict[str, Optional[float]]]:
    mapped: dict[int, dict[str, Optional[float]]] = {}
    wanted = {
        "Shots insidebox": "shots_inside_box",
        "Corner Kicks": "corners",
        "Shots on Goal": "shots_on_target",
        "Total Shots": "total_shots",
    }
    for team_row in stat_rows:
        team_id = (team_row.get("team") or {}).get("id")
        if team_id is None:
            continue
        vals: dict[str, Optional[float]] = {}
        for stat in team_row.get("statistics") or []:
            name = stat.get("type")
            if name in wanted:
                vals[wanted[name]] = _num(stat.get("value"))
        mapped[int(team_id)] = vals
    return mapped


def _poisson(lam: float, k: int) -> float:
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def _probability_model(home_gf: float, home_ga: float, away_gf: float, away_ga: float) -> dict[str, Any]:
    # Transparent baseline model: blend each side's recent attack with the opponent's recent defence.
    lam_home = max(0.15, min(4.5, (home_gf + away_ga) / 2.0 * 1.04))
    lam_away = max(0.15, min(4.5, (away_gf + home_ga) / 2.0))
    max_goals = 10
    hp = [_poisson(lam_home, i) for i in range(max_goals + 1)]
    ap = [_poisson(lam_away, i) for i in range(max_goals + 1)]
    # Normalize truncated distributions to prevent the omitted tail from biasing totals.
    hs, ass = sum(hp), sum(ap)
    hp = [p / hs for p in hp]
    ap = [p / ass for p in ap]
    p_home = p_draw = p_away = p_btts = 0.0
    totals = {1.5: 0.0, 2.5: 0.0, 3.5: 0.0, 4.5: 0.0}
    best_score = (0, 0)
    best_score_p = -1.0
    for h, ph in enumerate(hp):
        for a, pa in enumerate(ap):
            p = ph * pa
            if h > a:
                p_home += p
            elif h == a:
                p_draw += p
            else:
                p_away += p
            if h > 0 and a > 0:
                p_btts += p
            if h + a > 0:
                totals[1.5] += p
            if h + a > 1:
                totals[2.5] += p
            if h + a > 2:
                totals[3.5] += p
            if h + a > 3:
                totals[4.5] += p
            if p > best_score_p:
                best_score, best_score_p = (h, a), p
    probs = {"home_win": p_home, "draw": p_draw, "away_win": p_away}
    likely = max(probs, key=probs.get)
    return {
        "expected_goals": {"home": round(lam_home, 2), "away": round(lam_away, 2), "total": round(lam_home + lam_away, 2)},
        "result_probabilities_pct": {k: round(v * 100, 1) for k, v in probs.items()},
        "most_likely_outcome": likely,
        "over_probabilities_pct": {f"over_{line}": round(value * 100, 1) for line, value in totals.items()},
        "under_probabilities_pct": {f"under_{line}": round((1 - value) * 100, 1) for line, value in totals.items()},
        "both_teams_to_score_pct": round(p_btts * 100, 1),
        "most_likely_score": f"{best_score[0]}-{best_score[1]}",
        "model_method": "توزيع بواسون مبسط، يمزج متوسط التسجيل الأخير للفريق مع متوسط استقبال الأهداف الأخير للمنافس؛ ليس نموذجًا معايرًا رسميًا.",
    }


def _form_summary(history: list[dict[str, Any]]) -> dict[str, Any]:
    form = "".join(x.get("result") or "?" for x in history[:5])
    return {
        "last_matches_count": len(history),
        "last_five_form_newest_first": form,
        "goals_for_avg": round(_weighted_average([x.get("goals_for") for x in history]) or 0, 2),
        "goals_against_avg": round(_weighted_average([x.get("goals_against") for x in history]) or 0, 2),
        "corners_avg": round(_weighted_average([x.get("corners_for") for x in history]) or 0, 2) if sum(x.get("corners_for") is not None for x in history) >= 2 else None,
        "shots_inside_box_avg": round(_weighted_average([x.get("shots_inside_box_for") for x in history]) or 0, 2) if sum(x.get("shots_inside_box_for") is not None for x in history) >= 2 else None,
        "stats_sample_corners": sum(x.get("corners_for") is not None for x in history),
        "stats_sample_shots_inside_box": sum(x.get("shots_inside_box_for") is not None for x in history),
    }


def _injury_summary(rows: list[dict[str, Any]], home_id: int, away_id: int) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        team = row.get("team") or {}
        player = row.get("player") or {}
        output.append({
            "team": team.get("name"),
            "team_id": team.get("id"),
            "player": player.get("name"),
            "type": player.get("type"),
            "reason": player.get("reason"),
        })
    return output[:30]


async def _analyze_fixture(fixture_id: int, recent_matches: int, include_injuries: bool, timezone: str) -> dict[str, Any]:
    tz_name = _valid_timezone(timezone)
    fixture_rows = await _api_get("fixtures", {"id": int(fixture_id), "timezone": tz_name})
    if not fixture_rows:
        return {"fixture_id": fixture_id, "error": "لم يتم العثور على المباراة بهذا المعرّف."}
    fixture = fixture_rows[0]
    brief = _fixture_to_brief(fixture, tz_name)
    teams = fixture.get("teams") or {}
    home, away = teams.get("home") or {}, teams.get("away") or {}
    home_id, away_id = home.get("id"), away.get("id")
    if not home_id or not away_id:
        return {"fixture": brief, "error": "بيانات الفريقين غير مكتملة."}

    n = max(2, min(int(recent_matches), 8))
    tasks = [
        _api_get("fixtures", {"team": home_id, "last": n, "timezone": tz_name}),
        _api_get("fixtures", {"team": away_id, "last": n, "timezone": tz_name}),
        _api_get("predictions", {"fixture": int(fixture_id)}),
        _api_get("fixtures/headtohead", {"h2h": f"{home_id}-{away_id}", "last": 5, "timezone": tz_name}),
    ]
    if include_injuries:
        tasks.append(_api_get("injuries", {"fixture": int(fixture_id), "timezone": tz_name}))
    else:
        tasks.append(asyncio.sleep(0, result=[]))
    results = await asyncio.gather(*tasks, return_exceptions=True)
    history_home = results[0] if isinstance(results[0], list) else []
    history_away = results[1] if isinstance(results[1], list) else []
    pred_rows = results[2] if isinstance(results[2], list) else []
    h2h_rows = results[3] if isinstance(results[3], list) else []
    injury_rows = results[4] if isinstance(results[4], list) else []
    nonfatal_errors = []
    for label, result in zip(["home_recent", "away_recent", "provider_predictions", "head_to_head", "injuries"], results):
        if isinstance(result, Exception):
            nonfatal_errors.append({"source": label, "error": str(result)})

    def valid_history(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        filtered = [r for r in rows if int(((r.get("fixture") or {}).get("id") or -1)) != int(fixture_id) and _finished(_fixture_status(r))]
        filtered.sort(key=lambda r: str((r.get("fixture") or {}).get("date") or ""), reverse=True)
        return filtered[:n]

    history_home = valid_history(history_home)
    history_away = valid_history(history_away)
    all_history = history_home + history_away
    unique_history: dict[int, dict[str, Any]] = {}
    for row in all_history:
        mid = (row.get("fixture") or {}).get("id")
        if mid is not None:
            unique_history[int(mid)] = row

    stat_map_by_fixture: dict[int, dict[int, dict[str, Optional[float]]]] = {}
    stat_results = await asyncio.gather(
        *[_api_get("fixtures/statistics", {"fixture": mid}) for mid in unique_history.keys()],
        return_exceptions=True,
    )
    for mid, stat_rows in zip(unique_history.keys(), stat_results):
        if isinstance(stat_rows, list):
            stat_map_by_fixture[mid] = _team_stat_map(stat_rows)
        else:
            nonfatal_errors.append({"source": f"statistics_fixture_{mid}", "error": str(stat_rows)})

    def make_history(rows: list[dict[str, Any]], team_id: int) -> list[dict[str, Any]]:
        out = []
        for row in rows:
            fx = row.get("fixture") or {}
            teams_obj = row.get("teams") or {}
            goals = row.get("goals") or {}
            h, a = teams_obj.get("home") or {}, teams_obj.get("away") or {}
            is_home = int(h.get("id") or -1) == int(team_id)
            own = goals.get("home") if is_home else goals.get("away")
            opp = goals.get("away") if is_home else goals.get("home")
            if own is None or opp is None:
                continue
            own_i, opp_i = int(own), int(opp)
            result_char = "W" if own_i > opp_i else "D" if own_i == opp_i else "L"
            mid = int(fx.get("id"))
            one_stats = stat_map_by_fixture.get(mid, {}).get(int(team_id), {})
            out.append({
                "date": fx.get("date"),
                "opponent": (a if is_home else h).get("name"),
                "venue": "home" if is_home else "away",
                "score": f"{own_i}-{opp_i}",
                "result": result_char,
                "goals_for": float(own_i),
                "goals_against": float(opp_i),
                "corners_for": one_stats.get("corners"),
                "shots_inside_box_for": one_stats.get("shots_inside_box"),
                "shots_on_target_for": one_stats.get("shots_on_target"),
            })
        return out

    home_history = make_history(history_home, int(home_id))
    away_history = make_history(history_away, int(away_id))
    home_form = _form_summary(home_history)
    away_form = _form_summary(away_history)
    model = None
    if len(home_history) >= 2 and len(away_history) >= 2:
        model = _probability_model(
            home_form["goals_for_avg"], home_form["goals_against_avg"],
            away_form["goals_for_avg"], away_form["goals_against_avg"],
        )

    corners_total = None
    box_shots_total = None
    if home_form["corners_avg"] is not None and away_form["corners_avg"] is not None:
        corners_total = round(home_form["corners_avg"] + away_form["corners_avg"], 1)
    if home_form["shots_inside_box_avg"] is not None and away_form["shots_inside_box_avg"] is not None:
        box_shots_total = round(home_form["shots_inside_box_avg"] + away_form["shots_inside_box_avg"], 1)

    provider_prediction = None
    if pred_rows:
        p = pred_rows[0].get("predictions") or {}
        percent = p.get("percent") or {}
        provider_prediction = {
            "winner": p.get("winner"),
            "percent": percent,
            "under_over": p.get("under_over"),
            "goals": p.get("goals"),
            "advice": p.get("advice"),
            "comparison": pred_rows[0].get("comparison"),
            "note": "توقع المزود مبني على خوارزميات وإحصائيات، ويذكر المزود أنه لا يستخدم أسعار شركات المراهنات في هذا التوقع.",
        }

    h2h = []
    for row in h2h_rows[:5]:
        b = _fixture_to_brief(row, tz_name)
        h2h.append({"date": b.get("kickoff_local"), "home": b.get("home"), "away": b.get("away"), "score": b.get("score"), "status": b.get("status")})

    confidence = "منخفضة"
    if model and len(home_history) >= 3 and len(away_history) >= 3:
        confidence = "متوسطة" if len(home_history) >= 5 and len(away_history) >= 5 else "منخفضة إلى متوسطة"
    if model and provider_prediction and len(home_history) >= 5 and len(away_history) >= 5:
        confidence = "متوسطة"  # This is data-coverage confidence, not calibrated probability.

    return {
        "fixture": brief,
        "data_coverage_confidence": confidence,
        "confidence_definition": "تقدير نوعي لاكتمال عينة البيانات فقط، وليس احتمال صحة التوقع.",
        "independent_statistical_model": model,
        "provider_prediction": provider_prediction,
        "recent_form": {"home": {"team": home.get("name"), **home_form, "matches": home_history}, "away": {"team": away.get("name"), **away_form, "matches": away_history}},
        "projected_match_statistics": {
            "total_corners_estimate": corners_total,
            "home_corners_avg_last_matches": home_form["corners_avg"],
            "away_corners_avg_last_matches": away_form["corners_avg"],
            "corners_sample": {"home": home_form["stats_sample_corners"], "away": away_form["stats_sample_corners"]},
            "total_shots_inside_box_estimate": box_shots_total,
            "home_shots_inside_box_avg_last_matches": home_form["shots_inside_box_avg"],
            "away_shots_inside_box_avg_last_matches": away_form["shots_inside_box_avg"],
            "shots_inside_box_sample": {"home": home_form["stats_sample_shots_inside_box"], "away": away_form["stats_sample_shots_inside_box"]},
            "note": "توقعات استرشادية محسوبة من إحصائيات آخر المباريات المتاحة. إذا كانت القيمة null فالإحصائية غير متاحة أو العينة أقل من مباراتين؛ لا تستبدلها بتخمين.",
        },
        "injuries_and_absences": _injury_summary(injury_rows, int(home_id), int(away_id)),
        "head_to_head_last_five": h2h,
        "data_warnings": nonfatal_errors,
        "rate_limit": _last_rate_info or None,
        "methodology_warning": "هذه احتمالات تقديرية وليست نتائج مؤكدة أو نصيحة مالية. الإصابات والتشكيلات قد تتغير، وبعض البطولات لا توفر إحصاءات التسديدات والركنيات.",
    }


@mcp.tool()
async def analyze_match(
    fixture_id: int,
    recent_matches: int = 3,
    include_injuries: bool = True,
    timezone: str = DEFAULT_TZ,
) -> str:
    """حلل مباراة واحدة: فورمة آخر المباريات، فوز/تعادل/خسارة، احتمالات الأهداف، توقع الركنيات والتسديدات داخل المنطقة، الإصابات والمواجهات السابقة."""
    try:
        result = await _analyze_fixture(int(fixture_id), recent_matches, include_injuries, timezone)
        return _json(result)
    except Exception as exc:
        return _json({"fixture_id": fixture_id, "error": str(exc)})


@mcp.tool()
async def analyze_today_matches(
    date: Optional[str] = None,
    timezone: str = DEFAULT_TZ,
    league: Optional[str] = None,
    max_matches: int = 5,
    recent_matches: int = 3,
    include_injuries: bool = True,
) -> str:
    """اعرض مباريات اليوم ثم نفّذ تحليلًا معمقًا لعدد محدد منها. استخدم list_matches أولًا إذا كان المطلوب رؤية الجدول الكامل دون تحليل كل مباراة."""
    try:
        schedule = await _fetch_day_matches(date, timezone, league, None, None, None, 250)
        candidates = [x for x in schedule["matches"] if x.get("status") not in {"PST", "CANC", "ABD", "AWD"}]
        candidates.sort(key=lambda x: (0 if x.get("status") in {"NS", "TBD"} else 1, str(x.get("kickoff_local") or "")))
        limit = max(1, min(int(max_matches), 8))
        chosen = candidates[:limit]
        analyses = []
        for item in chosen:
            mid = item.get("fixture_id")
            if not mid:
                continue
            analyses.append(await _analyze_fixture(int(mid), recent_matches, include_injuries, timezone))
        return _json({
            "date": schedule["date"],
            "timezone": schedule["timezone"],
            "all_matches_found": schedule["matches_found_before_limit"],
            "deep_analyses_returned": len(analyses),
            "deep_analysis_limit": limit,
            "analyses": analyses,
            "note": "تم جلب جدول جميع مباريات التاريخ، لكن التحليل العميق اقتصر على العدد المحدد لتجنب استهلاك رصيد API كبير. يمكن استدعاء analyze_match لمباراة أخرى من القائمة.",
        })
    except Exception as exc:
        return _json({"error": str(exc)})


if __name__ == "__main__":
    mcp.run()
