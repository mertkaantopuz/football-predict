"""
data_collector.py
PitchAPI futbol verisi toplayıcı (model eğitimi için).

Kurulum:
    pip install requests pandas openpyxl python-dotenv

.env1 (veya .env) dosyası (Colab'da bunun yerine Secrets kullanılır):
    PITCH_API_KEY=pk_live_...

Colab'da: OUTPUT_DIR ortam değişkenini Drive klasörüne ayarlayın (bkz. adımlar).

Kullanım:
    python data_collector.py --test                 -> tek lig/sezon API testi
    python data_collector.py                        -> bir kez güncelle
    python data_collector.py --watch --interval 6   -> 6 saatte bir güncelle
    python data_collector.py --refresh-history      -> geçmiş sezonları da yeniden tara

Çıktılar (data/ klasörü):
    football_data.sqlite   -> asıl veri (matches, match_stats tabloları)
    matches.csv            -> maçlar + hedef etiketler
    match_stats.csv        -> maç başına TÜM istatistikler (geniş format)
    match_events.csv       -> gol dakikaları, ilk gol, penaltı/kendi kalesine gol özeti
    model_dataset.csv      -> bitmiş maçlar: maç bilgisi + etiketler + tüm istatistikler
    football_all_seasons.xlsx

Notlar:
- Geçmiş sezonlar (bir lig için DB'de zaten maç varsa) tekrar taranmaz.
  Sadece güncel sezon ve eksik olanlar çekilir. Zorlamak için --refresh-history.
- İstatistikler yalnızca bitmiş maçlar için, bir kez çekilir. Her 200 maçta
  bir kayıt yapılır; program yarıda kesilirse kaldığı yerden devam eder.
- API'nin döndürmediği HT skorları boş kalır (ilk yarı etiketleri o maçlarda boş olur).
"""

import os
import re
import time
import logging
import sqlite3
import argparse
from datetime import datetime, timedelta
from pathlib import Path

import requests
import pandas as pd
from dotenv import load_dotenv


# -------------------- AYARLAR --------------------

load_dotenv(".env")
load_dotenv(".env1")

API_KEY = os.getenv("PITCH_API_KEY")
BASE_URL = "https://api.pitchapi.dev/v1"
CURRENT_SEASON = "2026/2027"
HISTORICAL_SEASONS = [
    "2021/2022", "2022/2023", "2023/2024",
    "2024/2025", "2025/2026",
]
SEASONS = HISTORICAL_SEASONS + [CURRENT_SEASON]

# Colab'da OUTPUT_DIR ortam değişkeni Drive klasörüne ayarlanır.
# Ayarlanmazsa yerel "data" klasörü kullanılır.
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "data"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXCEL_PATH = OUTPUT_DIR / "football_all_seasons.xlsx"
MATCHES_CSV = OUTPUT_DIR / "matches.csv"
STATS_CSV = OUTPUT_DIR / "match_stats.csv"
MODEL_CSV = OUTPUT_DIR / "model_dataset.csv"
EVENTS_CSV = OUTPUT_DIR / "match_events.csv"
DB_PATH = OUTPUT_DIR / "football_data.sqlite"

DEFAULT_UPDATE_INTERVAL_HOURS = 12

REQUEST_TIMEOUT = 30
REQUEST_PAUSE = 0.15
RETRY_COUNT = 3
MAX_PAGES = 100            # sayfalama güvenlik sınırı
STATS_CHECKPOINT_EVERY = 200
EVENTS_VERSION = 2         # olay ayrıştırma mantığı değişince artır

MAX_FETCH_ATTEMPTS = 3     # bir maçın stats/olayı bu kadar başarısız denemeden sonra bırakılır
RECHECK_DAYS = 3           # yeni biten maçlarda boş gelen stats/olay bu süre boyunca tekrar denenir

FINISHED_STATUSES = {"finished", "ft", "ended", "completed", "full_time"}

# Colab Secrets yedeği: ortam değişkeni/.env yoksa Colab'dan okumayı dene
if not API_KEY:
    try:
        from google.colab import userdata
        API_KEY = userdata.get("PITCH_API_KEY")
    except Exception:
        pass

if not API_KEY:
    raise RuntimeError(
        "PITCH_API_KEY bulunamadı. .env1/.env dosyasına ya da Colab Secrets'a ekleyin."
    )

session = requests.Session()
session.headers.update({"X-API-KEY": API_KEY})


# -------------------- 15 ORGANİZASYON --------------------

LEAGUES = [
    {"country": "England", "name": "Premier League", "id": "l_4WFCIZ"},
    {"country": "England", "name": "Championship", "id": "l_4DWyTT"},

    {"country": "Spain", "name": "LaLiga", "id": "l_0ErfuF"},
    {"country": "Spain", "name": "LaLiga2", "id": "l_3kyHLd"},

    {"country": "Turkey", "name": "Super Lig", "id": "l_0S1uaf"},
    {"country": "Turkey", "name": "1. Lig", "id": "l_1BFfna"},

    {"country": "Italy", "name": "Serie A", "id": "l_0ALvwF"},
    {"country": "Italy", "name": "Serie B", "id": "l_0jy5yE"},

    {"country": "Germany", "name": "Bundesliga", "id": "l_1Isor4"},
    {"country": "Germany", "name": "2. Bundesliga", "id": "l_3hvsne"},

    {"country": "France", "name": "Ligue 1", "id": "l_3FJFUl"},
    {"country": "France", "name": "Ligue 2", "id": "l_0275VP"},

    {"country": "Europe", "name": "Champions League", "id": "l_0bfbkO"},

    # Bu ikisinin ID'si /leagues kataloğundan bulunur
    {"country": "Europe", "name": "Europa League", "id": None},
    {"country": "Europe", "name": "Conference League", "id": None},
]


# -------------------- HTTP YARDIMCISI --------------------

def api_get(url, params=None):
    """
    Başarılıysa data (dict) döndürür, hata varsa None.
    Boş ama başarılı cevap {} olarak döner (hata ile ayrışır).
    """
    for attempt in range(1, RETRY_COUNT + 1):
        try:
            response = session.get(url, params=params, timeout=REQUEST_TIMEOUT)

            if response.status_code == 429:
                wait = int(response.headers.get("Retry-After", 3))
                logging.warning("Rate limit. %s saniye bekleniyor.", wait)
                time.sleep(wait)
                continue

            if response.status_code == 404:
                logging.warning("404: %s %s", url, params)
                return None

            response.raise_for_status()
            payload = response.json()

            if payload.get("error"):
                raise RuntimeError(payload["error"])

            time.sleep(REQUEST_PAUSE)
            data = payload.get("data")
            return data if data is not None else {}

        except (requests.RequestException, ValueError, RuntimeError) as exc:
            if attempt == RETRY_COUNT:
                logging.error("İstek başarısız: %s | %s", url, exc)
                return None
            wait = attempt * 2
            logging.warning(
                "İstek hatası (%s/%s): %s. %s sn sonra tekrar.",
                attempt, RETRY_COUNT, exc, wait
            )
            time.sleep(wait)

    return None


# -------------------- LİG ID'LERİNİ TAMAMLA --------------------

def normalize_name(name):
    return (
        str(name).lower()
        .replace("uefa ", "")
        .replace("'", "")
        .replace("-", " ")
        .strip()
    )


def resolve_european_league_ids():
    """Europa League ve Conference League ID'lerini katalogdan bulur."""
    if all(item["id"] for item in LEAGUES):
        return

    catalog_data = api_get(f"{BASE_URL}/leagues")
    if not catalog_data:
        raise RuntimeError("PitchAPI /leagues kataloğu alınamadı.")

    catalog = catalog_data.get("leagues", [])

    aliases = {
        "Europa League": ["europa league"],
        "Conference League": ["conference league"],
    }
    # Yanlış eşleşmeleri engellemek için dışlanan kelimeler
    excluded = ["women", "youth", "u19", "u21", "u17", "qualif", "super cup"]

    for item in LEAGUES:
        if item["id"]:
            continue

        target = item["name"]
        candidates = aliases.get(target, [normalize_name(target)])
        found = None

        for league in catalog:
            api_name = normalize_name(league.get("name", ""))
            if any(x in api_name for x in excluded):
                continue
            if target == "Europa League" and "conference" in api_name:
                continue
            if any(alias in api_name for alias in candidates):
                found = league
                break

        if found:
            item["id"] = found.get("id")
            item["api_name"] = found.get("name")
            logging.info("Lig ID bulundu: %s -> %s (%s)",
                         target, item["id"], item["api_name"])
        else:
            logging.error(
                "%s katalogda bulunamadı. Olası UEFA/Avrupa ligleri: %s",
                target,
                [x.get("name") for x in catalog
                 if "uefa" in x.get("name", "").lower()
                 or "europe" in x.get("name", "").lower()]
            )

    missing = [x["name"] for x in LEAGUES if not x["id"]]
    if missing:
        raise RuntimeError(
            "Şu liglerin ID'si çözümlenemedi: " + ", ".join(missing)
            + ". Katalogdaki isimleri kontrol edin."
        )


# -------------------- TEST --------------------

def test_api(league_index=0, season="2025/2026"):
    """Tek lig + tek sezon için ham cevap yapısını yazdırır (--test)."""
    resolve_european_league_ids()
    league = LEAGUES[league_index]
    url = f"{BASE_URL}/leagues/{league['id']}/matches"
    response = session.get(
        url, params={"season": season, "status": "all"},
        timeout=REQUEST_TIMEOUT
    )
    print("HTTP Status:", response.status_code)
    payload = response.json()
    print("Response keys:", list(payload.keys()))
    print("API error:", payload.get("error"))
    data = payload.get("data", {})
    if isinstance(data, dict):
        print("Data keys:", list(data.keys()))
        matches = data.get("matches", [])
        print("Matches count (1. sayfa):", len(matches))
        print("Pagination:", data.get("pagination"))
        print("Meta:", data.get("meta"))
        if matches:
            print("İlk maç alanları:", list(matches[0].keys()))
            status_counts = {}
            for m in matches:
                s = m.get("status")
                status_counts[s] = status_counts.get(s, 0) + 1
            print("Status değerleri:", status_counts)

            # Bitmiş bir maç seç (stats için)
            finished = [m for m in matches
                        if str(m.get("status")).lower() in FINISHED_STATUSES]
            sample = finished[0] if finished else matches[0]
            stats = api_get(f"{BASE_URL}/matches/{sample.get('id')}/stats")
            if isinstance(stats, dict):
                print("Stats anahtarları:", list(stats.keys()))
                for p in stats.get("periods", []):
                    n_items = sum(len(g.get("items", [])) for g in p.get("groups", []))
                    print("  period:", p.get("period"), "| istatistik sayısı:", n_items)
                first_period = (stats.get("periods") or [{}])[0]
                print("Grup ve istatistik anahtarları (ilk period):")
                for g in first_period.get("groups", []):
                    keys = [i.get("key") or i.get("title") for i in g.get("items", [])]
                    print("  ", g.get("name") or g.get("title"), "->", keys)

            # HT skoru için alternatif endpoint'leri dene
            for suffix in ("", "/events", "/timeline"):
                detail = api_get(f"{BASE_URL}/matches/{sample.get('id')}{suffix}")
                label = f"/matches/{{id}}{suffix}"
                if isinstance(detail, dict):
                    print(f"{label} anahtarları:", list(detail.keys()))
                    if suffix == "/events":
                        evs = detail.get("events") or []
                        print("  olay sayısı:", len(evs))
                        types = {}
                        for e in evs:
                            if isinstance(e, dict):
                                t = e.get("type") or e.get("event_type") or e.get("kind")
                                types[t] = types.get(t, 0) + 1
                        print("  olay tipleri:", types)
                        print("  ilk 3 olay:")
                        for e in evs[:3]:
                            print("   ", e)
                        goals = [e for e in evs if isinstance(e, dict) and "goal" in str(e).lower()]
                        print("  'goal' içeren ilk 3 olay:")
                        for e in goals[:3]:
                            print("   ", e)
                    inner = detail.get("match")
                    if isinstance(inner, dict):
                        print(f"{label} -> match alanları:", list(inner.keys()))
                else:
                    print(f"{label}: cevap yok / hata")


# -------------------- MAÇ VERİSİ --------------------

def fetch_all_matches(league, season):
    """Sayfalama destekli maç çekimi. Hata olursa None döner."""
    url = f"{BASE_URL}/leagues/{league['id']}/matches"
    all_matches, seen_ids = [], set()
    page = 1

    while page <= MAX_PAGES:
        params = {"season": season, "status": "all", "page": page}
        data = api_get(url, params=params)

        if data is None:
            if page == 1:
                return None
            logging.warning("Sayfa %d alınamadı, eldeki veri kullanılıyor.", page)
            break

        items = data.get("matches", []) or []
        new_items = [m for m in items if m.get("id") not in seen_ids]
        if not new_items:
            break  # boş sayfa ya da API 'page' parametresini yok sayıyor

        for m in new_items:
            seen_ids.add(m.get("id"))
        all_matches.extend(new_items)

        pagination = data.get("pagination") or {}
        total_pages = pagination.get("total_pages") or pagination.get("pages")
        if total_pages and page >= int(total_pages):
            break
        if pagination and (
            pagination.get("has_next") is False
            or pagination.get("has_more") is False
        ):
            break

        page += 1

    return all_matches


def get_matches(league, season):
    matches = fetch_all_matches(league, season)
    if matches is None:
        logging.error("Maçlar alınamadı: %s | %s", league["name"], season)
        return None

    rows = []
    for match in matches:
        home = match.get("home_team") or {}
        away = match.get("away_team") or {}

        ht_home = match.get("ht_score_home")
        if ht_home is None:
            ht_home = match.get("score_ht_home")
        ht_away = match.get("ht_score_away")
        if ht_away is None:
            ht_away = match.get("score_ht_away")

        rows.append({
            "match_id": str(match.get("id")),
            "season": season,
            "country": league["country"],
            "league": league.get("api_name", league["name"]),
            "league_id": league["id"],
            "date": match.get("date"),
            "time_utc": match.get("time_utc"),
            "home_team": home.get("name"),
            "away_team": away.get("name"),
            "home_team_id": home.get("id"),
            "away_team_id": away.get("id"),
            "home_score": match.get("score_home"),
            "away_score": match.get("score_away"),
            "ht_home_score": ht_home,
            "ht_away_score": ht_away,
            "status": match.get("status"),
            "round_name": match.get("round_name"),
            "stadium": match.get("stadium"),
            "referee": match.get("referee"),
        })

    return rows


# -------------------- MAÇ İSTATİSTİKLERİ (TÜMÜ) --------------------

_PERCENT_RE = re.compile(r"^\s*-?\d+(\.\d+)?\s*%\s*$")


def clean_value(value):
    """'55%' -> 55.0, '12' -> 12.0; diğer metinler olduğu gibi kalır."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return value
    text = str(value).strip()
    if text in ("", "-", "N/A", "null"):
        return None
    if _PERCENT_RE.match(text):
        return float(text.replace("%", "").strip())
    try:
        return float(text)
    except ValueError:
        return text


def safe_col(text):
    text = str(text).strip().lower()
    text = re.sub(r"[^\w]+", "_", text)
    return text.strip("_")


def flatten_match_stats(stats_data):
    """
    /matches/{id}/stats cevabındaki tüm period/group/item kayıtlarını düzleştirir.
    Sütun adı: {period}_{home|away}_{istatistik}
    Örn: all_home_expected_goals, 1st_away_total_shots
    Aynı key birden fazla grupta geçerse grup adı da eklenir (çakışma olmaz).
    """
    result = {}
    if not stats_data:
        return result

    for period_obj in stats_data.get("periods", []) or []:
        period = safe_col(period_obj.get("period", "all")) or "all"

        for group in period_obj.get("groups", []) or []:
            group_name = safe_col(group.get("name") or group.get("title") or "")

            for item in group.get("items", []) or []:
                key = item.get("key") or item.get("title") or item.get("name")
                if not key:
                    continue
                key = safe_col(key)

                base = f"{period}_{{side}}_{key}"
                if base.format(side="home") in result:
                    # API son grupta (özet) aynı istatistikleri tekrar veriyor;
                    # gruplar isimsizse tekrarı atla, isimliyse grup adıyla ekle.
                    if not group_name:
                        continue
                    base = f"{period}_{{side}}_{group_name}_{key}"

                result[base.format(side="home")] = clean_value(item.get("home"))
                result[base.format(side="away")] = clean_value(item.get("away"))

    return result


def get_match_stats(match_id):
    """Başarılıysa satır dict'i, hata varsa None döner."""
    data = api_get(f"{BASE_URL}/matches/{match_id}/stats")
    if data is None:
        return None

    row = {"match_id": str(match_id)}
    flat = flatten_match_stats(data)
    row["stats_available"] = 1 if flat else 0
    row.update(flat)
    return row


# -------------------- OLAYLAR (/events) -> HT SKORU VE GOL DAKİKALARI --------------------

def _period_order(period):
    p = str(period).lower()
    if p == "firsthalf":
        return 0
    if p == "secondhalf":
        return 1
    return 2


def _pre_score(e):
    """Gol olayındaki score_home/score_away: golden ÖNCEKİ skor (API böyle veriyor)."""
    try:
        h, a = e.get("score_home"), e.get("score_away")
        if h is None or a is None:
            return None
        return int(h), int(a)
    except (TypeError, ValueError):
        return None


def reconstruct_ht(goals, sides, final_h, final_a):
    """
    Bazı maçlarda API bir gol olayını hiç vermiyor. Her gol olayındaki 'golden önceki
    skor' alanından eksik golün hangi iki olay arasında olduğu bulunur. Eksik gol
    ilk yarının içinde ya da ikinci yarının içindeyse ilk yarı skoru kesin bellidir;
    iki yarı arasına denk geliyorsa (ya da belirlenemiyorsa) HT skoru None kalır.
    Çözümlenemezse None döner.
    """
    if not goals:
        return None
    if any(g["pre"] is None or s is None for g, s in zip(goals, sides)):
        return None

    cur = [0, 0]
    ht = [0, 0]
    ambiguous = False
    missing = 0
    gap_first = gap_last = False
    prev_period = None  # None: henüz olay yok

    for idx, (g, s) in enumerate(zip(goals, sides)):
        ph, pa = g["pre"]
        gh, ga = ph - cur[0], pa - cur[1]
        if gh < 0 or ga < 0:
            return None  # tutarsız veri
        if gh or ga:
            missing += gh + ga
            if idx == 0:
                gap_first = True
            cur = [ph, pa]
            if g["period"] == "firsthalf":
                pass  # boşluk ilk yarıda; aşağıda ht = cur
            elif prev_period in (None, "firsthalf"):
                ambiguous = True  # eksik gol ilk/ikinci yarı arasında belirsiz
        cur[0 if s == "home" else 1] += 1
        if g["period"] == "firsthalf":
            ht = list(cur)
        prev_period = g["period"]

    end_h, end_a = final_h - cur[0], final_a - cur[1]
    if end_h < 0 or end_a < 0:
        return None
    if end_h or end_a:
        missing += end_h + end_a
        gap_last = True
        if prev_period == "firsthalf":
            ambiguous = True

    return {
        "ht": None if ambiguous else tuple(ht),
        "missing": missing,
        "gap_first": gap_first,
        "gap_last": gap_last,
    }


def prepare_events_table(df):
    """
    Eski sürümde kendi kalesine goller yanlış takıma yazılıyordu. Kendi kalesine golü
    olan ya da skorla uyuşmayan eski satırları yeniden çekilmek üzere eler;
    diğerleri aynen korunur.
    """
    if df.empty or "ev_version" in df.columns:
        return df
    own = (
        pd.to_numeric(df["home_own_goals"], errors="coerce").fillna(0)
        + pd.to_numeric(df["away_own_goals"], errors="coerce").fillna(0)
    )
    stale = (own > 0) | (pd.to_numeric(df["ev_goal_check_ok"], errors="coerce") == 0)
    logging.info("Eski sürümden yeniden çekilecek olay satırı: %d", int(stale.sum()))
    keep = df[~stale].copy()
    keep["ev_version"] = EVENTS_VERSION
    keep["ev_missing_goals"] = 0
    keep["ev_ht_source"] = "events"
    return keep


def get_match_events(match_id, info):
    """
    /matches/{id}/events cevabından gol olaylarını okuyup maç düzeyinde özet üretir.
    info: home_team_id, away_team_id, home_score, away_score
    Hata varsa None döner (sonraki turda tekrar denenir).
    """
    data = api_get(f"{BASE_URL}/matches/{match_id}/events")
    if data is None:
        return None

    events = data.get("events") or []
    home_id, away_id = info.get("home_team_id"), info.get("away_team_id")
    final_h, final_a = info.get("home_score"), info.get("away_score")

    goals = []
    for e in events:
        if str(e.get("event_type", "")).lower() != "goal":
            continue
        period = str(e.get("period", "")).lower()
        if "shootout" in period or "penalties" in period:
            continue  # seri penaltı atışlarını sayma
        goals.append({
            "minute": (e.get("minute") or 0) + (e.get("minute_added") or 0),
            "period": period,
            "team_id": e.get("team_id"),
            "own": bool(e.get("is_own_goal")),
            "pen": bool(e.get("is_penalty")),
            "pre": _pre_score(e),
        })
    goals.sort(key=lambda g: (_period_order(g["period"]), g["minute"]))

    def side_of(g, own_goal_to_opponent):
        if g["team_id"] == home_id:
            side = "home"
        elif g["team_id"] == away_id:
            side = "away"
        else:
            return None
        if g["own"] and own_goal_to_opponent:
            side = "away" if side == "home" else "home"
        return side

    def tally(own_goal_to_opponent):
        sides = [side_of(g, own_goal_to_opponent) for g in goals]
        return sides, sides.count("home"), sides.count("away")

    # API'de kendi kalesine golün team_id'si golün YAZILDIĞI (lehine olan) takımdır
    # (örn. Burnley'li Estève'nin kendi kalesine golü Man City'nin team_id'siyle gelir).
    # Varsayılan bu yorum; yalnızca diğer yorum nihai skora uyup bu uymuyorsa ona geçilir.
    sides_a, ha, aa = tally(True)    # team_id = oyuncunun takımı, gol rakibe yazılır
    sides_b, hb, ab = tally(False)   # team_id = golü kazanan takım
    ok_a = (ha, aa) == (final_h, final_a)
    ok_b = (hb, ab) == (final_h, final_a)
    if ok_a and not ok_b:
        sides, ok = sides_a, True
    else:
        sides, ok = sides_b, ok_b

    # Gol olayı yok ve skor 0-0 değilse olay verisi eksik demektir
    available = bool(goals) or (final_h == 0 and final_a == 0)

    row = {
        "match_id": str(match_id),
        "events_available": int(available),
        "ev_goal_check_ok": int(ok),
        "ev_ht_home": None,
        "ev_ht_away": None,
        "first_goal_minute": None,
        "first_goal_side": None,
        "last_goal_minute": None,
        "home_goal_minutes": None,
        "away_goal_minutes": None,
        "home_penalty_goals": None,
        "away_penalty_goals": None,
        "home_own_goals": None,
        "away_own_goals": None,
        "ev_missing_goals": 0 if ok else None,
        "ev_ht_source": None,
        "ev_version": EVENTS_VERSION,
    }
    if not available:
        return row

    gap_first = gap_last = False
    if ok:
        fh = [(g, s) for g, s in zip(goals, sides) if g["period"] == "firsthalf"]
        row["ev_ht_home"] = sum(1 for _, s in fh if s == "home")
        row["ev_ht_away"] = sum(1 for _, s in fh if s == "away")
        row["ev_ht_source"] = "events"
    else:
        # Gol olayı eksik: golden önceki skor alanlarından HT skorunu kurtarmayı dene
        rec = reconstruct_ht(goals, sides, final_h, final_a)
        if rec:
            row["ev_missing_goals"] = rec["missing"]
            gap_first, gap_last = rec["gap_first"], rec["gap_last"]
            if rec["ht"]:
                row["ev_ht_home"], row["ev_ht_away"] = rec["ht"]
                row["ev_ht_source"] = "reconstructed"
        else:
            gap_first = gap_last = True  # belirsiz
            try:
                row["ev_missing_goals"] = max(int(final_h + final_a) - len(goals), 0)
            except (TypeError, ValueError):
                pass

    if goals:
        if not gap_first:
            row["first_goal_minute"] = goals[0]["minute"]
            row["first_goal_side"] = sides[0]
        if not gap_last:
            row["last_goal_minute"] = goals[-1]["minute"]

    for side in ("home", "away"):
        mine = [(g, s) for g, s in zip(goals, sides) if s == side]
        row[f"{side}_goal_minutes"] = ",".join(str(g["minute"]) for g, _ in mine)
        row[f"{side}_penalty_goals"] = sum(1 for g, _ in mine if g["pen"])
        row[f"{side}_own_goals"] = sum(1 for g, _ in mine if g["own"])

    return row


def fill_ht_from_events(matches_df, events_df):
    """ht_home_score / ht_away_score boşsa events özetinden doldurur."""
    if matches_df.empty or events_df.empty:
        return matches_df

    sub = events_df[["match_id", "ev_ht_home", "ev_ht_away"]].copy()
    sub["ev_ht_home"] = pd.to_numeric(sub["ev_ht_home"], errors="coerce")
    sub["ev_ht_away"] = pd.to_numeric(sub["ev_ht_away"], errors="coerce")
    sub["_has_events"] = True

    out = matches_df.merge(sub, on="match_id", how="left")
    for c in ("ht_home_score", "ht_away_score"):
        out[c] = pd.to_numeric(out[c], errors="coerce")

    # Olayı çekilmiş maçlarda HT skoru olaylardan gelir (belirlenemediyse boş kalır,
    # eski/yanlış değer korunmaz). Olayı olmayan maçlarda mevcut değer korunur.
    has = out["_has_events"].fillna(False).astype(bool)
    out["ht_home_score"] = out["ev_ht_home"].where(has, out["ht_home_score"])
    out["ht_away_score"] = out["ev_ht_away"].where(has, out["ht_away_score"])
    return out.drop(columns=["ev_ht_home", "ev_ht_away", "_has_events"])


# -------------------- HEDEF ETİKETLER --------------------

def add_target_columns(df):
    """Model hedefleri: MS, İY, KG, alt/üst (maç ve ilk yarı), ikinci yarı vb."""
    if df.empty:
        return df
    df = df.copy()

    for c in ["home_score", "away_score", "ht_home_score", "ht_away_score"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    h, a = df["home_score"], df["away_score"]
    ok = h.notna() & a.notna()

    # --- Maç sonucu ---
    df["match_result"] = None
    df.loc[ok & (h > a), "match_result"] = "1"
    df.loc[ok & (h == a), "match_result"] = "X"
    df.loc[ok & (h < a), "match_result"] = "2"

    df["total_goals"] = (h + a).where(ok)
    df["goal_diff"] = (h - a).where(ok)

    def flag(cond, mask):
        out = pd.Series(pd.NA, index=df.index, dtype="Int64")
        out[mask] = cond[mask].astype(int)
        return out

    df["btts"] = flag((h > 0) & (a > 0), ok)
    for line in ("0_5", "1_5", "2_5", "3_5", "4_5"):
        df[f"over_{line}"] = flag(df["total_goals"] > float(line.replace("_", ".")), ok)

    df["home_win"] = flag(h > a, ok)
    df["draw"] = flag(h == a, ok)
    df["away_win"] = flag(h < a, ok)
    df["dc_1x"] = flag(h >= a, ok)
    df["dc_x2"] = flag(h <= a, ok)
    df["dc_12"] = flag(h != a, ok)
    df["home_clean_sheet"] = flag(a == 0, ok)
    df["away_clean_sheet"] = flag(h == 0, ok)

    # --- İlk yarı ---
    hh, ha = df["ht_home_score"], df["ht_away_score"]
    ht_ok = hh.notna() & ha.notna()

    df["ht_result"] = None
    df.loc[ht_ok & (hh > ha), "ht_result"] = "1"
    df.loc[ht_ok & (hh == ha), "ht_result"] = "X"
    df.loc[ht_ok & (hh < ha), "ht_result"] = "2"

    df["ht_total_goals"] = (hh + ha).where(ht_ok)
    df["ht_btts"] = flag((hh > 0) & (ha > 0), ht_ok)
    df["ht_over_0_5"] = flag(df["ht_total_goals"] > 0.5, ht_ok)
    df["ht_over_1_5"] = flag(df["ht_total_goals"] > 1.5, ht_ok)

    # --- İkinci yarı ve İY/MS ---
    both = ok & ht_ok
    df["sh_home_goals"] = (h - hh).where(both)
    df["sh_away_goals"] = (a - ha).where(both)
    df["sh_total_goals"] = (df["sh_home_goals"] + df["sh_away_goals"]).where(both)
    df["sh_over_0_5"] = flag(df["sh_total_goals"] > 0.5, both)
    df["sh_over_1_5"] = flag(df["sh_total_goals"] > 1.5, both)

    df["ht_ft"] = None
    df.loc[both, "ht_ft"] = df.loc[both, "ht_result"] + "/" + df.loc[both, "match_result"]

    return df


# -------------------- VERİTABANI --------------------

def table_exists(conn, name):
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone() is not None


def load_table(name):
    if not DB_PATH.exists():
        return pd.DataFrame()
    try:
        with sqlite3.connect(DB_PATH) as conn:
            if not table_exists(conn, name):
                return pd.DataFrame()
            df = pd.read_sql_query(f"SELECT * FROM {name}", conn)
        if "match_id" in df.columns:
            df["match_id"] = df["match_id"].astype(str)
        return df
    except Exception as exc:
        logging.warning("%s tablosu okunamadı: %s", name, exc)
        return pd.DataFrame()


def save_table(name, df):
    with sqlite3.connect(DB_PATH) as conn:
        df.to_sql(name, conn, if_exists="replace", index=False)


def load_fetch_log():
    """(tür, anahtar) -> sayı. 'season_final': geçmiş sezon tamamlandı (tekrar taranmaz);
    'match_stats'/'match_events': o maç için başarısız deneme sayısı."""
    df = load_table("fetch_log")
    if df.empty:
        return {}
    return {(str(r.kind), str(r.key)): int(r.attempts) for r in df.itertuples()}


def save_fetch_log(log):
    rows = [{"kind": k, "key": key, "attempts": v} for (k, key), v in log.items()]
    save_table("fetch_log", pd.DataFrame(rows, columns=["kind", "key", "attempts"]))


def merge_by_match_id(old_df, new_df):
    if old_df.empty:
        return new_df.drop_duplicates(subset=["match_id"], keep="last")
    if new_df.empty:
        return old_df
    merged = pd.concat([old_df, new_df], ignore_index=True)
    return merged.drop_duplicates(subset=["match_id"], keep="last")


# -------------------- DIŞA AKTARIM --------------------

def export_excel_and_csv():
    matches_df = load_table("matches")
    stats_df = load_table("match_stats")
    events_df = load_table("match_events")

    if matches_df.empty:
        logging.warning("Dışa aktarılacak maç yok.")
        return

    matches_df = add_target_columns(matches_df)

    matches_df.to_csv(MATCHES_CSV, index=False, encoding="utf-8-sig")
    stats_df.to_csv(STATS_CSV, index=False, encoding="utf-8-sig")

    is_finished = (
        matches_df["status"].astype(str).str.lower().isin(FINISHED_STATUSES)
        & matches_df["home_score"].notna()
        & matches_df["away_score"].notna()
    )
    upcoming_df = matches_df[~is_finished].copy()

    model_df = matches_df[is_finished].copy()
    if not events_df.empty:
        events_df = events_df.drop(columns=["ev_ht_home", "ev_ht_away"], errors="ignore")
        events_df.to_csv(EVENTS_CSV, index=False, encoding="utf-8-sig")
        model_df = model_df.merge(events_df, on="match_id", how="left")
    if not stats_df.empty:
        model_df = model_df.merge(stats_df, on="match_id", how="left")

    model_df = model_df.sort_values("date").reset_index(drop=True)
    model_df.to_csv(MODEL_CSV, index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(EXCEL_PATH, engine="openpyxl") as writer:
        matches_df.to_excel(writer, sheet_name="matches", index=False)
        upcoming_df.to_excel(writer, sheet_name="upcoming", index=False)
        model_df.to_excel(writer, sheet_name="model_dataset", index=False)

        for ws in writer.book.worksheets:
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
            for col_cells in ws.columns:
                width = max(
                    (len(str(c.value)) for c in col_cells[:100] if c.value is not None),
                    default=0,
                )
                ws.column_dimensions[col_cells[0].column_letter].width = min(max(width + 2, 10), 35)

    logging.info(
        "Dışa aktarım tamam | maç: %d | bitmiş: %d | stats sütunu: %d",
        len(matches_df), int(is_finished.sum()), max(len(stats_df.columns) - 2, 0)
    )


# -------------------- GÜNCELLEME TURU --------------------

def collect_table(table, old_df, ids, fetch_fn, label, log):
    """ids için fetch_fn çağırır, her STATS_CHECKPOINT_EVERY kayıtta DB'ye yazar.
    Defalarca başarısız olan maçlar (log'da MAX_FETCH_ATTEMPTS) atlanır."""
    ids = [m for m in ids if log.get((table, str(m)), 0) < MAX_FETCH_ATTEMPTS]
    logging.info("%s çekilecek maç: %d", label, len(ids))
    state = {"df": old_df, "buffer": []}

    def flush():
        save_fetch_log(log)
        if not state["buffer"]:
            return
        state["df"] = merge_by_match_id(state["df"], pd.DataFrame(state["buffer"]))
        save_table(table, state["df"])
        state["buffer"] = []

    try:
        for i, match_id in enumerate(ids, start=1):
            row = fetch_fn(match_id)
            if row is not None:
                state["buffer"].append(row)
                log.pop((table, str(match_id)), None)
            else:
                log[(table, str(match_id))] = log.get((table, str(match_id)), 0) + 1
            if i % STATS_CHECKPOINT_EVERY == 0:
                flush()
                logging.info("%s ilerlemesi: %d/%d", label, i, len(ids))
    finally:
        flush()  # Ctrl+C olsa bile eldekini kaydet

    return state["df"]


def update_once(refresh_history=False, seasons=None):
    seasons = seasons or SEASONS
    logging.info("Güncelleme başladı: %s", seasons)
    resolve_european_league_ids()

    old_matches = load_table("matches")
    old_stats = load_table("match_stats")

    log = {} if refresh_history else load_fetch_log()

    done_pairs, unfinished_pairs = set(), set()
    if not old_matches.empty and not refresh_history:
        done_pairs = set(zip(old_matches["season"], old_matches["league_id"]))
        fin_mask = (
            old_matches["status"].astype(str).str.lower().isin(FINISHED_STATUSES)
            & pd.to_numeric(old_matches["home_score"], errors="coerce").notna()
            & pd.to_numeric(old_matches["away_score"], errors="coerce").notna()
        )
        unfinished_pairs = set(zip(old_matches.loc[~fin_mask, "season"],
                                   old_matches.loc[~fin_mask, "league_id"]))

    new_rows = []
    for season in seasons:
        for idx, league in enumerate(LEAGUES, start=1):
            final_key = ("season_final", f"{season}|{league['id']}")
            if season != CURRENT_SEASON:
                if log.get(final_key):
                    continue  # geçmiş sezon tamamlandı, tekrar taranmaz
                pair = (season, league["id"])
                if pair in done_pairs and pair not in unfinished_pairs:
                    log[final_key] = 1  # kayıtlı ve hepsi bitmiş
                    continue
                # Kayıtlı ama bitmemiş maç kalmışsa (sezon o zaman sürüyordu) bir kez daha
                # taranır, sonra 'tamamlandı' işaretlenir.

            logging.info("[%d/%d] %s | %s", idx, len(LEAGUES), league["name"], season)
            rows = get_matches(league, season)
            if rows is None:
                continue
            if season != CURRENT_SEASON:
                log[final_key] = 1  # boş dönse bile tekrar tekrar denenmesin
            new_rows.extend(rows)

            finished = sum(
                1 for m in rows
                if str(m["status"]).lower() in FINISHED_STATUSES
                and m["home_score"] is not None
            )
            logging.info("   toplam=%d | bitmiş=%d | diğer=%d",
                         len(rows), finished, len(rows) - finished)

    save_fetch_log(log)

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        combined = merge_by_match_id(old_matches, new_df)
    elif not old_matches.empty:
        logging.warning("API'den yeni maç gelmedi; mevcut veri korunuyor.")
        combined = old_matches
    else:
        logging.warning("Hiç maç alınamadı.")
        return

    combined = add_target_columns(combined)
    save_table("matches", combined)

    # --- İstatistikler: yalnızca bitmiş ve henüz çekilmemiş maçlar ---
    finished_df = combined[
        combined["status"].astype(str).str.lower().isin(FINISHED_STATUSES)
        & combined["home_score"].notna()
        & combined["away_score"].notna()
    ]
    # 1) İstatistikler
    cutoff = datetime.now() - timedelta(days=RECHECK_DAYS)
    fdates = pd.to_datetime(finished_df["date"].astype(str).str.slice(0, 10), errors="coerce")
    recent_ids = set(finished_df.loc[fdates >= cutoff, "match_id"])

    have_stats = set()
    if not old_stats.empty:
        keep = (pd.to_numeric(old_stats["stats_available"], errors="coerce") == 1) \
            | ~old_stats["match_id"].isin(recent_ids)   # yeni biten + boş: tekrar dene
        have_stats = set(old_stats.loc[keep, "match_id"])
    missing_stats = [m for m in finished_df["match_id"] if m not in have_stats]
    stats_df = collect_table(
        "match_stats", old_stats, missing_stats, get_match_stats, "İstatistik", log
    )

    # 2) Olaylar (HT skoru, gol dakikaları)
    old_events = prepare_events_table(load_table("match_events"))
    have_events = set()
    if not old_events.empty:
        keep = (pd.to_numeric(old_events["events_available"], errors="coerce") == 1) \
            | ~old_events["match_id"].isin(recent_ids)
        have_events = set(old_events.loc[keep, "match_id"])
    missing_events = [m for m in finished_df["match_id"] if m not in have_events]
    info = finished_df.set_index("match_id")[
        ["home_team_id", "away_team_id", "home_score", "away_score"]
    ].to_dict("index")
    events_df = collect_table(
        "match_events", old_events, missing_events,
        lambda mid: get_match_events(mid, info[mid]), "Olay", log
    )

    if not events_df.empty:
        bad = events_df[events_df["ev_goal_check_ok"] == 0]
        no_ev = events_df[events_df["events_available"] == 0]
        recon = events_df[events_df["ev_ht_source"] == "reconstructed"]
        logging.info(
            "Olay kontrolü | skorla uyuşmayan: %d (HT kurtarılan: %d) | olayı eksik: %d | toplam: %d",
            len(bad), len(recon), len(no_ev), len(events_df)
        )

    # HT skorlarını olaylardan doldurup etiketleri yeniden hesapla
    combined = fill_ht_from_events(combined, events_df)
    combined = add_target_columns(combined)
    save_table("matches", combined)

    save_fetch_log(log)
    export_excel_and_csv()
    logging.info("Güncelleme tamamlandı. DB: %s", DB_PATH.resolve())


# -------------------- ÇALIŞTIRMA --------------------

def parse_args():
    parser = argparse.ArgumentParser(description="PitchAPI futbol verisi toplayıcı")
    parser.add_argument("--watch", action="store_true",
                        help="Periyodik olarak sürekli güncelle")
    parser.add_argument("--interval", type=float,
                        default=DEFAULT_UPDATE_INTERVAL_HOURS,
                        help="Güncelleme aralığı (saat)")
    parser.add_argument("--test", action="store_true",
                        help="Tek lig/sezon için API cevap yapısını test et")
    parser.add_argument("--refresh-history", action="store_true",
                        help="Geçmiş sezonları da yeniden tara")
    parser.add_argument("--seasons", nargs="+", default=None,
                        help="Sadece bu sezonları çek. Örn: --seasons 2025/2026 2026/2027")
    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )
    args = parse_args()

    if args.test:
        test_api()
        return

    if args.interval <= 0:
        raise ValueError("--interval sıfırdan büyük olmalı.")

    if not args.watch:
        update_once(refresh_history=args.refresh_history, seasons=args.seasons)
        return

    interval_seconds = args.interval * 3600
    logging.info("Sürekli güncelleme aktif. Aralık: %.2f saat.", args.interval)

    try:
        while True:
            started = time.time()
            try:
                update_once(refresh_history=args.refresh_history, seasons=args.seasons)
            except Exception:
                logging.exception("Güncelleme hatası. Sonraki tur devam edecek.")
            sleep_s = max(0, interval_seconds - (time.time() - started))
            logging.info("Sonraki güncelleme %.1f dakika sonra.", sleep_s / 60)
            time.sleep(sleep_s)
    except KeyboardInterrupt:
        logging.info("Kullanıcı programı durdurdu.")


if __name__ == "__main__":
    main()
