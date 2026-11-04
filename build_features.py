"""
build_features.py
model_dataset.csv -> features.csv (maç öncesi bilgilerle, veri sızıntısız özellikler)

Kullanım:
    python build_features.py                       -> data/features.csv üretir
    python build_features.py --all-stats           -> tüm istatistikler için özellik üret (çok sütun)
    python build_features.py --test-frac 0.15 --valid-frac 0.15

Colab'da (OUTPUT_DIR, data_collector.py ile aynı klasör):
    os.environ["OUTPUT_DIR"] = "/content/drive/MyDrive/football/data"
    !python build_features.py

Girdi  (data/):  model_dataset.csv (bitmiş maçlar + istatistikler), matches.csv (oynanmamış maçlar için)
Çıktı  (data/):  features.csv, feature_columns.txt

ÖNEMLİ - Sızıntı olmaması için:
- Bir maçın özellikleri YALNIZCA o maçtan ÖNCE oynanmış maçlardan hesaplanır
  (her takım için tüm liglerde/turnuvalarda, tarihe göre sıralı, bir maç kaydırılarak).
- Maçın kendi skoru, istatistiği ve olayları özelliklere girmez. Bunlar yalnızca hedef
  (etiket) olarak ayrı sütunlarda durur.
- Oynanmamış maçlar için (split = predict) özellikler, takımın en son oynadığı maçtan
  sonraki durumdan alınır. Tahmin yapmak için bunları kullanabilirsiniz.

Zaman sıralı bölme: split = train / valid / test (en yeni maçlar test), predict = oynanmamış.
split = history: --start-season öncesi sezonlar (varsayılan 2022/2023 öncesi). Özellik hesabında
(form, Elo vb.) kullanılır ama eğitim/doğrulama/teste girmez; modelde bu satırları kullanmayın.
hist_ok = 1: iki takımın da en az 5 geçmiş maçı var. Model eğitiminde hist_ok == 1 filtresi önerilir.
"""

import os
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd


# -------------------- AYARLAR --------------------

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "data"))

FINISHED = {"finished", "ft", "ended", "completed", "full_time"}
UPCOMING = {"not_started", "scheduled", "upcoming", "notstarted"}

WINDOWS = (5, 10)          # son N maç ortalamaları
VENUE_WINDOW = 5           # evde/deplasmanda son N maç
EWM_HALFLIFE = 4           # üstel ağırlıklı ortalama yarı ömrü (maç)
MIN_PERIODS = 3            # ortalama için en az gözlem
MIN_PERIODS_VENUE = 2
MIN_HISTORY_OK = 10

ELO_START, ELO_K, ELO_HOME_ADV, ELO_SEASON_KEEP = 1500.0, 20.0, 60.0, 0.75

# Varsayılan olarak kullanılan istatistikler ("all" periyodundan). --all-stats ile hepsi.
CORE_STATS = [
    "expected_goals", "expected_goals_on_target", "expected_goals_non_penalty",
    "total_shots", "shotsontarget", "shotsofftarget", "blocked_shots",
    "shots_inside_box", "big_chance", "big_chance_missed_title", "corners",
    "ballpossesion", "passes", "accurate_passes", "touches_opp_box",
    "fouls", "yellow_cards", "red_cards", "offsides", "keeper_saves",
    "matchstats_headers_tackles", "interceptions", "clearances",
    # "147 (70%)" biçimli istatistikler sayı ve yüzde olarak ayrılır (expand_ratio_stats);
    # yalnızca veride bulunanlar kullanılır
    "accurate_passes_pct", "accurate_crosses", "accurate_crosses_pct",
    "long_balls_accurate", "long_balls_accurate_pct",
    "aerials_won", "aerials_won_pct", "dribbles_succeeded", "dribbles_succeeded_pct",
    "duel_won", "duel_won_pct", "ground_duels_won", "ground_duels_won_pct",
    "opposition_half_passes", "own_half_passes",
]
FH_STATS = ["expected_goals", "total_shots", "shotsontarget"]  # ilk yarı istatistikleri

META_COLS = [
    "match_id", "season", "country", "league", "league_id", "date", "time_utc",
    "home_team", "away_team", "home_team_id", "away_team_id", "status",
]
TARGET_COLS = [
    "home_score", "away_score", "ht_home_score", "ht_away_score",
    "match_result", "total_goals", "goal_diff", "btts",
    "over_0_5", "over_1_5", "over_2_5", "over_3_5", "over_4_5",
    "home_win", "draw", "away_win", "dc_1x", "dc_x2", "dc_12",
    "home_clean_sheet", "away_clean_sheet",
    "ht_result", "ht_total_goals", "ht_btts", "ht_over_0_5", "ht_over_1_5",
    "sh_home_goals", "sh_away_goals", "sh_total_goals", "sh_over_0_5", "sh_over_1_5",
    "ht_ft",
]


def num(s):
    return pd.to_numeric(s, errors="coerce")


# -------------------- GİRDİ --------------------

def load_inputs(data_dir):
    fin = pd.read_csv(data_dir / "model_dataset.csv", low_memory=False)
    fin["match_id"] = fin["match_id"].astype(str)

    up = pd.DataFrame()
    mp = data_dir / "matches.csv"
    if mp.exists():
        m = pd.read_csv(mp, low_memory=False)
        m["match_id"] = m["match_id"].astype(str)
        up = m[
            ~m["match_id"].isin(fin["match_id"])
            & m["status"].astype(str).str.lower().isin(UPCOMING)
        ].copy()
    return fin, up


def parse_kickoff(df):
    """
    Başlama zamanını okur. Desteklenen time_utc biçimleri:
      - "19:30" veya "19:30:00"                (date sütunuyla birleştirilir)
      - "2025-09-27T19:30:00Z" / "2025-09-27 19:30"   (ISO tarih-saat)
      - Unix zaman damgası (saniye veya milisaniye)
    Çözümlenemeyen satırlarda yalnızca tarih (00:00) kullanılır.
    Dönüş: (dt, has_time)
    """
    d = df["date"].astype(str).str.slice(0, 10)
    date_dt = pd.to_datetime(d, format="%Y-%m-%d", errors="coerce")
    has_time = pd.Series(False, index=df.index)
    if "time_utc" not in df.columns:
        return date_dt, has_time

    raw = df["time_utc"]
    s = raw.astype(str).str.strip().where(raw.notna(), "")
    parsed = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")

    # a) sadece saat
    m = s.str.extract(r"^(\d{1,2}):(\d{2})(?::\d{2})?$")
    ok = m[0].notna() & date_dt.notna()
    if ok.any():
        parsed.loc[ok] = (
            date_dt[ok]
            + pd.to_timedelta(m.loc[ok, 0].astype(int), unit="h")
            + pd.to_timedelta(m.loc[ok, 1].astype(int), unit="m")
        )

    # b) ISO tarih-saat
    iso = s.str.match(r"^\d{4}-\d{2}-\d{2}[T ]\d{1,2}:\d{2}") & ~ok
    if iso.any():
        try:
            p = pd.to_datetime(s[iso], errors="coerce", utc=True, format="mixed")
        except (TypeError, ValueError):  # eski pandas: eleman eleman çöz
            p = s[iso].apply(lambda x: pd.to_datetime(x, errors="coerce", utc=True))
            p = pd.to_datetime(p, utc=True)
        parsed.loc[iso] = p.dt.tz_localize(None)

    # c) Unix zaman damgası
    numv = pd.to_numeric(s, errors="coerce")
    sec = numv.between(9e8, 4e9) & parsed.isna()
    ms = numv.between(9e11, 4e12) & parsed.isna()
    if sec.any():
        parsed.loc[sec] = pd.to_datetime(numv[sec], unit="s")
    if ms.any():
        parsed.loc[ms] = pd.to_datetime(numv[ms], unit="ms")

    return parsed.fillna(date_dt), parsed.notna()


def add_dt(df):
    df = df.copy()
    df["dt"], df["_has_time"] = parse_kickoff(df)
    return df


def team_key(df, side):
    key = df[f"{side}_team_id"]
    return key.where(key.notna(), df[f"{side}_team"]).astype(str)


_RATIO_RE = r"^\s*(-?\d+(?:\.\d+)?)\s*\(\s*(-?\d+(?:\.\d+)?)\s*%\s*\)\s*$"


def expand_ratio_stats(df):
    """
    "147 (70%)" biçimli metin istatistiklerini ayırır:
      all_home_accurate_passes      -> 147   (sayı)
      all_home_accurate_passes_pct  -> 70    (yüzde)
    Dönüş: (df, ayrıştırılan sütun sayısı)
    """
    prefixes = ("all_home_", "all_away_", "firsthalf_home_", "firsthalf_away_")
    new = {}
    for c in df.columns:
        if not c.startswith(prefixes) or pd.api.types.is_numeric_dtype(df[c]):
            continue
        sample = df[c].dropna().astype(str)
        if sample.empty or sample.str.match(_RATIO_RE).mean() < 0.5:
            continue
        ext = df[c].astype("string").str.extract(_RATIO_RE)
        new[c] = num(ext[0])
        new[c + "_pct"] = num(ext[1])
    for k, v in new.items():
        df[k] = v
    return df, len(new) // 2


def pick_stats(df, all_stats, min_coverage=0.5):
    """Dönüş: (seçilenler, elenenler[(ad, doluluk)]). Doluluk: ham istatistiğin dolu olduğu maç oranı."""
    avail = []
    for c in df.columns:
        if c.startswith("all_home_"):
            s = c[len("all_home_"):]
            if f"all_away_{s}" in df.columns:
                avail.append(s)
    if not all_stats:
        avail = [s for s in CORE_STATS if s in avail]
    chosen, dropped = [], []
    for s in avail:
        cov = num(df[f"all_home_{s}"]).notna().mean()
        (chosen if cov >= min_coverage else dropped).append(s if cov >= min_coverage else (s, cov))
    return chosen, dropped


def log_stat_coverage(df, stats, threshold=0.97, prefix="all"):
    """Ham istatistiklerin lig ve sezon bazında doluluğunu raporlar (eksik veri nedenini gösterir)."""
    keys = [df["league"], df["season"]]
    for s in stats:
        v = num(df[f"{prefix}_home_{s}"]).notna()
        overall = v.mean()
        grp = v.groupby(keys).agg(["mean", "size"])
        grp = grp[grp["size"] >= 50]
        if overall >= threshold and (grp.empty or grp["mean"].min() >= threshold):
            continue
        low = grp[grp["mean"] < 0.9].sort_values("mean")
        txt = ", ".join(f"{lg}/{se}={m:.0%}" for (lg, se), m in low["mean"].head(6).items())
        logging.info("  ham doluluk %s_%-28s genel %.0f%% | düşük lig/sezon: %s",
                     prefix, s, 100 * overall, txt or "-")


# -------------------- TAKIM-MAÇ (UZUN) TABLO --------------------

def build_long(df, stats, fh_stats):
    parts = []
    for side, other in (("home", "away"), ("away", "home")):
        d = pd.DataFrame({
            "match_id": df["match_id"].values,
            "dt": df["dt"].values,
            "season": df["season"].values,
            "league": df["league"].values,
            "team": team_key(df, side).values,
            "is_home": int(side == "home"),
            "gf": num(df[f"{side}_score"]).values,
            "ga": num(df[f"{other}_score"]).values,
            "htf": num(df[f"ht_{side}_score"]).values,
            "hta": num(df[f"ht_{other}_score"]).values,
        })
        for s in stats:
            d[f"{s}_f"] = num(df[f"all_{side}_{s}"]).values
            d[f"{s}_a"] = num(df[f"all_{other}_{s}"]).values
        for s in fh_stats:
            d[f"fh_{s}_f"] = num(df[f"firsthalf_{side}_{s}"]).values
            d[f"fh_{s}_a"] = num(df[f"firsthalf_{other}_{s}"]).values

        gf, ga = d["gf"], d["ga"]
        d["pts"] = np.where(gf > ga, 3.0, np.where(gf == ga, 1.0, 0.0))
        d["win"] = (gf > ga).astype(float)
        d["btts"] = ((gf > 0) & (ga > 0)).astype(float)
        d["o25"] = ((gf + ga) > 2.5).astype(float)
        d["cs"] = (ga == 0).astype(float)
        d["fts"] = (gf == 0).astype(float)
        ht_known = d["htf"].notna() & d["hta"].notna()
        d["ht_o05"] = np.where(ht_known, ((d["htf"] + d["hta"]) > 0).astype(float), np.nan)
        d["sh_gf"] = np.where(ht_known, gf - d["htf"], np.nan)
        d["sh_ga"] = np.where(ht_known, ga - d["hta"], np.nan)

        if "first_goal_side" in df.columns:
            first = df["first_goal_side"].astype(str).values
            tot = (gf + ga).values
            sf = np.where(first == side, 1.0, np.where(first == other, 0.0, np.nan))
            cf = np.where(first == other, 1.0, np.where(first == side, 0.0, np.nan))
            sf = np.where(tot == 0, 0.0, sf)
            cf = np.where(tot == 0, 0.0, cf)
            d["sf"], d["cf"] = sf, cf
        parts.append(d)

    long = pd.concat(parts, ignore_index=True)
    long = long.sort_values(["team", "dt", "match_id", "is_home"]).reset_index(drop=True)
    return long


def metric_lists(long):
    base = [m for m in ["gf", "ga", "pts", "win", "btts", "o25", "cs", "fts",
                        "htf", "hta", "ht_o05", "sh_gf", "sh_ga", "sf", "cf"] if m in long.columns]
    stat_cols = [c for c in long.columns if c.endswith("_f") or c.endswith("_a")]
    stat_cols = [c for c in stat_cols if c not in ("gf", "ga")]
    metrics = base + stat_cols
    venue = [m for m in ["gf", "ga", "pts", "btts", "o25", "expected_goals_f", "expected_goals_a",
                         "total_shots_f", "total_shots_a"] if m in long.columns]
    ewm = [m for m in ["gf", "ga", "pts", "expected_goals_f", "expected_goals_a"] if m in long.columns]
    return metrics, venue, ewm


def games_last_30d_all(long):
    out = np.zeros(len(long))
    days = (long["dt"] - pd.Timestamp("2000-01-01")).dt.total_seconds().to_numpy() / 86400.0
    for _, idx in long.groupby("team").indices.items():
        d = days[idx]
        lo = np.searchsorted(d, d - 30, side="left")
        out[idx] = np.arange(len(d)) - lo
    return out


def compute_team_features(long, metrics, venue_metrics, ewm_metrics):
    """
    Dönüş:
      feats_fin : bitmiş maçlar için (match_id, team, is_home, özellikler) -> maç ÖNCESİ durum
      gen_after / ven_after : her satırdan SONRAKİ durum (oynanmamış maçlar için anlık görüntü)
    """
    g = long.groupby("team", sort=False)

    gen_parts = []
    for w in WINDOWS:
        a = g[metrics].transform(lambda s, w=w: s.rolling(w, min_periods=MIN_PERIODS).mean())
        gen_parts.append(a.add_suffix(f"_r{w}"))
    e = g[ewm_metrics].transform(
        lambda s: s.ewm(halflife=EWM_HALFLIFE, min_periods=MIN_PERIODS).mean()
    )
    gen_parts.append(e.add_suffix("_ew"))
    gen_after = pd.concat(gen_parts, axis=1)

    gv = long.groupby(["team", "is_home"], sort=False)
    ven_after = gv[venue_metrics].transform(
        lambda s: s.rolling(VENUE_WINDOW, min_periods=MIN_PERIODS_VENUE).mean()
    ).add_suffix(f"_v{VENUE_WINDOW}")

    gen_before = gen_after.groupby(long["team"]).shift(1)
    ven_before = ven_after.groupby([long["team"], long["is_home"]]).shift(1)

    extras = pd.DataFrame(index=long.index)
    extras["hist_n"] = g.cumcount()
    extras["season_games"] = long.groupby(["team", "season"]).cumcount()
    extras["rest_days"] = (g["dt"].diff().dt.total_seconds() / 86400.0).clip(lower=0, upper=30)
    extras["games_30d"] = games_last_30d_all(long)

    feats_fin = pd.concat([long[["match_id", "team", "is_home"]], gen_before, ven_before, extras], axis=1)
    return feats_fin, gen_after, ven_after


def upcoming_team_features(up, long, gen_after, ven_after, feat_cols):
    """Oynanmamış maçlar için takımın en son durumunu (son maçtan sonra) kullanır."""
    up = up.reset_index(drop=True)
    rows = []
    for side in ("home", "away"):
        rows.append(pd.DataFrame({
            "match_id": up["match_id"].values,
            "team": team_key(up, side).values,
            "is_home": int(side == "home"),
            "dt": up["dt"].values,
            "season": up["season"].values,
        }))
    uf = pd.concat(rows, ignore_index=True)

    gen_last = pd.concat([long[["team"]], gen_after], axis=1).groupby("team").tail(1)
    ven_last = pd.concat([long[["team", "is_home"]], ven_after], axis=1).groupby(["team", "is_home"]).tail(1)
    uf = uf.merge(gen_last, on="team", how="left")
    uf = uf.merge(ven_last, on=["team", "is_home"], how="left")

    hist = long.groupby("team").size().rename("hist_n").reset_index()
    uf = uf.merge(hist, on="team", how="left")
    uf["hist_n"] = uf["hist_n"].fillna(0)

    sg = long.groupby(["team", "season"]).size().rename("season_games").reset_index()
    uf = uf.merge(sg, on=["team", "season"], how="left")
    uf["season_games"] = uf["season_games"].fillna(0)

    last_dt = long.groupby("team")["dt"].max().rename("_last_dt").reset_index()
    uf = uf.merge(last_dt, on="team", how="left")
    uf["rest_days"] = ((uf["dt"] - uf["_last_dt"]).dt.total_seconds() / 86400.0).clip(lower=0, upper=30)

    base = pd.Timestamp("2000-01-01")
    played_days = {
        t: ((grp["dt"] - base).dt.total_seconds().to_numpy() / 86400.0)
        for t, grp in long.groupby("team")
    }
    fx_days = ((uf["dt"] - base).dt.total_seconds() / 86400.0).to_numpy()
    g30 = np.zeros(len(uf))
    for i, t in enumerate(uf["team"].to_numpy()):
        d = played_days.get(t)
        if d is None:
            continue
        g30[i] = np.searchsorted(d, fx_days[i], side="left") - np.searchsorted(d, fx_days[i] - 30, side="left")
    uf["games_30d"] = g30

    for c in feat_cols:
        if c not in uf.columns:
            uf[c] = np.nan
    return uf[["match_id", "team", "is_home"] + feat_cols]


# -------------------- ELO --------------------

def compute_elo(fin_sorted, up):
    rating, last_season = {}, {}

    def get(team, season):
        r = rating.get(team, ELO_START)
        ls = last_season.get(team)
        if ls is not None and ls != season:
            r = ELO_START + ELO_SEASON_KEEP * (r - ELO_START)
        return r

    pre = {}
    ht_k = team_key(fin_sorted, "home").values
    at_k = team_key(fin_sorted, "away").values
    for mid, th, ta, season, gh, ga in zip(
        fin_sorted["match_id"].values, ht_k, at_k, fin_sorted["season"].values,
        num(fin_sorted["home_score"]).values, num(fin_sorted["away_score"]).values,
    ):
        rh, ra = get(th, season), get(ta, season)
        pre[mid] = (rh, ra)
        exp_h = 1.0 / (1.0 + 10 ** (-((rh + ELO_HOME_ADV) - ra) / 400.0))
        actual = 1.0 if gh > ga else (0.5 if gh == ga else 0.0)
        delta = ELO_K * (np.log(abs(gh - ga) + 1.0) + 1.0) * (actual - exp_h)
        rating[th], rating[ta] = rh + delta, ra - delta
        last_season[th] = last_season[ta] = season

    if len(up):
        for mid, th, ta, season in zip(
            up["match_id"].values, team_key(up, "home").values,
            team_key(up, "away").values, up["season"].values,
        ):
            pre[mid] = (get(th, season), get(ta, season))

    elo = pd.DataFrame.from_dict(pre, orient="index", columns=["elo_h", "elo_a"])
    elo.index.name = "match_id"
    elo["elo_diff"] = elo["elo_h"] - elo["elo_a"]
    elo["elo_exp_home"] = 1.0 / (1.0 + 10 ** (-(elo["elo_diff"] + ELO_HOME_ADV) / 400.0))
    return elo


# -------------------- LİG ÖNCELLERİ --------------------

def league_priors(fin_sorted, up):
    f = fin_sorted[["match_id", "league", "dt"]].copy()
    gh, ga = num(fin_sorted["home_score"]), num(fin_sorted["away_score"])
    f["lg_goals"] = gh + ga
    f["lg_home_win"] = (gh > ga).astype(float)
    f["lg_draw"] = (gh == ga).astype(float)
    f["lg_btts"] = ((gh > 0) & (ga > 0)).astype(float)
    f["lg_o25"] = ((gh + ga) > 2.5).astype(float)
    cols = ["lg_goals", "lg_home_win", "lg_draw", "lg_btts", "lg_o25"]

    pri = f[["match_id"]].copy()
    for c in cols:
        pri[c] = f.groupby("league")[c].transform(lambda s: s.shift(1).expanding(min_periods=30).mean())

    if len(up):
        means = f.groupby("league")[cols].mean()
        u = up[["match_id", "league"]].merge(means, left_on="league", right_index=True, how="left")
        pri = pd.concat([pri, u[["match_id"] + cols]], ignore_index=True)
    return pri.set_index("match_id")


# -------------------- ANA AKIŞ --------------------

def build(data_dir, all_stats=False, test_frac=0.15, valid_frac=0.15, min_coverage=0.5,
          start_season="2022/2023", test_start=None, valid_start=None):
    fin, up = load_inputs(data_dir)
    fin = add_dt(fin)
    fin, n_ratio = expand_ratio_stats(fin)
    if n_ratio:
        logging.info("Metin biçimli ('147 (70%%)') %d sütun sayı ve yüzde olarak ayrıldı.", n_ratio)
    bad = fin["dt"].isna()
    if bad.any():
        logging.warning("Tarihi çözümlenemeyen %d maç atlandı.", int(bad.sum()))
        fin = fin[~bad]
    fin = fin[num(fin["home_score"]).notna() & num(fin["away_score"]).notna()]
    fin = fin.sort_values(["dt", "match_id"]).reset_index(drop=True)
    logging.info("Başlama saati çözümlenemeyen bitmiş maç: %d / %d",
                 int((~fin["_has_time"]).sum()), len(fin))

    if len(up):
        up = add_dt(up)
        up = up[up["dt"].notna()].reset_index(drop=True)

    # takım kimliği tutarlılığı
    ids = pd.concat([
        pd.DataFrame({"id": team_key(fin, "home"), "name": fin["home_team"]}),
        pd.DataFrame({"id": team_key(fin, "away"), "name": fin["away_team"]}),
    ])
    multi = ids.groupby("id")["name"].nunique()
    logging.info("Takım sayısı: %d | birden fazla isimli kimlik: %d", len(multi), int((multi > 1).sum()))

    # Doluluk ve istatistik seçimi yalnızca modelleme penceresine (start_season ve sonrası) bakar
    fin_w = fin[fin["season"].astype(str) >= start_season] if start_season else fin
    logging.info("Modelleme penceresi: %s ve sonrası | pencere maç sayısı: %d | geçmiş (yalnızca özellik hesabı için): %d",
                 start_season or "tüm sezonlar", len(fin_w), len(fin) - len(fin_w))
    stats, dropped = pick_stats(fin_w, all_stats, min_coverage)
    fh_all = [s for s in FH_STATS if f"firsthalf_home_{s}" in fin.columns and f"firsthalf_away_{s}" in fin.columns]
    fh_stats = [s for s in fh_all if num(fin_w[f"firsthalf_home_{s}"]).notna().mean() >= min_coverage]
    logging.info("Kullanılan istatistik sayısı: %d (ilk yarı: %d)", len(stats), len(fh_stats))
    if dropped:
        def sample(s):
            vals = fin_w[f"all_home_{s}"].dropna().astype(str).head(3).tolist()
            return vals
        logging.info("Doluluğu %.0f%% altında olduğu için elenen istatistikler (sayıya çevrilemeyenlerin örnek değerleri): %s",
                     100 * min_coverage, " | ".join(f"{s} ({c:.0%}) örnek={sample(s)}" for s, c in dropped))
    log_stat_coverage(fin_w, stats)
    log_stat_coverage(fin_w, fh_stats, prefix="firsthalf")

    long = build_long(fin, stats, fh_stats)
    metrics, venue_m, ewm_m = metric_lists(long)
    feats_fin, gen_after, ven_after = compute_team_features(long, metrics, venue_m, ewm_m)
    feat_cols = [c for c in feats_fin.columns if c not in ("match_id", "team", "is_home")]

    parts = [feats_fin]
    if len(up):
        parts.append(upcoming_team_features(up, long, gen_after, ven_after, feat_cols))
    lf = pd.concat(parts, ignore_index=True)

    home = lf[lf["is_home"] == 1].drop(columns=["team", "is_home"]).set_index("match_id").add_prefix("h_")
    away = lf[lf["is_home"] == 0].drop(columns=["team", "is_home"]).set_index("match_id").add_prefix("a_")
    wide = home.join(away, how="outer")

    # maç bazlı tablo
    meta_t = [c for c in META_COLS + TARGET_COLS if c in fin.columns]
    base_f = fin[meta_t + ["dt", "_has_time"]].copy()
    base_f["is_upcoming"] = 0
    if len(up):
        meta_u = [c for c in META_COLS if c in up.columns]
        base_u = up[meta_u + ["dt", "_has_time"]].copy()
        base_u["is_upcoming"] = 1
        base = pd.concat([base_f, base_u], ignore_index=True)
    else:
        base = base_f

    out = base.merge(wide, left_on="match_id", right_index=True, how="left")
    elo = compute_elo(fin, up)
    out = out.merge(elo, left_on="match_id", right_index=True, how="left")
    out = out.merge(league_priors(fin, up), left_on="match_id", right_index=True, how="left")

    # takım farkları ve karşılaşma özellikleri
    for m in ["pts", "gf", "ga", "expected_goals_f", "expected_goals_a"]:
        for suf in ("r5", "r10"):
            h, a = f"h_{m}_{suf}", f"a_{m}_{suf}"
            if h in out.columns and a in out.columns:
                out[f"diff_{m}_{suf}"] = out[h] - out[a]
    if "h_expected_goals_f_r10" in out.columns:
        out["mu_h_xg"] = (out["h_expected_goals_f_r10"] + out["a_expected_goals_a_r10"]) / 2
        out["mu_a_xg"] = (out["a_expected_goals_f_r10"] + out["h_expected_goals_a_r10"]) / 2
        out["mu_total_xg"] = out["mu_h_xg"] + out["mu_a_xg"]
    out["mu_h_goals"] = (out["h_gf_r10"] + out["a_ga_r10"]) / 2
    out["mu_a_goals"] = (out["a_gf_r10"] + out["h_ga_r10"]) / 2
    out["mu_total_goals"] = out["mu_h_goals"] + out["mu_a_goals"]

    # takvim
    out["month"] = out["dt"].dt.month
    out["dayofweek"] = out["dt"].dt.dayofweek
    out["hour_utc"] = np.where(out["_has_time"], out["dt"].dt.hour, np.nan)
    out["is_european"] = out["country"].astype(str).eq("Europe").astype(int)

    out["hist_ok"] = ((out["h_hist_n"] >= MIN_HISTORY_OK) & (out["a_hist_n"] >= MIN_HISTORY_OK)).astype(int)

    # zaman sıralı bölme
    out = out.sort_values(["dt", "match_id"]).reset_index(drop=True)
    is_up = out["is_upcoming"] == 1
    if start_season:
        is_hist = (~is_up) & (out["season"].astype(str) < start_season)
    else:
        is_hist = pd.Series(False, index=out.index)
    fin_dt = out.loc[~is_up & ~is_hist, "dt"].to_numpy()   # bölme yalnızca modelleme penceresinden
    n = len(fin_dt)
    c_test = fin_dt[min(int(n * (1 - test_frac)), n - 1)]
    c_valid = fin_dt[min(int(n * (1 - test_frac - valid_frac)), n - 1)]
    if test_start:
        c_test = pd.Timestamp(test_start)
    if valid_start:
        c_valid = pd.Timestamp(valid_start)
    window_split = np.where(out["dt"] >= c_test, "test", np.where(out["dt"] >= c_valid, "valid", "train"))
    out["split"] = np.where(is_up, "predict", np.where(is_hist, "history", window_split))

    out["date"] = out["date"].astype(str).str.slice(0, 10)
    out = out.drop(columns=["_has_time", "is_upcoming"])
    out = out.rename(columns={"dt": "kickoff_dt"})

    non_feat = set(meta_t) | {"kickoff_dt", "split", "hist_ok"}
    feature_columns = [c for c in out.columns if c not in non_feat]
    return out, feature_columns


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    p = argparse.ArgumentParser(description="Maç öncesi özellik üretici")
    p.add_argument("--data-dir", default=str(OUTPUT_DIR))
    p.add_argument("--all-stats", action="store_true", help="Tüm istatistikler için özellik üret")
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--valid-frac", type=float, default=0.15)
    p.add_argument("--min-coverage", type=float, default=0.5,
                   help="Ham istatistik doluluğu bu oranın altındaysa özellik üretme (örn. 0.95)")
    p.add_argument("--start-season", default="2022/2023",
                   help="Eğitim/doğrulama/test bu sezondan başlar; öncesi 'history' olarak yalnızca "
                        "özellik hesabında kullanılır. Hepsini kullanmak için boş bırakın: --start-season ''")
    p.add_argument("--test-start", default=None, help="YYYY-MM-DD, test döneminin başı (sabitlemek için)")
    p.add_argument("--valid-start", default=None, help="YYYY-MM-DD, valid döneminin başı")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    out, feature_columns = build(data_dir, args.all_stats, args.test_frac, args.valid_frac,
                                 args.min_coverage, args.start_season or None,
                                 args.test_start, args.valid_start)

    out.to_csv(data_dir / "features.csv", index=False, encoding="utf-8-sig")
    (data_dir / "feature_columns.txt").write_text("\n".join(feature_columns), encoding="utf-8")

    logging.info("Yazıldı: %s | satır: %d | özellik sayısı: %d",
                 data_dir / "features.csv", len(out), len(feature_columns))
    chk = out[out["split"].isin(["train", "valid", "test"]) & (out["hist_ok"] == 1)]
    na = chk[feature_columns].isna().mean().sort_values(ascending=False)
    logging.info("Boş değer oranı (bitmiş, hist_ok=1) | ortalama: %.1f%% | en boş 8: %s",
                 100 * na.mean(), ", ".join(f"{c}={v:.0%}" for c, v in na.head(8).items()))
    for sp, grp in out.groupby("split"):
        logging.info("  %-8s %6d maç | %s -> %s | hist_ok: %.0f%%", sp, len(grp),
                     grp["kickoff_dt"].min().date(), grp["kickoff_dt"].max().date(),
                     100 * grp["hist_ok"].mean())


if __name__ == "__main__":
    main()
