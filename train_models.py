"""
train_models.py
features.csv -> her hedef için LightGBM modeli, zaman sıralı değerlendirme, tahmin dosyası.

Colab:
    !pip -q install lightgbm
    import os
    os.environ["OUTPUT_DIR"] = "/content/drive/MyDrive/football/data"
    !python train_models.py

Sadece BUGÜNÜN maçları tahmin edilir (Europe/Istanbul takvimine göre):
    !python train_models.py                       -> bugün
    !python train_models.py --date 2026-11-05     -> belirli gün
    !python train_models.py --all                 -> oynanmamış tüm maçlar (eski davranış)
    !python train_models.py --tz UTC              -> farklı saat dilimi

Çıktılar (OUTPUT_DIR içine):
    model_results.csv        -> hedef başına test metrikleri (model vs. temel çizgi)
    predictions_YYYY-MM-DD.csv -> seçilen günün maçları için olasılıklar
    predictions_predict.csv  -> aynı dosyanın son çalıştırma kopyası
    feature_importance.csv   -> hedef başına en önemli özellikler (gain)

Değerlendirme mantığı:
    - train ile eğit, valid ile erken durdur, test ile ölç (zaman sıralı, sızıntısız).
    - Temel çizgi: train'deki sınıf oranlarını sabit tahmin etmek.
    - skill = 1 - logloss_model / logloss_base  (>0 ise model temel çizgiden iyi)
    - Tahmin için model train+valid+test ile, bulunan iterasyon sayısıyla yeniden eğitilir.
"""

import os
import logging
import argparse
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder

DATA_DIR = Path(os.getenv("OUTPUT_DIR", "data"))

# hedef adı -> sütun adı (features.csv içinde ya da aşağıda türetilir)
TARGETS = {
    "ms_1x2": "match_result",
    "iy_1x2": "ht_result",
    "iy_ms": "ht_ft",
    "kg_var": "btts",
    "iy_kg_var": "ht_btts",
    "ms_ust_0_5": "over_0_5",
    "ms_ust_1_5": "over_1_5",
    "ms_ust_2_5": "over_2_5",
    "ms_ust_3_5": "over_3_5",
    "ms_ust_4_5": "over_4_5",
    "iy_ust_0_5": "ht_over_0_5",
    "iy_ust_1_5": "ht_over_1_5",
    "2y_ust_0_5": "sh_over_0_5",
    "2y_ust_1_5": "sh_over_1_5",
    "toplam_gol": "goals_cat",       # 0,1,2,3,4,5+
    "iy_toplam_gol": "ht_goals_cat",  # 0,1,2,3+
    "korner_ust_8_5": "corners_o8_5",
    "korner_ust_9_5": "corners_o9_5",
    "korner_ust_10_5": "corners_o10_5",
}

# Daha güçlü regülarizasyon (eski: num_leaves=15, min_data_in_leaf=60, lambda_l2=10, lr=0.03)
PARAMS = dict(
    learning_rate=0.02,
    num_leaves=8,
    max_depth=4,
    min_data_in_leaf=120,
    feature_fraction=0.5,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l1=5.0,
    lambda_l2=30.0,
    min_gain_to_split=0.0,
    verbosity=-1,
    seed=42,
)
EARLY_STOP = 150


def load():
    df = pd.read_csv(DATA_DIR / "features.csv", low_memory=False, encoding="utf-8-sig")
    df["match_id"] = df["match_id"].astype(str)

    # türetilmiş hedefler
    if "total_goals" in df.columns:
        df["goals_cat"] = np.minimum(pd.to_numeric(df["total_goals"], errors="coerce"), 5)
    if "ht_total_goals" in df.columns:
        df["ht_goals_cat"] = np.minimum(pd.to_numeric(df["ht_total_goals"], errors="coerce"), 3)

    # korner hedefleri model_dataset.csv'den
    try:
        m = pd.read_csv(DATA_DIR / "model_dataset.csv", low_memory=False,
                        usecols=["match_id", "all_home_corners", "all_away_corners"])
        m["match_id"] = m["match_id"].astype(str)
        m["total_corners"] = (pd.to_numeric(m["all_home_corners"], errors="coerce")
                              + pd.to_numeric(m["all_away_corners"], errors="coerce"))
        for line in (8.5, 9.5, 10.5):
            name = "corners_o" + str(line).replace(".", "_")
            m[name] = np.where(m["total_corners"].notna(), (m["total_corners"] > line).astype(float), np.nan)
        keep = ["match_id", "total_corners"] + [c for c in m.columns if c.startswith("corners_o")]
        df = df.merge(m[keep], on="match_id", how="left")
    except Exception as e:  # noqa: BLE001
        logging.warning("Korner hedefleri eklenemedi: %s", e)
    return df


def local_kickoff_date(df, tz):
    """Maç başlangıcının yerel (tz) takvim günü. kickoff_dt UTC kabul edilir."""
    fallback = df["date"].astype(str).str.slice(0, 10)
    if "kickoff_dt" not in df.columns:
        return fallback
    k = pd.to_datetime(df["kickoff_dt"], errors="coerce")
    if k.notna().sum() == 0:
        return fallback
    k = k.dt.tz_localize("UTC") if k.dt.tz is None else k.dt.tz_convert("UTC")
    return k.dt.tz_convert(tz).dt.strftime("%Y-%m-%d").fillna(fallback)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--date", default=None, help="YYYY-MM-DD (varsayılan: bugün)")
    p.add_argument("--tz", default="Europe/Istanbul", help="gün hesabı için saat dilimi")
    p.add_argument("--all", action="store_true", help="tüm oynanmamış maçları tahmin et")
    p.add_argument("--top-n", type=int, default=120,
                   help="her hedef için en önemli N özelliği tut (0 = seçim yapma)")
    args, _ = p.parse_known_args()  # Colab/Jupyter ek argümanlarına takılmasın
    return args


def time_weights(dt, ref, half_life_days=365.0, floor=0.1):
    """Yakın maçlara daha çok ağırlık: yaş (gün) arttıkça ağırlık yarılanır, en az `floor`.
    dt: kickoff_dt sütunu (metin ya da tarih), ref: referans (genelde en yeni maç)."""
    age = (pd.to_datetime(ref) - pd.to_datetime(dt)).dt.total_seconds().to_numpy() / 86400.0
    return np.clip(0.5 ** (np.clip(age, 0, None) / half_life_days), floor, 1.0)


def get_features(df):
    p = DATA_DIR / "feature_columns.txt"
    feats = p.read_text(encoding="utf-8").split() if p.exists() else list(df.columns)
    feats = [c for c in feats if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    tr = df["split"] == "train"
    feats = [c for c in feats if df.loc[tr, c].notna().any()]
    return feats


def fit(X_tr, y_tr, k, X_va=None, y_va=None, rounds=None):
    p = dict(PARAMS)
    if k == 2:
        p.update(objective="binary", metric="binary_logloss")
    else:
        p.update(objective="multiclass", num_class=k, metric="multi_logloss")
    dtr = lgb.Dataset(X_tr, y_tr)
    if rounds is None:
        dva = lgb.Dataset(X_va, y_va, reference=dtr)
        return lgb.train(p, dtr, num_boost_round=3000, valid_sets=[dva],
                         callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
    return lgb.train(p, dtr, num_boost_round=rounds)


def proba(model, X, k):
    it = model.best_iteration if model.best_iteration and model.best_iteration > 0 else None
    p = model.predict(X, num_iteration=it)
    return np.column_stack([1 - p, p]) if k == 2 else p


def brier(y, P, k):
    onehot = np.eye(k)[y]
    return float(np.mean(np.sum((P - onehot) ** 2, axis=1)))


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    df = load()
    feats = get_features(df)
    logging.info("Özellik sayısı: %d", len(feats))

    ok = df["hist_ok"] == 1
    results, importances = [], []
    pred_rows = df[df["split"] == "predict"]
    if args.all:
        day = "all"
    else:
        day = args.date or datetime.now(ZoneInfo(args.tz)).strftime("%Y-%m-%d")
        pred_rows = pred_rows[local_kickoff_date(pred_rows, args.tz) == day]
    logging.info("Tahmin edilecek maç (%s): %d", day, len(pred_rows))
    if len(pred_rows) == 0:
        logging.warning("Bu gün için oynanmamış maç yok; yalnızca değerlendirme çalışacak. "
                        "(data_collector ve build_features'ı çalıştırdığınızdan emin olun.)")
    pred_out = pred_rows[["match_id", "date", "league", "home_team", "away_team", "hist_ok"]].copy()

    for name, col in TARGETS.items():
        if col not in df.columns:
            logging.warning("%s atlandı: '%s' sütunu yok", name, col)
            continue
        d = df[ok & df[col].notna() & df["split"].isin(["train", "valid", "test"])]
        if d[col].nunique() < 2:
            continue

        enc = LabelEncoder().fit(d[col].astype(str))
        k = len(enc.classes_)
        y = enc.transform(d[col].astype(str))
        sp = d["split"].to_numpy()
        X = d[feats]
        tr, va, te = sp == "train", sp == "valid", sp == "test"
        if tr.sum() < 500 or va.sum() < 100 or te.sum() < 100:
            logging.warning("%s atlandı: yetersiz satır", name)
            continue

        cols = feats
        if args.top_n and args.top_n < len(feats):
            m0 = fit(X[tr][feats], y[tr], k, X[va][feats], y[va])  # yalnızca train+valid kullanılır
            g = pd.Series(m0.feature_importance("gain"), index=feats).sort_values(ascending=False)
            cols = list(g.index[:args.top_n])
        X = d[cols]

        model = fit(X[tr], y[tr], k, X[va], y[va])
        best = model.best_iteration
        P = proba(model, X[te], k)

        prior = np.bincount(y[tr], minlength=k) / tr.sum()
        prior = np.clip(prior, 1e-6, None)
        P0 = np.tile(prior / prior.sum(), (te.sum(), 1))
        labels = list(range(k))
        ll, ll0 = log_loss(y[te], P, labels=labels), log_loss(y[te], P0, labels=labels)

        results.append(dict(
            hedef=name, sinif_sayisi=k, test_maci=int(te.sum()), iterasyon=best, ozellik_sayisi=len(cols),
            logloss_model=round(ll, 4), logloss_temel=round(ll0, 4),
            skill=round(1 - ll / ll0, 4),
            brier_model=round(brier(y[te], P, k), 4), brier_temel=round(brier(y[te], P0, k), 4),
        ))
        logging.info("%-16s k=%d iter=%4d | logloss %.4f vs temel %.4f | skill %+.2f%%",
                     name, k, best, ll, ll0, 100 * (1 - ll / ll0))

        imp = pd.Series(model.feature_importance("gain"), index=cols).sort_values(ascending=False).head(25)
        importances.append(pd.DataFrame({"hedef": name, "ozellik": imp.index, "gain": imp.values}))

        # tahmin için: tüm bitmiş veriyle yeniden eğit
        if len(pred_rows):
            final = fit(X, y, k, rounds=max(50, int(best * 1.15)))
            Pp = proba(final, pred_rows[cols], k)
            for j, cls in enumerate(enc.classes_):
                pred_out[f"{name}__{cls}"] = np.round(Pp[:, j], 4)

    res = pd.DataFrame(results)
    res.to_csv(DATA_DIR / "model_results.csv", index=False, encoding="utf-8-sig")
    if importances:
        pd.concat(importances).to_csv(DATA_DIR / "feature_importance.csv", index=False, encoding="utf-8-sig")
    if len(pred_rows):
        pred_out.to_csv(DATA_DIR / f"predictions_{day}.csv", index=False, encoding="utf-8-sig")
        pred_out.to_csv(DATA_DIR / "predictions_predict.csv", index=False, encoding="utf-8-sig")
        logging.info("Tahminler yazıldı: predictions_%s.csv", day)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
