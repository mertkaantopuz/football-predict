"""
poisson_models.py
Ev/deplasman beklenen golünü (lambda) LightGBM-Poisson ile tahmin eder, skor matrisinden
(Dixon-Coles düzeltmeli) tüm gol marketlerini türetir ve v1 sonuçlarıyla karşılaştırır.

Kullanım (train_models.py ile aynı klasörde):
    python poisson_models.py                  -> değerlendirme + bugünün maçları
    python poisson_models.py --date 2026-11-05
    python poisson_models.py --all

Çıktılar (data/ içine):
    poisson_results.csv        hedef başına test metrikleri
    karsilastirma.csv          v1 (model_results.csv) ile skill karşılaştırması
    poisson_predictions_<gün>.csv
"""
import logging
from datetime import datetime
from turtle import stamp
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import poisson
from sklearn.metrics import log_loss

from train_models import (DATA_DIR, PARAMS, load, get_features, parse_args,
                          local_kickoff_date, brier, time_weights)

# ---- features.csv'deki gol sütunları: aşağıdaki adaylardan ilk bulunan kullanılır ----
FT_CANDIDATES = [("home_goals", "away_goals"), ("home_score", "away_score"),
                 ("ft_home_goals", "ft_away_goals"), ("home_ft_goals", "away_ft_goals"),
                 ("h_goals", "a_goals"), ("home_team_goals", "away_team_goals")]
HT_CANDIDATES = [("ht_home_goals", "ht_away_goals"), ("home_ht_goals", "away_ht_goals"),
                 ("ht_home_score", "ht_away_score"), ("fh_home_goals", "fh_away_goals"),
                 ("ht_h_goals", "ht_a_goals")]
SIGNIFICANT_DIFF = 0.004   # skill farkı bunun altındaysa gürültü say
# ---------------------------------------------------------------------------------------


def find_cols(df, candidates):
    for h, a in candidates:
        if h in df.columns and a in df.columns:
            return h, a
    return None


def fit_pois(X_tr, y_tr, X_va=None, y_va=None, rounds=None, w_tr=None):
    p = dict(PARAMS, objective="poisson", metric="poisson")
    dtr = lgb.Dataset(X_tr, y_tr, weight=w_tr)
    if rounds is None:
        dva = lgb.Dataset(X_va, y_va, reference=dtr)
        return lgb.train(p, dtr, num_boost_round=3000, valid_sets=[dva],
                         callbacks=[lgb.early_stopping(100, verbose=False)])
    return lgb.train(p, dtr, num_boost_round=rounds)


def score_probs(lh, la, rho, G):
    """(N,G,G) skor olasılıkları; [n, ev_gol, dep_gol]. Dixon-Coles düşük skor düzeltmesi."""
    g = np.arange(G)
    P = (poisson.pmf(g[None, :], lh[:, None])[:, :, None]
         * poisson.pmf(g[None, :], la[:, None])[:, None, :])
    P[:, 0, 0] *= 1 - lh * la * rho
    P[:, 0, 1] *= 1 + lh * rho
    P[:, 1, 0] *= 1 + la * rho
    P[:, 1, 1] *= 1 - rho
    P = np.clip(P, 1e-12, None)
    return P / P.sum(axis=(1, 2), keepdims=True)


def fit_rho(lh, la, hg, ag, G):
    best, best_ll = 0.0, -np.inf
    hi, ai = np.minimum(hg, G - 1).astype(int), np.minimum(ag, G - 1).astype(int)
    for rho in np.linspace(-0.15, 0.10, 26):
        P = score_probs(lh, la, rho, G)
        ll = np.log(P[np.arange(len(hg)), hi, ai]).sum()
        if ll > best_ll:
            best, best_ll = rho, ll
    return float(best)


def market_defs(block):
    if block == "ft":
        return ([("ms_1x2", "1x2", None), ("kg_var", "btts", None)]
                + [(f"ms_ust_{l}".replace(".", "_"), "over", l) for l in (0.5, 1.5, 2.5, 3.5, 4.5)]
                + [("toplam_gol", "cat", 5)])
    return [("iy_1x2", "1x2", None), ("iy_kg_var", "btts", None),
            ("iy_ust_0_5", "over", 0.5), ("iy_ust_1_5", "over", 1.5), ("iy_toplam_gol", "cat", 3)]


def market_probs(P, kind, par):
    G = P.shape[1]
    i, j = np.arange(G)[:, None], np.arange(G)[None, :]
    tot = i + j
    if kind == "1x2":
        masks = [i > j, i == j, i < j]
    elif kind == "btts":
        both = (i >= 1) & (j >= 1)
        masks = [~both, both]
    elif kind == "over":
        masks = [tot <= par, tot > par]
    else:  # cat
        masks = [tot == c for c in range(par)] + [tot >= par]
    out = np.column_stack([P[:, m].sum(axis=1) for m in masks])
    return out / out.sum(axis=1, keepdims=True)


def market_truth(hg, ag, kind, par):
    if kind == "1x2":
        return np.where(hg > ag, 0, np.where(hg == ag, 1, 2))
    if kind == "btts":
        return ((hg >= 1) & (ag >= 1)).astype(int)
    if kind == "over":
        return ((hg + ag) > par).astype(int)
    return np.minimum(hg + ag, par).astype(int)


def class_names(kind, par):
    if kind == "1x2":
        return ["1", "X", "2"]
    if kind in ("btts", "over"):
        return ["0", "1"]
    return [str(c) for c in range(par)] + [f"{par}+"]


def run_block(block, df, feats, ok, cols, pred_rows, pred_out):
    hc, ac = cols
    G = 11 if block == "ft" else 8
    d = df[ok & df[hc].notna() & df[ac].notna() & df["split"].isin(["train", "valid", "test"])]
    sp = d["split"].to_numpy()
    tr, va, te = sp == "train", sp == "valid", sp == "test"
    if tr.sum() < 500 or va.sum() < 100 or te.sum() < 100:
        logging.warning("%s bloğu atlandı: yetersiz satır", block)
        return []
    X = d[feats]
    hg, ag = d[hc].to_numpy(float), d[ac].to_numpy(float)

    kt = d["kickoff_dt"]
    w_tr = time_weights(kt[tr], kt[tr].max())
    mh = fit_pois(X[tr], hg[tr], X[va], hg[va], w_tr=w_tr)
    ma = fit_pois(X[tr], ag[tr], X[va], ag[va], w_tr=w_tr)
    bh, ba = max(1, mh.best_iteration), max(1, ma.best_iteration)
    lh_va, la_va = mh.predict(X[va], num_iteration=bh), ma.predict(X[va], num_iteration=ba)
    rho = fit_rho(lh_va, la_va, hg[va], ag[va], G)
    lh, la = mh.predict(X[te], num_iteration=bh), ma.predict(X[te], num_iteration=ba)
    logging.info("[%s] iter ev=%d dep=%d | rho=%.3f | ort. lambda ev %.3f (gerçek %.3f), dep %.3f (gerçek %.3f)",
                 block, bh, ba, rho, lh.mean(), hg[te].mean(), la.mean(), ag[te].mean())
    P = score_probs(lh, la, rho, G)

    # tahmin için: tüm bitmiş veriyle yeniden eğit
    Pp = None
    if len(pred_rows):
        w_all = time_weights(kt, kt.max())
        fh = fit_pois(X, hg, rounds=max(50, int(bh * 1.15)), w_tr=w_all)
        fa = fit_pois(X, ag, rounds=max(50, int(ba * 1.15)), w_tr=w_all)
        lhp, lap = fh.predict(pred_rows[feats]), fa.predict(pred_rows[feats])
        Pp = score_probs(lhp, lap, rho, G)
        pred_out[f"{block}_lambda_ev"] = np.round(lhp, 3)
        pred_out[f"{block}_lambda_dep"] = np.round(lap, 3)

    results = []
    for name, kind, par in market_defs(block):
        y = market_truth(hg, ag, kind, par)
        Pm = market_probs(P, kind, par)
        k = Pm.shape[1]
        prior = np.clip(np.bincount(y[tr], minlength=k) / tr.sum(), 1e-6, None)
        P0 = np.tile(prior / prior.sum(), (te.sum(), 1))
        labels = list(range(k))
        ll = log_loss(y[te], Pm, labels=labels)
        ll0 = log_loss(y[te], P0, labels=labels)
        results.append(dict(hedef=name, sinif_sayisi=k, test_maci=int(te.sum()),
                            iterasyon_ev=bh, iterasyon_dep=ba, rho=round(rho, 3),
                            valid_ev=round(mh.best_score["valid_0"]["poisson"], 4),
                            valid_dep=round(ma.best_score["valid_0"]["poisson"], 4),
                            logloss_model=round(ll, 4), logloss_temel=round(ll0, 4),
                            skill=round(1 - ll / ll0, 4),
                            brier_model=round(brier(y[te], Pm, k), 4),
                            brier_temel=round(brier(y[te], P0, k), 4)))
        logging.info("%-14s k=%d | logloss %.4f vs temel %.4f | skill %+.2f%%",
                     name, k, ll, ll0, 100 * (1 - ll / ll0))
        if Pp is not None:
            Pmp = market_probs(Pp, kind, par)
            for j, cls in enumerate(class_names(kind, par)):
                pred_out[f"{name}__{cls}"] = np.round(Pmp[:, j], 4)
    return results


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    df = load()
    feats = get_features(df)
    ok = df["hist_ok"] == 1

    ft = find_cols(df, FT_CANDIDATES)
    if ft is None:
        hint = [c for c in df.columns if any(s in c.lower() for s in ("goal", "score"))]
        raise SystemExit("Maç sonu gol sütunları bulunamadı. FT_CANDIDATES listesine ekle. "
                         f"'goal/score' içeren sütunlar: {hint}")
    ht = find_cols(df, HT_CANDIDATES)
    if ht is None:
        logging.warning("İlk yarı gol sütunları bulunamadı, iy_* marketleri atlanacak.")
    logging.info("FT sütunları: %s | HT sütunları: %s", ft, ht)

    pred_rows = df[df["split"] == "predict"]
    if args.all:
        day = "all"
    else:
        day = args.date or datetime.now(ZoneInfo(args.tz)).strftime("%Y-%m-%d")
        pred_rows = pred_rows[local_kickoff_date(pred_rows, args.tz) == day]
    logging.info("Tahmin edilecek maç (%s): %d", day, len(pred_rows))
    pred_out = pred_rows[["match_id", "date", "league", "home_team", "away_team", "hist_ok"]].copy()

    results = run_block("ft", df, feats, ok, ft, pred_rows, pred_out)
    if ht is not None:
        results += run_block("ht", df, feats, ok, ht, pred_rows, pred_out)

    res = pd.DataFrame(results)
    res.to_csv(DATA_DIR / "poisson_results.csv", index=False, encoding="utf-8-sig")

    v1_path = DATA_DIR / "model_results.csv"
    if v1_path.exists() and len(res):
        v1 = pd.read_csv(v1_path, encoding="utf-8-sig")[["hedef", "skill"]].rename(columns={"skill": "skill_v1"})
        cmp_ = res[["hedef", "skill"]].rename(columns={"skill": "skill_poisson"}).merge(v1, on="hedef", how="left")
        cmp_["fark"] = (cmp_["skill_poisson"] - cmp_["skill_v1"]).round(4)
        cmp_["sonuc"] = np.where(cmp_["fark"].abs() < SIGNIFICANT_DIFF, "fark yok (gürültü)",
                                 np.where(cmp_["fark"] > 0, "Poisson daha iyi", "v1 daha iyi"))
        cmp_.to_csv(DATA_DIR / "karsilastirma.csv", index=False, encoding="utf-8-sig")
        print("\n", cmp_.to_string(index=False))
    else:
        print(res.to_string(index=False))

    if len(pred_rows):
        stamp = datetime.now(ZoneInfo(args.tz)).strftime("%H%M")
        pred_out.to_csv(DATA_DIR / f"poisson_predictions_{day}_{stamp}.csv", index=False, encoding="utf-8-sig")
        pred_out.to_csv(DATA_DIR / f"poisson_predictions_{day}.csv", index=False, encoding="utf-8-sig")
        logging.info("Tahminler yazıldı: poisson_predictions_%s_%s.csv", day, stamp)

if __name__ == "__main__":
    main()
