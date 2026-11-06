"""
poisson_model.py
Gol tabanlı (Poisson) model + walk-forward değerlendirme + LightGBM sınıflandırıcı ile harman.

Mantık:
    6 adet Poisson regresyonu (LightGBM): ev/deplasman golü için maç sonu, ilk yarı, ikinci yarı.
    + toplam korner için 1 adet. Beklenen gollerden Dixon-Coles düzeltmeli skor matrisi kurulur;
    tüm hedefler (1X2, alt/üst, KG, toplam gol, İY, İY/MS, 2Y, korner) bu matrislerden TUTARLI şekilde türetilir.

Kullanım (football klasöründe, train_models.py ile aynı yerde):
    python poisson_model.py eval                 # walk-forward: sınıflandırıcı vs Poisson vs harman
    python poisson_model.py eval --folds 3 --targets ms_1x2,ms_ust_2_5
    python poisson_model.py predict              # bugünün maçları (method=auto: eval'de en iyi çıkan)
    python poisson_model.py predict --method poisson --date 2026-11-05

Çıktılar (OUTPUT_DIR içine):
    walkforward_summary.csv   hedef başına: skill (sınıflandırıcı / poisson / harman), fark ve %95 aralığı
    walkforward_folds.csv     dönem (fold) bazında skill
    walkforward_methods.csv   yöntemlerin ortalama skill'i (predict --method auto bunu okur)
    predictions_final_YYYY-MM-DD.csv
"""

import argparse
import logging
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson

import train_models as tm
from train_models import DATA_DIR, EARLY_STOP, PARAMS, TARGETS, fit, local_kickoff_date, proba

G = 11  # skor matrisi 0..10 gol
GOAL_COLS = {"hg": "home_score", "ag": "away_score",
             "hhg": "ht_home_score", "hag": "ht_away_score",
             "shg": "sh_home_goals", "sag": "sh_away_goals"}
PAIRS = {"ft": ("hg", "ag"), "ht": ("hhg", "hag"), "sh": ("shg", "sag")}
REG_PARAMS = dict(PARAMS, objective="poisson", metric="poisson")
VALID_SHARE = 0.15
I_IDX, J_IDX = np.indices((G, G))
SUM_IDX, DIF_IDX = I_IDX + J_IDX, I_IDX - J_IDX


# ------------------------------------------------------------------ Poisson parçaları
def fit_reg(Xtr, ytr, Xva=None, yva=None, rounds=None):
    dtr = lgb.Dataset(Xtr, ytr)
    if rounds is None:
        dva = lgb.Dataset(Xva, yva, reference=dtr)
        return lgb.train(REG_PARAMS, dtr, num_boost_round=3000, valid_sets=[dva],
                         callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
    return lgb.train(REG_PARAMS, dtr, num_boost_round=rounds)


def mu_of(model, X, hi=8.0):
    """Beklenen değer; goller için en çok 8, korner için hi=40 verilir."""
    it = model.best_iteration if model.best_iteration and model.best_iteration > 0 else None
    return np.clip(model.predict(X, num_iteration=it), 0.03, hi)


def _tau(mh, ma, x, y, rho):
    t = np.ones(len(x))
    m = (x == 0) & (y == 0); t[m] = 1 - mh[m] * ma[m] * rho
    m = (x == 0) & (y == 1); t[m] = 1 + mh[m] * rho
    m = (x == 1) & (y == 0); t[m] = 1 + ma[m] * rho
    m = (x == 1) & (y == 1); t[m] = 1 - rho
    return t


def fit_rho(mh, ma, x, y):
    """Dixon-Coles düşük skor düzeltmesi: log-olabilirliği en yüksek rho (ızgara araması)."""
    base = poisson.logpmf(x, mh) + poisson.logpmf(y, ma)
    best, best_ll = 0.0, -np.inf
    for rho in np.linspace(-0.25, 0.25, 101):
        t = _tau(mh, ma, x, y, rho)
        if (t <= 0).any():
            continue
        ll = float(np.sum(base + np.log(t)))
        if ll > best_ll:
            best, best_ll = rho, ll
    return best


def score_matrix(mh, ma, rho):
    k = np.arange(G)
    M = poisson.pmf(k[None, :], mh[:, None])[:, :, None] * poisson.pmf(k[None, :], ma[:, None])[:, None, :]
    M[:, 0, 0] *= 1 - mh * ma * rho
    M[:, 0, 1] *= 1 + mh * rho
    M[:, 1, 0] *= 1 + ma * rho
    M[:, 1, 1] *= 1 - rho
    M = np.clip(M, 0, None)
    return M / M.sum(axis=(1, 2), keepdims=True)


def dists(M):
    T = np.stack([M[:, SUM_IDX == k].sum(1) for k in range(2 * G - 1)], axis=1)   # toplam gol
    D = np.stack([M[:, DIF_IDX == d].sum(1) for d in range(-(G - 1), G)], axis=1)  # ev - dep farkı
    return T, D


def over(T, line):
    return T[:, int(np.floor(line)) + 1:].sum(1)


def three(D):
    return {"1": D[:, G:].sum(1), "X": D[:, G - 1], "2": D[:, :G - 1].sum(1)}


def binary(p):
    return {"0.0": 1 - p, "1.0": p}


def btts(M):
    return 1 - M[:, 0, :].sum(1) - M[:, :, 0].sum(1) + M[:, 0, 0]


def htft(Dh, Ds):
    """İY farkı + 2Y farkı (bağımsız) -> İY/MS 9 sınıf."""
    sg = lambda d: "1" if d > 0 else ("X" if d == 0 else "2")
    out = {f"{a}/{b}": np.zeros(len(Dh)) for a in "1X2" for b in "1X2"}
    for i1 in range(2 * G - 1):
        d1 = i1 - (G - 1)
        for i2 in range(2 * G - 1):
            d2 = i2 - (G - 1)
            out[f"{sg(d1)}/{sg(d1 + d2)}"] += Dh[:, i1] * Ds[:, i2]
    return out


def derive(M, mu_tc=None, nb_r=None):
    """Skor matrislerinden hedef -> {sınıf: olasılık vektörü} sözlüğü."""
    out = {}
    key = lambda l: str(l).replace(".", "_")
    if "ft" in M:
        T, D = dists(M["ft"])
        out["ms_1x2"] = three(D)
        out["kg_var"] = binary(btts(M["ft"]))
        for l in (0.5, 1.5, 2.5, 3.5, 4.5):
            out[f"ms_ust_{key(l)}"] = binary(over(T, l))
        out["toplam_gol"] = {**{f"{k}.0": T[:, k] for k in range(5)}, "5.0": T[:, 5:].sum(1)}
    if "ht" in M:
        T, D = dists(M["ht"])
        out["iy_1x2"] = three(D)
        out["iy_kg_var"] = binary(btts(M["ht"]))
        for l in (0.5, 1.5):
            out[f"iy_ust_{key(l)}"] = binary(over(T, l))
        out["iy_toplam_gol"] = {"0.0": T[:, 0], "1.0": T[:, 1], "2.0": T[:, 2], "3.0": T[:, 3:].sum(1)}
    if "sh" in M:
        T, _ = dists(M["sh"])
        for l in (0.5, 1.5):
            out[f"2y_ust_{key(l)}"] = binary(over(T, l))
    if "ht" in M and "sh" in M:
        out["iy_ms"] = htft(dists(M["ht"])[1], dists(M["sh"])[1])
    if mu_tc is not None:
        for l in (8.5, 9.5, 10.5):
            if nb_r is not None and nb_r < 1e5:
                cdf = nbinom.cdf(int(l), nb_r, nb_r / (nb_r + mu_tc))
            else:
                cdf = poisson.cdf(int(l), mu_tc)
            out[f"korner_ust_{key(l)}"] = binary(1 - cdf)
    return out


class PoissonSystem:
    def __init__(self, feats):
        self.feats, self.models, self.best, self.rho, self.nb_r, self.cols = feats, {}, {}, {}, None, {}

    def fit(self, tr, va):
        f = self.feats
        self.cols = dict(GOAL_COLS)
        if "total_corners" in tr.columns and tr["total_corners"].notna().sum() > 500:
            self.cols["tc"] = "total_corners"
        for key, col in self.cols.items():
            a, b = tr[tr[col].notna()], va[va[col].notna()]
            if len(a) < 300 or len(b) < 50:
                continue
            m = fit_reg(a[f], a[col], b[f], b[col])
            self.models[key], self.best[key] = m, max(int(m.best_iteration or 0), 50)
        for pair, (kh, ka) in PAIRS.items():
            if kh in self.models and ka in self.models:
                ch, ca = GOAL_COLS[kh], GOAL_COLS[ka]
                b = va[va[ch].notna() & va[ca].notna()]
                if len(b) >= 100:
                    self.rho[pair] = fit_rho(mu_of(self.models[kh], b[f]), mu_of(self.models[ka], b[f]),
                                             b[ch].to_numpy(int), b[ca].to_numpy(int))
        if "tc" in self.models:
            b = va[va["total_corners"].notna()]
            mu, y = mu_of(self.models["tc"], b[f], 40.0), b["total_corners"].to_numpy(float)
            den = float(np.sum((y - mu) ** 2 - mu))
            self.nb_r = float(np.clip(np.sum(mu ** 2) / den, 1.0, 1e6)) if den > 0 else 1e6
        return self

    def refit(self, full):
        for key, col in self.cols.items():
            if key in self.models:
                a = full[full[col].notna()]
                self.models[key] = fit_reg(a[self.feats], a[col], rounds=int(self.best[key] * 1.15) + 1)
        return self

    def predict(self, X):
        mus = {k: mu_of(m, X[self.feats], 40.0 if k == "tc" else 8.0) for k, m in self.models.items()}
        M = {p: score_matrix(mus[kh], mus[ka], self.rho.get(p, 0.0))
             for p, (kh, ka) in PAIRS.items() if kh in mus and ka in mus}
        self.last_mu = mus
        return derive(M, mus.get("tc"), self.nb_r)


# ------------------------------------------------------------------ sınıflandırıcı + yardımcılar
def to_matrix(dct, classes, n):
    P = np.column_stack([dct.get(c, np.zeros(n)) for c in classes])
    P = np.clip(P, 1e-6, None)
    return P / P.sum(axis=1, keepdims=True)


def clf_predict(tr, va, Xte, col, classes, feats, refit_df=None):
    idx, k = {c: i for i, c in enumerate(classes)}, len(classes)
    a, b = tr[tr[col].notna()], va[va[col].notna()]
    if len(a) < 300 or len(b) < 50:
        return None
    m = fit(a[feats], a[col].astype(str).map(idx).to_numpy(), k,
            b[feats], b[col].astype(str).map(idx).to_numpy())
    if refit_df is None:
        return proba(m, Xte[feats], k)
    f = refit_df[refit_df[col].notna()]
    m2 = fit(f[feats], f[col].astype(str).map(idx).to_numpy(), k,
             rounds=max(50, int((m.best_iteration or 50) * 1.15)))
    return proba(m2, Xte[feats], k)


def select_features(D, feats, top_n):
    """Ev/deplasman golü Poisson modellerinin gain'lerine göre en iyi N özellik (yalnızca verilen veriyle)."""
    if not top_n or top_n >= len(feats):
        return feats
    cut = int(len(D) * (1 - VALID_SHARE))
    tr, va = D.iloc[:cut], D.iloc[cut:]
    g = pd.Series(0.0, index=feats)
    for col in ("home_score", "away_score"):
        m = fit_reg(tr[feats], tr[col], va[feats], va[col])
        imp = pd.Series(m.feature_importance("gain"), index=feats)
        g += imp / max(imp.sum(), 1e-9)
    return list(g.sort_values(ascending=False).index[:top_n])


def loss_vec(P, y):
    return -np.log(np.clip(P[np.arange(len(y)), y], 1e-9, 1.0))


def prepare(df, feats):
    ok = (df["hist_ok"] == 1) & df["split"].isin(["train", "valid", "test"])
    D = df[ok].copy()
    D["_dt"] = pd.to_datetime(D["kickoff_dt"], errors="coerce")
    D = D[D["_dt"].notna()].sort_values(["_dt", "match_id"]).reset_index(drop=True)
    for c in list(GOAL_COLS.values()) + ["total_corners"]:
        if c in D.columns:
            D[c] = pd.to_numeric(D[c], errors="coerce")
    return D


def active_targets(D, only=None):
    out = {}
    for name, col in TARGETS.items():
        if col in D.columns and D[col].notna().sum() > 1000 and (not only or name in only):
            out[name] = (col, sorted(set(D.loc[D[col].notna(), col].astype(str))))
    return out


# ------------------------------------------------------------------ WALK-FORWARD
def run_eval(df, feats, args):
    D = prepare(df, feats)
    n = len(D)
    start = int(n * (1 - args.eval_frac))
    edges = np.linspace(start, n, args.folds + 1).astype(int)
    feats = select_features(D.iloc[:start], feats, args.top_n)
    only = set(args.targets.split(",")) if args.targets else None
    targets = active_targets(D, only)
    logging.info("Maç: %d | fold: %d (her biri ~%d maç) | özellik: %d | hedef: %d",
                 n, args.folds, edges[1] - edges[0], len(feats), len(targets))

    acc = {t: {k: [] for k in ("y", "Pc", "Pp", "P0", "fold")} for t in targets}
    for f in range(args.folds):
        t0 = time.time()
        lo, hi = edges[f], edges[f + 1]
        past, test = D.iloc[:lo], D.iloc[lo:hi]
        cut = int(len(past) * (1 - VALID_SHARE))
        tr, va = past.iloc[:cut], past.iloc[cut:]
        ps = PoissonSystem(feats).fit(tr, va)
        PP = ps.predict(test)
        for name, (col, classes) in targets.items():
            if name not in PP:
                continue
            idx = {c: i for i, c in enumerate(classes)}
            m = test[col].notna().to_numpy()
            Pc = clf_predict(tr, va, test, col, classes, feats)
            if Pc is None or m.sum() == 0:
                continue
            y = test.loc[m, col].astype(str).map(idx).to_numpy()
            ytr = past.loc[past[col].notna(), col].astype(str).map(idx).to_numpy()
            prior = np.clip(np.bincount(ytr, minlength=len(classes)) / len(ytr), 1e-6, None)
            Pp = to_matrix(PP[name], classes, len(test))
            a = acc[name]
            a["y"].append(y); a["Pc"].append(Pc[m]); a["Pp"].append(Pp[m])
            a["P0"].append(np.tile(prior / prior.sum(), (m.sum(), 1))); a["fold"].append(np.full(m.sum(), f))
        logging.info("fold %d/%d bitti (%.0f sn)", f + 1, args.folds, time.time() - t0)

    rows, fold_rows = [], []
    for name, a in acc.items():
        if not a["y"]:
            continue
        y, fo = np.concatenate(a["y"]), np.concatenate(a["fold"])
        P = {"clf": np.vstack(a["Pc"]), "poisson": np.vstack(a["Pp"])}
        P["blend"] = 0.5 * (P["clf"] + P["poisson"])
        L0 = loss_vec(np.vstack(a["P0"]), y)
        L = {k: loss_vec(v, y) for k, v in P.items()}
        sk = {k: 1 - L[k].mean() / L0.mean() for k in L}
        diff = L["poisson"] - L["clf"]
        se = diff.std(ddof=1) / np.sqrt(len(diff))
        rows.append(dict(hedef=name, n=len(y), skill_clf=sk["clf"], skill_poisson=sk["poisson"],
                         skill_blend=sk["blend"], ll_fark_poi_clf=diff.mean(),
                         ci_alt=diff.mean() - 1.96 * se, ci_ust=diff.mean() + 1.96 * se,
                         en_iyi=max(sk, key=sk.get)))
        for f in np.unique(fo):
            mk = fo == f
            fold_rows.append(dict(hedef=name, fold=int(f), n=int(mk.sum()),
                                  **{f"skill_{k}": 1 - L[k][mk].mean() / L0[mk].mean() for k in L}))
    res = pd.DataFrame(rows).round(4)
    res.to_csv(DATA_DIR / "walkforward_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(fold_rows).round(4).to_csv(DATA_DIR / "walkforward_folds.csv", index=False, encoding="utf-8-sig")
    meth = pd.DataFrame({"method": ["clf", "poisson", "blend"],
                         "ortalama_skill": [res["skill_clf"].mean(), res["skill_poisson"].mean(),
                                            res["skill_blend"].mean()]}).round(5)
    meth.to_csv(DATA_DIR / "walkforward_methods.csv", index=False, encoding="utf-8-sig")
    print(res.to_string(index=False))
    print("\nOrtalama skill:\n" + meth.to_string(index=False))
    sig = int(((res["ci_ust"] < 0) | (res["ci_alt"] > 0)).sum())
    print(f"\nPoisson-sınıflandırıcı farkı %95 aralıkla sıfırdan ayrışan hedef: {sig}/{len(res)} "
          f"(ll_fark<0 => Poisson daha iyi)")


# ------------------------------------------------------------------ TAHMİN
def run_predict(df, feats, args):
    pred = df[df["split"] == "predict"]
    if args.all:
        day = "all"
    else:
        day = args.date or datetime.now(ZoneInfo(args.tz)).strftime("%Y-%m-%d")
        pred = pred[local_kickoff_date(pred, args.tz) == day]
    logging.info("Tahmin edilecek maç (%s): %d", day, len(pred))
    if len(pred) == 0:
        logging.warning("Bugün için oynanmamış maç yok (collector + build_features çalıştı mı?).")
        return

    method = args.method
    if method == "auto":
        p = DATA_DIR / "walkforward_methods.csv"
        if p.exists():
            m = pd.read_csv(p, encoding="utf-8-sig")
            method = str(m.loc[m["ortalama_skill"].idxmax(), "method"])
        else:
            method = "blend"
        logging.info("Seçilen yöntem: %s", method)

    D = prepare(df, feats)
    feats = select_features(D, feats, args.top_n)
    cut = int(len(D) * (1 - VALID_SHARE))
    tr, va = D.iloc[:cut], D.iloc[cut:]
    ps = PoissonSystem(feats).fit(tr, va).refit(D)
    PP = ps.predict(pred)

    out = pred[["match_id", "date", "league", "home_team", "away_team", "hist_ok"]].copy()
    out["yontem"] = method
    for key, nm in (("hg", "beklenen_gol_ev"), ("ag", "beklenen_gol_dep"), ("tc", "beklenen_korner")):
        if key in ps.last_mu:
            out[nm] = np.round(ps.last_mu[key], 2)

    for name, (col, classes) in active_targets(D).items():
        if name not in PP:
            continue
        Pp = to_matrix(PP[name], classes, len(pred))
        Pc = clf_predict(tr, va, pred, col, classes, feats, refit_df=D) if method != "poisson" else None
        P = {"poisson": Pp, "clf": Pc, "blend": None if Pc is None else 0.5 * (Pp + Pc)}[method]
        if P is None:
            P = Pp
        for j, c in enumerate(classes):
            out[f"{name}__{c}"] = np.round(P[:, j], 4)
    out.to_csv(DATA_DIR / f"predictions_final_{day}.csv", index=False, encoding="utf-8-sig")
    logging.info("Yazıldı: predictions_final_%s.csv", day)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["eval", "predict"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--eval-frac", type=float, default=0.5, help="verinin son bu kadarı walk-forward test")
    ap.add_argument("--top-n", type=int, default=150, help="özellik sayısı (0 = hepsi)")
    ap.add_argument("--targets", default=None, help="virgülle hedef adları (eval için)")
    ap.add_argument("--date", default=None)
    ap.add_argument("--tz", default="Europe/Istanbul")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--method", choices=["auto", "poisson", "clf", "blend"], default="auto")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    df = tm.load()
    feats = tm.get_features(df)
    (run_eval if args.mode == "eval" else run_predict)(df, feats, args)


if __name__ == "__main__":
    main()
