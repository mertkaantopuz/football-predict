"""
Maç Tahminleri - Streamlit uygulaması

Ana sayfa: seçilen günün maçları (lige göre gruplu). Bir maça tıklayınca o maçın tüm tahminleri açılır.
Veri: train_models.py'nin ürettiği günlük dosyalar (predictions_YYYY-MM-DD.csv) ve son çalışmanın
kopyası predictions_predict.csv; hepsi birleştirilir, aynı maç birden çok dosyada varsa günlük dosya (en yeni gün) kalır.
Zaman damgalı dosyalar (predictions_YYYY-MM-DD_HHMM.csv) okunmaz. İsteğe bağlı: model_results.csv.
İkinci model (isteğe bağlı): poisson_predictions_YYYY-MM-DD.csv (+ poisson_results.csv). Ana sayfa yalnızca
birinci modelin maçlarını listeler; maç sayfasında iki modelin tahmini match_id ile eşleşip yan yana gösterilir.

Dosya yeri (sırayla aranır):
    1) OUTPUT_DIR ortam değişkeni (varsa)   2) <proje>/data   3) bu dosyanın yanı (Streamlit Cloud)
Yerelde çalıştır:   streamlit run web/app.py        (proje klasöründen)
Zorlamak için (secrets/ortam değişkeni): PREDICTIONS_URL, RESULTS_URL
"""

import os
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

# ============================================================================
# AYARLAR: sayfada neyin görüneceğini buradan belirle.
# True = göster, False = gizle. Metinleri doğrudan değiştirebilirsin.
# Değişiklikten sonra dosyayı kaydet, GitHub'a gönder (git add/commit/push);
# Streamlit kendisi yeniler. Yerelde "streamlit run web/app.py" anında yansır.
# ============================================================================

# --- Genel metinler
SEKME_BASLIGI = "Maç Tahminleri"          # tarayıcı sekmesinde yazan
BASLIK = "⚽ Günün Maçları"                # ana sayfa başlığı ("" yazarsan gizlenir)
UYARI_METNI = "İstatistiksel model çıktısıdır, bahis tavsiyesi değildir."
GOSTER_TEKNIK_DETAY = True                # hata/boş durum mesajlarında dosya adı ve hata ayrıntısı

# --- Ana sayfa (maç listesi)
GOSTER_TARIH_FILTRESI_UYARISI = True      # "Geçmiş tarih gösteriliyor." gibi bilgi kutusu
GOSTER_LIG_FILTRESI = True                # "Lig" seçim kutusu
GOSTER_MAC_SAYISI_NOTU = True             # "12 maç. Tahminini görmek için maça tıkla."
GOSTER_KAYNAK_DOSYA = True                # "Kaynak dosya: predictions_2026-11-05.csv"
GOSTER_GUNCELLEME_ZAMANI = True           # "Tahminler en son güncellendi: ..."
GOSTER_MS_OZETI = True                    # her maçın altındaki "MS tahmini: ..." satırı
GOSTER_UYARI_NOTU = True                  # en alttaki UYARI_METNI (ana sayfa ve maç sayfası)

# --- Maç sayfası: bölümler
BOLUMLER = {
    "ms_sonucu": True,    # Maç sonucu (1 / X / 2)
    "iy_sonucu": True,    # İlk yarı sonucu
    "iy_ms": True,        # İlk yarı / Maç sonu kombinasyonları
    "kg": True,           # Karşılıklı gol
    "alt_ust_ms": True,   # Alt/Üst: maç sonu gol
    "alt_ust_iy": True,   # Alt/Üst: ilk yarı gol
    "alt_ust_2y": True,   # Alt/Üst: ikinci yarı gol
    "toplam_gol": True,   # Kaç gol olur? grafikleri
    "korner": True,       # Korner alt/üst
}

# --- İkinci model (yalnızca maç sayfasında yan yana gösterilir; ana sayfa değişmez)
GOSTER_IKINCI_MODEL = True                # False: yalnızca birinci model görünür
MODEL_ADI_1 = "LightGBM"                  # predictions_*.csv dosyasının model adı
MODEL_ADI_2 = "Poisson"                   # poisson_predictions_*.csv dosyasının model adı

# --- Maç sayfası: ayrıntılar
GOSTER_MODEL_GUCU = True                  # 🟢/🟡/⚪ rozetler ve tablodaki "Model gücü" sütunu
GOSTER_ADIL_ORAN = True                   # tablodaki "Adil oran" sütunları
GOSTER_VERI_YETERSIZ_UYARISI = True       # "Bu maçta geçmiş verisi yetersiz" uyarısı
GOSTER_ACIKLAMA_NOTU = True               # en alttaki "Adil oran = 1 / olasılık ..." açıklaması
# ============================================================================

st.set_page_config(page_title=SEKME_BASLIGI, page_icon="⚽", layout="centered")

TZ = ZoneInfo("Europe/Istanbul")


def cfg(name, default):
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:  # secrets dosyası yoksa
        pass
    return os.getenv(name, default)


BASE = Path(__file__).resolve().parent


def data_dirs():
    out = []
    env = os.getenv("OUTPUT_DIR")
    if env:
        p = Path(env)
        out.append(p if p.is_absolute() else BASE.parent / p)
    out += [BASE.parent / "data", BASE]
    return out


def find_source(filename, cfg_key):
    explicit = cfg(cfg_key, "")
    if explicit:
        return str(explicit)
    for d in data_dirs():
        if (d / filename).exists():
            return str(d / filename)
    return str(BASE / filename)


def mtime_of(src):
    try:
        return os.path.getmtime(src)
    except OSError:  # URL ya da bulunamadı
        return 0.0


def prediction_files():
    explicit = cfg("PREDICTIONS_URL", "")
    if explicit:
        return [str(explicit)]
    for d in data_dirs():
        daily = sorted(d.glob("predictions_????-??-??.csv"))
        latest = d / "predictions_predict.csv"
        if daily or latest.exists():
            # predictions_predict.csv günlük dosyalardan birinin kopyasıdır: en başa konur ki
            # aynı maç iki dosyada varsa kaynak olarak günlük dosya (en yeni gün) kalsın
            return ([str(latest)] if latest.exists() else []) + [str(f) for f in daily]
    return [str(BASE / "predictions_predict.csv")]


def prediction2_files():
    explicit = cfg("PREDICTIONS2_URL", "")
    if explicit:
        return [str(explicit)]
    for d in data_dirs():
        daily = sorted(d.glob("poisson_predictions_????-??-??.csv"))
        if daily:
            return [str(f) for f in daily]
    return []


PRED_FILES = prediction_files()
PRED2_FILES = prediction2_files() if GOSTER_IKINCI_MODEL else []
RES_SRC = find_source("model_results.csv", "RESULTS_URL")
RES2_SRC = find_source("poisson_results.csv", "RESULTS2_URL")


# ----------------------------------------------------------------------------- veri
@st.cache_data(ttl=600, show_spinner="Veriler yükleniyor...")
def load_predictions(srcs, mtimes):  # mtimes: dosya değişince önbellek yenilensin diye
    frames = []
    for src in srcs:
        try:
            f = pd.read_csv(src, encoding="utf-8-sig")
        except Exception:  # noqa: BLE001
            continue
        f["kaynak"] = Path(str(src)).name
        frames.append(f)
    if not frames:
        raise FileNotFoundError(", ".join(map(str, srcs)))
    df = pd.concat(frames, ignore_index=True)
    df["match_id"] = df["match_id"].astype(str)
    df = df.drop_duplicates("match_id", keep="last")  # sonraki dosya öncekini geçer (günlük dosyalar eski -> yeni)
    when = "kickoff_dt" if "kickoff_dt" in df.columns else "date"
    df["kickoff_dt"] = pd.to_datetime(df[when], errors="coerce")
    df = df[df["kickoff_dt"].notna()].copy()
    df["gun"] = df["kickoff_dt"].dt.date
    return df.sort_values(["kickoff_dt", "league", "home_team"]).reset_index(drop=True)


@st.cache_data(ttl=600, show_spinner=False)
def load_second(srcs, mtimes):
    """İkinci modelin günlük dosyaları: match_id ile eşleştirilir, aynı maç için en yeni dosya kalır."""
    frames = []
    for src in srcs:
        try:
            f = pd.read_csv(src, encoding="utf-8-sig")
        except Exception:  # noqa: BLE001
            continue
        f["kaynak"] = Path(str(src)).name
        frames.append(f)
    if not frames:
        raise FileNotFoundError(", ".join(map(str, srcs)))
    df = pd.concat(frames, ignore_index=True)
    df["match_id"] = df["match_id"].astype(str)
    return df.drop_duplicates("match_id", keep="last").set_index("match_id")


@st.cache_data(ttl=600)
def load_skills(src, mtime):
    """{hedef: skill}. 'skill' sütunu yoksa logloss_model/logloss_temel'den hesaplanır; yoksa boş döner."""
    try:
        r = pd.read_csv(src, encoding="utf-8-sig")
        if "skill" not in r.columns and {"logloss_model", "logloss_temel"} <= set(r.columns):
            r["skill"] = 1 - r["logloss_model"] / r["logloss_temel"]
        return dict(zip(r["hedef"], r["skill"]))
    except Exception:
        return {}


def target_columns(df):
    """{'ms_1x2': [('1', 'ms_1x2__1'), ('X', ...), ...], ...} (dosyadaki sütun sırasıyla)"""
    t = {}
    for c in df.columns:
        if "__" in c:
            name, cls = c.split("__", 1)
            t.setdefault(name, []).append((cls, c))
    return t


def make_model(name, row, targets, skills):
    """Bir modelin o maça ait tahmin satırı + sütun haritası + model gücü değerleri."""
    return SimpleNamespace(name=name, row=row, targets=targets, skills=skills)


def get_probs(model, name):
    out = []
    for cls, col in model.targets.get(name, []):
        if col in model.row.index and pd.notna(model.row[col]):
            out.append((cls, float(model.row[col])))
    return out


# ----------------------------------------------------------------------------- yardımcılar
def cls_label(c, last=False, plus=False):
    try:
        txt = str(int(float(c)))
    except ValueError:
        return str(c)
    return txt + "+" if (last and plus) else txt


def outcome_labels(row):
    return {
        "1": f"{row['home_team']} kazanır", "H": f"{row['home_team']} kazanır",
        "X": "Beraberlik", "D": "Beraberlik",
        "2": f"{row['away_team']} kazanır", "A": f"{row['away_team']} kazanır",
    }


def ordered_1x2(probs):
    order = {"1": 0, "H": 0, "X": 1, "D": 1, "2": 2, "A": 2}
    if all(c in order for c, _ in probs):
        return sorted(probs, key=lambda x: order[x[0]])
    return probs


def badge(model, name):
    s = model.skills.get(name)
    if s is None or not GOSTER_MODEL_GUCU:
        return ""
    if s >= 0.02:
        return f"🟢 Model gücü: +%{s * 100:.1f}"
    if s >= 0.01:
        return f"🟡 Model gücü: +%{s * 100:.1f}"
    return f"⚪ Zayıf sinyal (+%{s * 100:.1f}): temkinli yorumla"


def show_badge(model, name):
    text = badge(model, name)
    if text:
        st.caption(text)


def prob_rows(items):
    for label, p in items:
        c1, c2 = st.columns([3, 5])
        c1.write(label)
        c2.progress(min(max(p, 0.0), 1.0), text=f"%{p * 100:.1f}")


def quick_pick(row):
    probs = get_probs(make_model(MODEL_ADI_1, row, TARGETS, SKILLS), "ms_1x2")
    if not probs:
        return ""
    cls, p = max(probs, key=lambda x: x[1])
    return f"MS tahmini: {outcome_labels(row).get(cls, cls)} (%{p * 100:.0f})"


def fair_odds(p):
    return f"{1 / p:.2f}" if p is not None and p > 0.001 else "—"


def over_under_table(models, lines):
    """Alt/üst tablosu. Birden çok model varsa aynı metrikler yan yana sütunlar olur."""
    multi = len(models) > 1
    active = [m for m in models if any(len(get_probs(m, n)) == 2 for n, _ in lines)]
    if not active:
        return

    def lab(text, m):
        return f"{text} ({m.name})" if multi else text

    rows = []
    for name, line in lines:
        p_over = {}
        for m in active:
            probs = get_probs(m, name)
            p_over[m.name] = probs[-1][1] if len(probs) == 2 else None  # son sınıf = 1 / üst
        if all(v is None for v in p_over.values()):
            continue
        row = {"Çizgi": line}
        for m in active:
            p = p_over[m.name]
            row[lab("Üst %", m)] = round(p * 100, 1) if p is not None else None
        for m in active:
            p = p_over[m.name]
            row[lab("Alt %", m)] = round((1 - p) * 100, 1) if p is not None else None
        if GOSTER_ADIL_ORAN:
            for m in active:
                p = p_over[m.name]
                row[lab("Adil Üst" if multi else "Adil oran (Üst)", m)] = fair_odds(p)
            for m in active:
                p = p_over[m.name]
                row[lab("Adil Alt" if multi else "Adil oran (Alt)", m)] = fair_odds(None if p is None else 1 - p)
        if GOSTER_MODEL_GUCU:
            for m in active:
                sk = m.skills.get(name)
                row[lab("Güç" if multi else "Model gücü", m)] = f"%{sk * 100:+.1f}" if sk is not None else "—"
        rows.append(row)
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    if multi:
        for m in models:
            if all(m is not a for a in active):
                st.caption(f"{m.name}: bu bölüm için tahmin yok.")


# ----------------------------------------------------------------------------- sayfalar
def list_view(df):
    if BASLIK:
        st.title(BASLIK)
    today = datetime.now(TZ).date()
    days = sorted(df["gun"].unique(), reverse=True)  # en yeni üstte
    default = today if today in days else min((d for d in days if d >= today), default=days[0])

    if GOSTER_LIG_FILTRESI:
        c1, c2 = st.columns(2)
    else:
        c1, c2 = st.container(), None
    sel = c1.selectbox(
        "Tarih", days, index=days.index(default),
        format_func=lambda d: d.strftime("%d.%m.%Y") + (" (bugün)" if d == today else ""),
    )
    day_df = df[df["gun"] == sel]
    league = (c2.selectbox("Lig", ["Tüm ligler"] + sorted(day_df["league"].dropna().unique()))
              if c2 is not None else "Tüm ligler")
    if league != "Tüm ligler":
        day_df = day_df[day_df["league"] == league]

    if sel != today and GOSTER_TARIH_FILTRESI_UYARISI:
        st.info("Bugün için tahmin bulunmuyor; en yakın tarih gösteriliyor." if sel > today else "Geçmiş tarih gösteriliyor.")
    if GOSTER_MAC_SAYISI_NOTU:
        st.caption(f"{len(day_df)} maç. Tahminini görmek için maça tıkla.")
    if GOSTER_KAYNAK_DOSYA:
        st.caption("Kaynak dosya: " + ", ".join(sorted(day_df["kaynak"].astype(str).unique())))
    mt = max((mtime_of(f) for f in PRED_FILES), default=0)
    if mt and GOSTER_GUNCELLEME_ZAMANI:
        st.caption("Tahminler en son güncellendi: "
                   + datetime.fromtimestamp(mt, TZ).strftime("%d.%m.%Y %H:%M"))

    for lg, g in day_df.groupby("league", sort=True):
        st.subheader(lg)
        for _, r in g.iterrows():
            hhmm = r["kickoff_dt"].strftime("%H:%M")
            if st.button(f"{hhmm}   {r['home_team']} - {r['away_team']}", key=f"b_{r['match_id']}",
                         use_container_width=True):
                st.session_state["mid"] = r["match_id"]
                st.rerun()
            pick = quick_pick(r) if GOSTER_MS_OZETI else ""
            if pick:
                st.caption(pick)

    if GOSTER_UYARI_NOTU and UYARI_METNI:
        st.divider()
        st.caption(UYARI_METNI)


def build_models(row):
    """[birinci model] ya da [birinci, ikinci] (ikinci modelde o maç varsa), ikinci model eksik mi?"""
    models = [make_model(MODEL_ADI_1, row, TARGETS, SKILLS)]
    second_missing = False
    if DF2 is not None:
        if row["match_id"] in DF2.index:
            models.append(make_model(MODEL_ADI_2, DF2.loc[row["match_id"]], TARGETS2, SKILLS2))
        else:
            second_missing = True
    return models, second_missing


def section(models, render):
    """Her model için aynı bölümü yan yana sütunlarda çizer (tek modelde tam genişlik)."""
    multi = len(models) > 1
    cols = st.columns(len(models)) if multi else [st.container()]
    for col, m in zip(cols, models):
        with col:
            if multi:
                st.caption(f"▪ {m.name}")
            ok = render(m)
            if multi and ok is False:
                st.caption("Bu modelde bu tahmin yok.")


def detail_view(row):
    if st.button("← Maç listesine dön"):
        st.session_state["mid"] = None
        st.rerun()

    models, second_missing = build_models(row)

    st.title(f"{row['home_team']} - {row['away_team']}")
    st.caption(f"{row['league']}  |  {row['kickoff_dt'].strftime('%d.%m.%Y %H:%M')}")
    if second_missing:
        st.caption(f"{MODEL_ADI_2} modelinde bu maç için tahmin yok; yalnızca {MODEL_ADI_1} gösteriliyor.")
    if (GOSTER_VERI_YETERSIZ_UYARISI and "hist_ok" in row and pd.notna(row["hist_ok"])
            and int(row["hist_ok"]) == 0):
        st.warning("Bu maçta takımların geçmiş verisi yetersiz; tahminler daha güvenilmez.")

    labels = outcome_labels(row)

    def r_1x2(target):
        def render(m):
            probs = ordered_1x2(get_probs(m, target))
            if not probs:
                return False
            prob_rows([(labels.get(c, c), p) for c, p in probs])
            show_badge(m, target)
        return render

    def r_iy_ms(m):
        probs = get_probs(m, "iy_ms")
        if not probs:
            return False
        prob_rows([(c, p) for c, p in sorted(probs, key=lambda x: -x[1])[:5]])
        show_badge(m, "iy_ms")

    def r_kg(m):
        shown = False
        for name, title in (("kg_var", "Maç sonu"), ("iy_kg_var", "İlk yarı")):
            probs = get_probs(m, name)
            if len(probs) == 2:
                p = probs[-1][1]
                st.write(f"**{title}**")
                prob_rows([("KG Var", p), ("KG Yok", 1 - p)])
                show_badge(m, name)
                shown = True
        return shown or False

    def r_goals(name, title):
        def render(m):
            probs = get_probs(m, name)
            if not probs:
                return False
            n = len(probs)
            chart = pd.DataFrame({"Olasılık %": [round(p * 100, 1) for _, p in probs]},
                                 index=[cls_label(c, last=(i == n - 1), plus=True) for i, (c, _) in enumerate(probs)])
            st.write(f"**{title}**")
            st.bar_chart(chart)
            show_badge(m, name)
        return render

    if BOLUMLER.get("ms_sonucu"):
        st.header("Maç sonucu")
        section(models, r_1x2("ms_1x2"))

    if BOLUMLER.get("iy_sonucu"):
        st.header("İlk yarı sonucu")
        section(models, r_1x2("iy_1x2"))

    if BOLUMLER.get("iy_ms") and any(get_probs(m, "iy_ms") for m in models):
        st.subheader("İlk yarı / Maç sonu (en olası 5)")
        section(models, r_iy_ms)

    if BOLUMLER.get("kg"):
        st.header("Karşılıklı gol")
        section(models, r_kg)

    if any(BOLUMLER.get(k) for k in ("alt_ust_ms", "alt_ust_iy", "alt_ust_2y")):
        st.header("Alt / Üst")
    if BOLUMLER.get("alt_ust_ms"):
        st.subheader("Maç sonu gol")
        over_under_table(models, [(f"ms_ust_{x.replace('.', '_')}", x) for x in ("0.5", "1.5", "2.5", "3.5", "4.5")])
    if BOLUMLER.get("alt_ust_iy"):
        st.subheader("İlk yarı gol")
        over_under_table(models, [(f"iy_ust_{x.replace('.', '_')}", x) for x in ("0.5", "1.5")])
    if BOLUMLER.get("alt_ust_2y"):
        st.subheader("İkinci yarı gol")
        over_under_table(models, [(f"2y_ust_{x.replace('.', '_')}", x) for x in ("0.5", "1.5")])

    if BOLUMLER.get("toplam_gol"):
        st.header("Kaç gol olur?")
        section(models, r_goals("toplam_gol", "Maç sonu toplam gol"))
        section(models, r_goals("iy_toplam_gol", "İlk yarı toplam gol"))

    if BOLUMLER.get("korner") and any(get_probs(m, f"korner_ust_{x.replace('.', '_')}")
                                       for m in models for x in ("8.5", "9.5", "10.5")):
        st.header("Korner")
        over_under_table(models, [(f"korner_ust_{x.replace('.', '_')}", x) for x in ("8.5", "9.5", "10.5")])

    notes = []
    if GOSTER_ACIKLAMA_NOTU:
        if GOSTER_ADIL_ORAN:
            notes.append("Adil oran = 1 / model olasılığı; bahis şirketi oranı değildir.")
        if GOSTER_MODEL_GUCU:
            notes.append("Model gücü, test döneminde sabit tahmine göre log-loss iyileşmesidir; "
                         "düşükse o tahmine fazla güvenme.")
    if GOSTER_UYARI_NOTU and UYARI_METNI:
        notes.append(UYARI_METNI)
    if notes:
        st.divider()
        st.caption(" ".join(notes))


# ----------------------------------------------------------------------------- ana akış
try:
    DF = load_predictions(tuple(PRED_FILES), tuple(mtime_of(f) for f in PRED_FILES))
except Exception as e:  # noqa: BLE001
    if GOSTER_TEKNIK_DETAY:
        st.error(f"Tahmin dosyası okunamadı ({', '.join(PRED_FILES)}): {e}")
    else:
        st.error("Tahminler şu anda yüklenemedi. Lütfen daha sonra tekrar dene.")
    st.stop()

if DF.empty:
    st.info("Henüz tahmin yok. train_models.py çalıştırıp predictions_predict.csv dosyasını üret."
            if GOSTER_TEKNIK_DETAY else "Henüz tahmin yok.")
    st.stop()

TARGETS = target_columns(DF)
SKILLS = load_skills(RES_SRC, mtime_of(RES_SRC))

# İkinci model (isteğe bağlı): yoksa ya da okunamazsa uygulama yalnızca birinci modelle çalışır
DF2, TARGETS2, SKILLS2 = None, {}, {}
if PRED2_FILES:
    try:
        DF2 = load_second(tuple(PRED2_FILES), tuple(mtime_of(f) for f in PRED2_FILES))
        TARGETS2 = target_columns(DF2)
        SKILLS2 = load_skills(RES2_SRC, mtime_of(RES2_SRC))
    except Exception:  # noqa: BLE001
        DF2 = None

if "mid" not in st.session_state:
    st.session_state["mid"] = None

mid = st.session_state["mid"]
match = DF[DF["match_id"] == mid]
if mid and not match.empty:
    detail_view(match.iloc[0])
else:
    st.session_state["mid"] = None
    list_view(DF)
