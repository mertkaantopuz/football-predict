"""
Maç Tahminleri - Streamlit uygulaması

Ana sayfa: seçilen günün maçları (lige göre gruplu). Bir maça tıklayınca o maçın tüm tahminleri açılır.
Veri: train_models.py'nin ürettiği günlük dosyalar (predictions_YYYY-MM-DD.csv) ve son çalışmanın
kopyası predictions_predict.csv; hepsi birleştirilir, aynı maç birden çok dosyada varsa günlük dosya (en yeni gün) kalır.
Zaman damgalı dosyalar (predictions_YYYY-MM-DD_HHMM.csv) okunmaz. İsteğe bağlı: model_results.csv.

Dosya yeri (sırayla aranır):
    1) OUTPUT_DIR ortam değişkeni (varsa)   2) <proje>/data   3) bu dosyanın yanı (Streamlit Cloud)
Yerelde çalıştır:   streamlit run web/app.py        (proje klasöründen)
Zorlamak için (secrets/ortam değişkeni): PREDICTIONS_URL, RESULTS_URL
"""

import os
from datetime import datetime
from pathlib import Path
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


PRED_FILES = prediction_files()
RES_SRC = find_source("model_results.csv", "RESULTS_URL")


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


@st.cache_data(ttl=600)
def load_skills(src, mtime):
    try:
        r = pd.read_csv(src, encoding="utf-8-sig")
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


def get_probs(row, name):
    return [(cls, float(row[col])) for cls, col in TARGETS.get(name, []) if pd.notna(row[col])]


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


def badge(name):
    s = SKILLS.get(name)
    if s is None or not GOSTER_MODEL_GUCU:
        return ""
    if s >= 0.02:
        return f"🟢 Model gücü: +%{s * 100:.1f}"
    if s >= 0.01:
        return f"🟡 Model gücü: +%{s * 100:.1f}"
    return f"⚪ Zayıf sinyal (+%{s * 100:.1f}): temkinli yorumla"


def show_badge(name):
    text = badge(name)
    if text:
        st.caption(text)


def prob_rows(items):
    for label, p in items:
        c1, c2 = st.columns([3, 5])
        c1.write(label)
        c2.progress(min(max(p, 0.0), 1.0), text=f"%{p * 100:.1f}")


def quick_pick(row):
    probs = get_probs(row, "ms_1x2")
    if not probs:
        return ""
    cls, p = max(probs, key=lambda x: x[1])
    return f"MS tahmini: {outcome_labels(row).get(cls, cls)} (%{p * 100:.0f})"


def fair_odds(p):
    return f"{1 / p:.2f}" if p and p > 0.001 else "—"


def over_under_table(lines):
    rows = []
    for name, line in lines:
        probs = get_probs(ROW, name)
        if len(probs) != 2:
            continue
        p_over = probs[-1][1]  # LabelEncoder sıralaması: son sınıf = 1 / üst
        skill = SKILLS.get(name)
        row = {
            "Çizgi": line,
            "Üst %": round(p_over * 100, 1),
            "Alt %": round((1 - p_over) * 100, 1),
        }
        if GOSTER_ADIL_ORAN:
            row["Adil oran (Üst)"] = fair_odds(p_over)
            row["Adil oran (Alt)"] = fair_odds(1 - p_over)
        if GOSTER_MODEL_GUCU:
            row["Model gücü"] = f"%{skill * 100:+.1f}" if skill is not None else "—"
        rows.append(row)
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


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


def detail_view(row):
    if st.button("← Maç listesine dön"):
        st.session_state["mid"] = None
        st.rerun()

    st.title(f"{row['home_team']} - {row['away_team']}")
    st.caption(f"{row['league']}  |  {row['kickoff_dt'].strftime('%d.%m.%Y %H:%M')}")
    if (GOSTER_VERI_YETERSIZ_UYARISI and "hist_ok" in row and pd.notna(row["hist_ok"])
            and int(row["hist_ok"]) == 0):
        st.warning("Bu maçta takımların geçmiş verisi yetersiz; tahminler daha güvenilmez.")

    labels = outcome_labels(row)

    if BOLUMLER.get("ms_sonucu"):
        st.header("Maç sonucu")
        probs = ordered_1x2(get_probs(row, "ms_1x2"))
        if probs:
            prob_rows([(labels.get(c, c), p) for c, p in probs])
            show_badge("ms_1x2")

    if BOLUMLER.get("iy_sonucu"):
        st.header("İlk yarı sonucu")
        probs = ordered_1x2(get_probs(row, "iy_1x2"))
        if probs:
            prob_rows([(labels.get(c, c), p) for c, p in probs])
            show_badge("iy_1x2")

    if BOLUMLER.get("iy_ms"):
        probs = get_probs(row, "iy_ms")
        if probs:
            st.subheader("İlk yarı / Maç sonu (en olası 5)")
            prob_rows([(c, p) for c, p in sorted(probs, key=lambda x: -x[1])[:5]])
            show_badge("iy_ms")

    if BOLUMLER.get("kg"):
        st.header("Karşılıklı gol")
        for name, title in (("kg_var", "Maç sonu"), ("iy_kg_var", "İlk yarı")):
            probs = get_probs(row, name)
            if len(probs) == 2:
                p = probs[-1][1]
                st.write(f"**{title}**")
                prob_rows([("KG Var", p), ("KG Yok", 1 - p)])
                show_badge(name)

    if any(BOLUMLER.get(k) for k in ("alt_ust_ms", "alt_ust_iy", "alt_ust_2y")):
        st.header("Alt / Üst")
    if BOLUMLER.get("alt_ust_ms"):
        st.subheader("Maç sonu gol")
        over_under_table([(f"ms_ust_{x.replace('.', '_')}", x) for x in ("0.5", "1.5", "2.5", "3.5", "4.5")])
    if BOLUMLER.get("alt_ust_iy"):
        st.subheader("İlk yarı gol")
        over_under_table([(f"iy_ust_{x.replace('.', '_')}", x) for x in ("0.5", "1.5")])
    if BOLUMLER.get("alt_ust_2y"):
        st.subheader("İkinci yarı gol")
        over_under_table([(f"2y_ust_{x.replace('.', '_')}", x) for x in ("0.5", "1.5")])

    if BOLUMLER.get("toplam_gol"):
        st.header("Kaç gol olur?")
        for name, title in (("toplam_gol", "Maç sonu toplam gol"), ("iy_toplam_gol", "İlk yarı toplam gol")):
            probs = get_probs(row, name)
            if probs:
                n = len(probs)
                chart = pd.DataFrame({"Olasılık %": [round(p * 100, 1) for _, p in probs]},
                                     index=[cls_label(c, last=(i == n - 1), plus=True) for i, (c, _) in enumerate(probs)])
                st.write(f"**{title}**")
                st.bar_chart(chart)
                show_badge(name)

    if BOLUMLER.get("korner"):
        st.header("Korner")
        over_under_table([(f"korner_ust_{x.replace('.', '_')}", x) for x in ("8.5", "9.5", "10.5")])

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
    st.error(f"Tahmin dosyası okunamadı ({', '.join(PRED_FILES)}): {e}")
    st.stop()

if DF.empty:
    st.info("Henüz tahmin yok. train_models.py çalıştırıp predictions_predict.csv dosyasını üret.")
    st.stop()

TARGETS = target_columns(DF)
SKILLS = load_skills(RES_SRC, mtime_of(RES_SRC))

if "mid" not in st.session_state:
    st.session_state["mid"] = None

mid = st.session_state["mid"]
match = DF[DF["match_id"] == mid]
if mid and not match.empty:
    ROW = match.iloc[0]
    detail_view(ROW)
else:
    st.session_state["mid"] = None
    list_view(DF)
