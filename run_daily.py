"""
Günlük akışı tek komutla çalıştırır:
    1) data_collector_claude.py   (biten maçları ve takvimi güncelle)
    2) build_features.py          (özellikleri yeniden üret)
    3) train_models.py            (bugünün tahminleri)
    4) publish_predictions.py     (GitHub'a gönder -> Streamlit güncellenir)

Kullanım (proje klasöründe, .venv açıkken):
    python run_daily.py                       -> tüm adımlar
    python run_daily.py --skip-collect        -> veri toplamayı atla (zaten güncelse)
    python run_daily.py --date 2026-11-06     -> train_models.py'ye --date ilet (başka bir gün için tahmin)

Bir adım hata verirse durur (eski veriyle yayın yapılmaz). Çıktılar run_daily.log dosyasına da yazılır.

train_models.py her çalışmada data/ altına predictions_YYYY-MM-DD_HHMM.csv, predictions_YYYY-MM-DD.csv ve
predictions_predict.csv yazar. Yayın her zaman predictions_predict.csv'yi (son çalışmanın kopyası) gönderir.
Tahmin adımından sonra bu dosyanın BU çalıştırmada yazıldığı kontrol edilir; yazılmadıysa (oynanacak maç yok)
dünkü dosya yeniden yayınlanmasın diye yayın adımı atlanır.
Zamanlama için Windows Görev Zamanlayıcı kullanılabilir (açıklama sohbette).
"""
import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOG = ROOT / "run_daily.log"
DATA = ROOT / "data"

STEPS = [
    ("Veri toplama", "data_collector_claude.py"),
    ("Özellik üretimi", "build_features.py"),
    ("Tahmin", "train_models.py"),
    ("Yayın", "publish_predictions.py"),
]


def check_fresh(since_ts):
    """Bu çalıştırmada yazılmış tahmin dosyası var mı? (maç sayısı, zaman damgalı dosya adı) ya da None."""
    main_file = DATA / "predictions_predict.csv"
    if not main_file.exists() or main_file.stat().st_mtime < since_ts:
        return None
    stamped = sorted(
        (f for f in DATA.glob("predictions_????-??-??_????.csv") if f.stat().st_mtime >= since_ts),
        key=lambda f: f.stat().st_mtime,
    )
    with open(main_file, encoding="utf-8-sig") as f:
        rows = max(sum(1 for _ in f) - 1, 0)
    return rows, (stamped[-1].name if stamped else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-collect", action="store_true", help="veri toplama adımını atla")
    ap.add_argument("--date", default=None, help="train_models.py'ye iletilir (YYYY-MM-DD)")
    args = ap.parse_args()

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass

    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    t0 = datetime.now()
    t_start = time.time() - 2

    with open(LOG, "a", encoding="utf-8") as log:
        def say(msg):
            line = f"{datetime.now():%Y-%m-%d %H:%M:%S} | {msg}\n"
            print(line, end="")
            log.write(line)
            log.flush()

        say("=== Günlük akış başladı ===")
        for title, script in STEPS:
            if args.skip_collect and script == "data_collector_claude.py":
                say(f"[atlandı] {title}")
                continue
            cmd = [sys.executable, str(ROOT / script)]
            if script == "train_models.py" and args.date:
                cmd += ["--date", args.date]

            say(f"[başladı] {title}: {' '.join(cmd[1:])}")
            proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, encoding="utf-8", errors="replace", bufsize=1)
            for line in proc.stdout:
                print(line, end="")
                log.write(line)
            code = proc.wait()
            log.flush()
            if code != 0:
                say(f"[HATA] {title} adımı {code} koduyla bitti. Akış durduruldu, yayın yapılmadı.")
                sys.exit(code)
            say(f"[bitti] {title}")

            if script == "train_models.py":
                fresh = check_fresh(t_start)
                if fresh is None or fresh[0] == 0:
                    say("[UYARI] Bu çalıştırmada yeni tahmin dosyası üretilmedi (oynanacak maç yok ya da hepsi "
                        "atlandı). Yayın adımı atlandı; GitHub'daki dosya olduğu gibi kalıyor.")
                    say("=== Tamamlandı (yayın yapılmadı) ===")
                    return
                say(f"Tahmin dosyası: {fresh[1] or 'predictions_predict.csv'} | maç sayısı: {fresh[0]}")

        say(f"=== Tamamlandı ({(datetime.now() - t0).seconds // 60} dk) ===")


if __name__ == "__main__":
    main()
