"""
Güncel tahminleri GitHub'a gönderir (Streamlit Community Cloud uygulamayı kendisi yeniler).
train_models.py bittikten sonra proje klasöründen çalıştır:

    python publish_predictions.py

Yaptığı: data/predictions_predict.csv, günlük data/predictions_YYYY-MM-DD.csv dosyaları ve
data/model_results.csv, ikinci model için poisson_predictions_YYYY-MM-DD.csv ve poisson_results.csv
dosyalarını commit'ler ve push eder. (Bunlar .gitignore'da istisna olarak tanımlı;
data/ altındaki diğer her şey, zaman damgalı predictions_..._HHMM.csv dosyaları dahil, GitHub'a gitmez.)
Depo henüz kurulmadıysa: README'deki "GitHub'a ilk yükleme" adımlarına bak.
"""
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FIXED = ["data/predictions_predict.csv", "data/model_results.csv", "data/poisson_results.csv"]


def git(*args, check=True):
    return subprocess.run(["git", "-C", str(ROOT), *args], check=check)


def main():
    if not (ROOT / ".git").exists():
        sys.exit("Bu klasör henüz bir git deposu değil. Önce ilk yüklemeyi yap (git init / Publish to GitHub).")

    daily = sorted(f"data/{p.name}" for pat in ("predictions_????-??-??.csv", "poisson_predictions_????-??-??.csv")
                   for p in (ROOT / "data").glob(pat))
    files = FIXED + daily
    existing = [f for f in files if (ROOT / f).exists()]
    for f in set(files) - set(existing):
        print(f"Atlandı (bulunamadı): {f}")
    if not existing:
        sys.exit("Gönderilecek tahmin dosyası yok.")

    git("add", *existing)
    if git("diff", "--cached", "--quiet", check=False).returncode == 0:
        print("Yeni değişiklik yok. Daha önce gönderilememiş commit varsa gönderiliyor...")
        git("push", "-u", "origin", "HEAD", check=False)
        return
    git("commit", "-m", f"Tahminleri güncelle {datetime.now():%Y-%m-%d %H:%M}")
    git("push", "-u", "origin", "HEAD")
    print("Gönderildi.")


if __name__ == "__main__":
    main()
