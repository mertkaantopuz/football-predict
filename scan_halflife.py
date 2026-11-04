"""Yarı ömür taraması: HALF_LIFE_DAYS değerlerini dener, validation'a göre karşılaştırır."""
import os
import subprocess
import sys

import pandas as pd

from train_models import DATA_DIR

HALF_LIVES = [0, 180, 365, 730]   # 0 = ağırlıksız

v1, pois = [], []
for hl in HALF_LIVES:
    env = dict(os.environ, HALF_LIFE_DAYS=0)
    for script in ("train_models.py", "poisson_models.py"):
        # geçmiş bir tarih -> tahmin satırı yok, final yeniden eğitim atlanır (hızlı)
        subprocess.run([sys.executable, script, "--date", "1900-01-01"], check=True, env=env)
    v1.append(pd.read_csv(DATA_DIR / "model_results.csv", encoding="utf-8-sig").assign(half_life=hl))
    pois.append(pd.read_csv(DATA_DIR / "poisson_results.csv", encoding="utf-8-sig").assign(half_life=hl))

v1, pois = pd.concat(v1), pd.concat(pois)
pois["valid_poisson"] = pois["valid_ev"] + pois["valid_dep"]

v1.to_csv(DATA_DIR / "scan_v1.csv", index=False, encoding="utf-8-sig")
pois.to_csv(DATA_DIR / "scan_poisson.csv", index=False, encoding="utf-8-sig")

pd.set_option("display.width", 200)
print("\nv1 | VALID logloss (seçim bunun üzerinden, düşük iyi)")
print(v1.pivot(index="hedef", columns="half_life", values="valid_logloss"))
print("\nv1 | TEST skill (yalnızca seçilen yarı ömrü doğrulamak için bak)")
print(v1.pivot(index="hedef", columns="half_life", values="skill"))
print("\nPoisson | VALID poisson nll (ev+dep, düşük iyi)")
print(pois.pivot(index="hedef", columns="half_life", values="valid_poisson"))
print("\nPoisson | TEST skill")
print(pois.pivot(index="hedef", columns="half_life", values="skill"))
