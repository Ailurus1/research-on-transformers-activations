import csv
from pathlib import Path

root = Path.cwd()
output_dir = root / "outputs" / "domain_effect"
summary_path = root / "experiment_domain_effect.csv"

rows = []
for domain_dir in sorted(p for p in output_dir.iterdir() if p.is_dir()):
    domain = domain_dir.name
    for model_dir in sorted(p for p in domain_dir.iterdir() if p.is_dir()):
        model = model_dir.name.replace("__", "/")
        acta_root = model_dir / "acta"
        outlier_any = False

        if acta_root.exists():
            candidates = sorted(acta_root.glob("*/acta_results.csv"))
            if candidates:
                latest = candidates[-1]
                with latest.open("r", encoding="utf-8", newline="") as f:
                    reader = csv.DictReader(f)
                    for rec in reader:
                        if str(rec.get("outliers", "")).strip().lower() == "true":
                            outlier_any = True
                            break

        rows.append(
            {"domain": domain, "model": model, "outliers": str(outlier_any).lower()}
        )

with summary_path.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["domain", "model", "outliers"])
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote: {summary_path}")
