import wandb
import pandas as pd

ENTITY  = "tobias-nauen-dfki"
PROJECT = "eomt"
METRIC  = "metrics/val_iou_all"

RUN_IDS = [
    "fu2or07l",
    "abc123xy",
    # add more IDs here...
]

api = wandb.Api()
results = []

for run_id in RUN_IDS:
    run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")

    history = run.history(keys=[METRIC], pandas=True)

    if METRIC not in history.columns:
        print(f"[WARN] '{METRIC}' not found in run {run_id} ({run.name})")
        continue

    max_iou = history[METRIC].max()
    results.append({
        "run_id":        run_id,
        "run_name":      run.name,
        "max_val_iou":   max_iou,
    })
    print(f"✓ {run.name} ({run_id}) — max IoU: {max_iou:.4f}")

df = pd.DataFrame(results)
print("\n" + df.to_string(index=False))
df.to_csv("val_iou_max.csv", index=False)
print("\nSaved to val_iou_max.csv")
