import wandb

ENTITY  = "tobias-nauen-dfki"
PROJECT = "eomt"
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".gif", ".webp")
MEDIA_PREFIXES = ("media/images/", "media/videos/")
DRY_RUN = False

api = wandb.Api()
runs = api.runs(f"{ENTITY}/{PROJECT}")

total_deleted = 0
total_kept = 0

for run_stub in runs:
    print(f"\n{'='*60}")
    
    # Re-fetch the full run object to populate _project_internal_id
    run = api.run(f"{ENTITY}/{PROJECT}/{run_stub.id}")
    print(f"Run: {run.name} ({run.id}) — state: {run.state}")

    image_files = [
        f for f in run.files()
        if f.name.endswith(IMAGE_EXTS) or any(f.name.startswith(p) for p in MEDIA_PREFIXES)
    ]

    if not image_files:
        print("  No image files found, skipping.")
        continue

    image_files.sort(key=lambda f: f.updated_at)

    newest = image_files[-1]
    to_delete = image_files[:-1]

    print(f"  Total image files : {len(image_files)}")
    print(f"  Keeping (newest)  : {newest.name}  [{newest.updated_at}]")
    print(f"  Deleting          : {len(to_delete)} files")

    for f in to_delete:
        if DRY_RUN:
            print(f"  [DRY RUN] Would delete: {f.name}")
        else:
            # print(f"  Deleting: {f.name}")
            f.delete()
        total_deleted += 1

    total_kept += 1

print(f"\n{'='*60}")
print(f"Done. Kept: {total_kept} | Deleted: {total_deleted}")
if DRY_RUN:
    print("⚠️  DRY_RUN=True — set to False to apply.")
