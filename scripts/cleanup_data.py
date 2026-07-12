#!/usr/bin/env python3
"""
python scripts/cleanup_data.py --stats
scripts/cleanup_data.py
Veri klasörlerini temizlemek için yardımcı script.

Varsayılan olarak DRY-RUN modunda çalışır — hiçbir şey silmez, sadece ne
sileceğini gösterir. Gerçekten silmek için --confirm ekleyin.

Kullanım örnekleri:
    # Sadece durumu göster
    python scripts/cleanup_data.py --stats

    # Reddedilen adayları neler olduğunu göster (silmeden)
    python scripts/cleanup_data.py --remove-rejected

    # Reddedilenleri gerçekten sil
    python scripts/cleanup_data.py --remove-rejected --confirm

    # 30 günden eski ham arşiv klasörlerini sil
    python scripts/cleanup_data.py --remove-old-raw 30 --confirm

    # Etiketleme kuyruğunu TAMAMEN boşalt (pending dahil — dikkatli kullanın)
    python scripts/cleanup_data.py --wipe-queue --confirm

    # Eski eğitim çalışmalarını sil (sadece en son N tanesini tut)
    python scripts/cleanup_data.py --keep-last-runs 3 --confirm
"""
import argparse
import json
import shutil
import time
from pathlib import Path

DATA_DIR = Path("data")
RAW_CAPTURES_DIR = DATA_DIR / "raw_captures"
LABELING_QUEUE_DIR = DATA_DIR / "labeling_queue"
TRAINING_RUNS_DIR = DATA_DIR / "training_runs"


def show_stats():
    print("=== Veri durumu ===\n")

    # Labeling queue durumu
    status_counts = {"pending": 0, "labeled": 0, "approved": 0, "rejected": 0}
    total_size = 0
    if LABELING_QUEUE_DIR.exists():
        for meta_file in LABELING_QUEUE_DIR.glob("*.json"):
            if "_labels" in meta_file.name:
                continue
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                status = meta.get("status", "pending")
                status_counts[status] = status_counts.get(status, 0) + 1
            except (json.JSONDecodeError, KeyError):
                pass
        total_size = sum(f.stat().st_size for f in LABELING_QUEUE_DIR.rglob("*") if f.is_file())

    print(f"labeling_queue/  ({total_size / 1e6:.1f} MB)")
    for status, count in status_counts.items():
        print(f"  {status}: {count}")

    # Raw captures
    if RAW_CAPTURES_DIR.exists():
        day_dirs = sorted([d for d in RAW_CAPTURES_DIR.iterdir() if d.is_dir()])
        raw_size = sum(f.stat().st_size for f in RAW_CAPTURES_DIR.rglob("*") if f.is_file())
        print(f"\nraw_captures/  ({raw_size / 1e6:.1f} MB, {len(day_dirs)} günlük klasör)")
        if day_dirs:
            print(f"  en eski: {day_dirs[0].name}  en yeni: {day_dirs[-1].name}")

    # Training runs
    if TRAINING_RUNS_DIR.exists():
        runs = sorted([d for d in TRAINING_RUNS_DIR.iterdir() if d.is_dir()])
        runs_size = sum(f.stat().st_size for f in TRAINING_RUNS_DIR.rglob("*") if f.is_file())
        print(f"\ntraining_runs/  ({runs_size / 1e6:.1f} MB, {len(runs)} çalışma)")
        for r in runs:
            print(f"  {r.name}")


def remove_rejected(confirm: bool):
    if not LABELING_QUEUE_DIR.exists():
        print("labeling_queue/ bulunamadı.")
        return

    to_delete = []
    for meta_file in LABELING_QUEUE_DIR.glob("*.json"):
        if "_labels" in meta_file.name:
            continue
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, KeyError):
            continue
        if meta.get("status") == "rejected":
            sample_id = meta["id"]
            to_delete.append(sample_id)

    print(f"{len(to_delete)} reddedilmiş aday bulundu.")
    for sample_id in to_delete:
        for suffix in [".png", ".json", "_labels.json"]:
            f = LABELING_QUEUE_DIR / f"{sample_id}{suffix}"
            if f.exists():
                print(f"  {'[SİLİNECEK]' if not confirm else '[SİLİNDİ]'} {f}")
                if confirm:
                    f.unlink()

    if not confirm and to_delete:
        print("\nGerçekten silmek için --confirm ekleyin.")


def remove_old_raw(days: int, confirm: bool):
    if not RAW_CAPTURES_DIR.exists():
        print("raw_captures/ bulunamadı.")
        return

    cutoff = time.time() - (days * 86400)
    to_delete = []
    for day_dir in RAW_CAPTURES_DIR.iterdir():
        if not day_dir.is_dir():
            continue
        try:
            dir_time = time.mktime(time.strptime(day_dir.name, "%Y-%m-%d"))
        except ValueError:
            continue
        if dir_time < cutoff:
            to_delete.append(day_dir)

    print(f"{len(to_delete)} klasör {days} günden eski.")
    for d in to_delete:
        print(f"  {'[SİLİNECEK]' if not confirm else '[SİLİNDİ]'} {d}")
        if confirm:
            shutil.rmtree(d)

    if not confirm and to_delete:
        print("\nGerçekten silmek için --confirm ekleyin.")


def wipe_queue(confirm: bool):
    if not LABELING_QUEUE_DIR.exists():
        print("labeling_queue/ bulunamadı.")
        return
    files = list(LABELING_QUEUE_DIR.iterdir())
    print(f"labeling_queue/ içinde {len(files)} dosya var (pending dahil hepsi silinecek).")
    if confirm:
        for f in files:
            f.unlink()
        print("Silindi.")
    else:
        print("Gerçekten silmek için --confirm ekleyin.")


def keep_last_runs(n: int, confirm: bool):
    if not TRAINING_RUNS_DIR.exists():
        print("training_runs/ bulunamadı.")
        return
    runs = sorted(
        [d for d in TRAINING_RUNS_DIR.iterdir() if d.is_dir()],
        key=lambda d: d.stat().st_mtime,
    )
    to_delete = runs[:-n] if n > 0 else runs
    print(f"{len(runs)} çalışma bulundu, en son {n} tanesi tutulacak, {len(to_delete)} tanesi silinecek.")
    for d in to_delete:
        size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 1e6
        print(f"  {'[SİLİNECEK]' if not confirm else '[SİLİNDİ]'} {d}  ({size:.1f} MB)")
        if confirm:
            shutil.rmtree(d)

    if not confirm and to_delete:
        print("\nGerçekten silmek için --confirm ekleyin.")


def main():
    parser = argparse.ArgumentParser(description="Vitra veri temizlik aracı")
    parser.add_argument("--stats", action="store_true", help="Durum özetini göster")
    parser.add_argument("--remove-rejected", action="store_true", help="Reddedilmiş adayları temizle")
    parser.add_argument("--remove-old-raw", type=int, metavar="GUN", help="N günden eski raw_captures klasörlerini sil")
    parser.add_argument("--wipe-queue", action="store_true", help="labeling_queue'yu TAMAMEN boşalt (dikkat!)")
    parser.add_argument("--keep-last-runs", type=int, metavar="N", help="training_runs'ta sadece en son N çalışmayı tut")
    parser.add_argument("--confirm", action="store_true", help="Gerçekten sil (verilmezse sadece gösterir)")
    args = parser.parse_args()

    if args.stats or not any([args.remove_rejected, args.remove_old_raw, args.wipe_queue, args.keep_last_runs]):
        show_stats()

    if args.remove_rejected:
        remove_rejected(args.confirm)
    if args.remove_old_raw is not None:
        remove_old_raw(args.remove_old_raw, args.confirm)
    if args.wipe_queue:
        wipe_queue(args.confirm)
    if args.keep_last_runs is not None:
        keep_last_runs(args.keep_last_runs, args.confirm)


if __name__ == "__main__":
    main()