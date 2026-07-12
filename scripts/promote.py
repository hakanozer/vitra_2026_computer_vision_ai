#!/usr/bin/env python3
"""
scripts/promote.py
Modeli production'a geçirme ve rollback CLI scripti.
Çalıştırma (promote): python scripts/promote.py promote --version v20240115_001
Çalıştırma (rollback): python scripts/promote.py rollback
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.registry.model_promoter import ModelPromoter


def main():
    parser = argparse.ArgumentParser(description="Model promote/rollback")
    subparsers = parser.add_subparsers(dest="command")

    promote_p = subparsers.add_parser("promote", help="Modeli production'a geçir")
    promote_p.add_argument("--version", required=True, help="Registry versiyon adı")

    subparsers.add_parser("rollback", help="Önceki modele geri dön")

    subparsers.add_parser("status", help="Mevcut production modelini göster")

    args = parser.parse_args()
    promoter = ModelPromoter()

    if args.command == "promote":
        success = promoter.promote(args.version)
        if success:
            print(json.dumps({"status": "promoted", "version": args.version}))
        else:
            print(json.dumps({"status": "failed"}), file=sys.stderr)
            sys.exit(1)
    elif args.command == "rollback":
        success = promoter.rollback()
        if success:
            print(json.dumps({"status": "rolled_back"}))
        else:
            print(json.dumps({"status": "no_previous_model"}), file=sys.stderr)
            sys.exit(1)
    elif args.command == "status":
        current = promoter.get_current_production()
        print(json.dumps(current or {"status": "no_production_model"}, indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
