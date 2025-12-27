#!/usr/bin/env python3
"""Split pool into train/test once and move test rows into `test_data` table.
Usage:
  python scripts/split_pool.py --fraction 0.2 --seed 42 [--yes]
"""
import argparse
from data_utils.database import database


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fraction", type=float, default=0.2, help="Fraction to move to test_data (0-1)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--yes", action="store_true", help="Don't ask for confirmation")
    args = p.parse_args()

    # Count eligible pool rows
    with database.get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM pool WHERE description IS NOT NULL;")
            total = cur.fetchone()[0]

    k = int(total * args.fraction)
    print(f"Pool total rows with description: {total}")
    print(f"Will move {k} rows ({args.fraction*100:.1f}%) to test_data")

    if k <= 0:
        print("Nothing to move. Exiting.")
        return

    if not args.yes:
        proceed = input("Proceed with split and delete from pool? (y/N): ")
        if proceed.strip().lower() != 'y':
            print("Aborted by user.")
            return

    res = database.split_pool_to_test(fraction=args.fraction, seed=args.seed)
    print(f"Moved {res['moved_count']} rows. Sample ids: {res['moved_ids'][:10]}")


if __name__ == '__main__':
    main()
