import os
import argparse
import boto3
from botocore import UNSIGNED
from botocore.client import Config

BUCKET = "iai-robust-rearrangement"

# Released checkpoints: checkpoints/<kind>/<task>/<randomness>/actor_chkpt.pt
#   kind: bc (π_base, diffusion policy) | rppo (π_base + residual π_res)
TASKS = ["one_leg", "round_table", "lamp", "mug_rack", "factory_peg_hole"]
KINDS = ["bc", "rppo"]
RANDOMNESS = ["low", "med"]


def download_checkpoints(s3_client, prefixes, out_dir, dry_run=False):
    """
    Download every object under the given S3 prefixes, keeping the bucket layout
    below out_dir. Files that already exist with the same size are skipped.
    """
    paginator = s3_client.get_paginator("list_objects_v2")
    n_found = 0
    for prefix in prefixes:
        for result in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
            for obj in result.get("Contents", []):
                key, size = obj["Key"], obj["Size"]
                if key.endswith("/"):
                    continue
                n_found += 1
                local_path = os.path.join(out_dir, key)

                if os.path.exists(local_path) and os.path.getsize(local_path) == size:
                    print(f"Exists, skipping: {local_path}")
                    continue

                print(f"{'Would download' if dry_run else 'Downloading'}: "
                      f"s3://{BUCKET}/{key} ({size / 1e6:.1f} MB) -> {local_path}")
                if dry_run:
                    continue

                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                s3_client.download_file(BUCKET, key, local_path)

    if n_found == 0:
        print("No checkpoints found. Not every task has both randomness levels: "
              "mug_rack and factory_peg_hole only have 'low'.")


def main():
    parser = argparse.ArgumentParser(description="Download released ResiP checkpoints from S3.")
    parser.add_argument("--task", nargs="+", default=["one_leg"], choices=TASKS + ["all"])
    parser.add_argument("--kind", nargs="+", default=KINDS, choices=KINDS,
                        help="bc = base diffusion policy, rppo = base + residual policy")
    parser.add_argument("--randomness", nargs="+", default=RANDOMNESS, choices=RANDOMNESS)
    parser.add_argument("--out-dir", default=".",
                        help="Files are saved to <out-dir>/checkpoints/<kind>/<task>/<randomness>/actor_chkpt.pt")
    parser.add_argument("--dry-run", action="store_true", help="List what would be downloaded")
    args = parser.parse_args()

    tasks = TASKS if "all" in args.task else args.task
    prefixes = [
        f"checkpoints/{kind}/{task}/{rand}/"
        for kind in args.kind
        for task in tasks
        for rand in args.randomness
    ]

    # Create S3 client without requiring credentials (for public buckets)
    s3_client = boto3.client("s3", config=Config(signature_version=UNSIGNED))

    download_checkpoints(s3_client, prefixes, args.out_dir, dry_run=args.dry_run)
    print("Done.")


if __name__ == "__main__":
    main()
