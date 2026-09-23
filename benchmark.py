"""Readable CLI for the reproducible JSON benchmark worker."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=["small", "medium", "large", "xl", "10B"], default="small")
parser.add_argument("--context-length", type=int, default=512)
parser.add_argument("--batch-size", type=int, default=4)
parser.add_argument("--mode", choices=["forward", "backward", "train"], default="train")
parser.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
parser.add_argument("--warmup", type=int, default=5)
parser.add_argument("--steps", type=int, default=10)
parser.add_argument("--compile", action="store_true")
parser.add_argument("--memory-history", action="store_true")
parser.add_argument("--checkpoint-segments", type=int, default=0)
parser.add_argument("--output", type=Path, required=True)
for name in ("d-model", "d-ff", "num-layers", "num-heads"):
    parser.add_argument("--" + name, type=int)
if __name__ == "__main__":
    args = vars(parser.parse_args())
    output = args.pop("output")
    cfg = {"name": "custom-model", "kind": "model", "size": args.pop("model"),
           "seq": args.pop("context_length"), "batch": args.pop("batch_size"),
           **{k: v for k, v in args.items() if v is not None}}
    subprocess.run([sys.executable, "-m", "cs336_systems.experiments", "--config", json.dumps(cfg),
                    "--output", str(output)], check=True)
