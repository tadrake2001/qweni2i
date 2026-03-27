#!/usr/bin/env python3
"""
Qwen Image Edit RunPod API Test Script
Calls /runsync with handler input and validates results.
"""

import os
import sys
import json
import base64
import argparse
from pathlib import Path
import uuid

def load_test_env():
    """Load test.env from project root."""
    env_path = Path(__file__).resolve().parent.parent / "test.env"
    if env_path.exists():
        with open(env_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.strip()
                    if v.startswith('"') and v.endswith('"'):
                        v = v[1:-1]
                    os.environ.setdefault(k, v)

load_test_env()

try:
    import requests
except ImportError:
    print("requests required: pip install requests")
    sys.exit(1)

def get_s3_config():
    """Read RunPod Network Volume S3 config from test.env or environment variables."""
    endpoint_url = os.getenv("url") or os.getenv("S3_ENDPOINT_URL")
    region = os.getenv("region") or os.getenv("S3_REGION")
    access_key_id = os.getenv("access_key_id") or os.getenv("S3_ACCESS_KEY_ID")
    secret_access_key = os.getenv("secret_access_key") or os.getenv("S3_SECRET_ACCESS_KEY")
    bucket_name = os.getenv("bucket_name") or os.getenv("S3_BUCKET_NAME")

    if not (endpoint_url and region and access_key_id and secret_access_key and bucket_name):
        return None

    return {
        "endpoint_url": endpoint_url.strip(),
        "region": region.strip(),
        "access_key_id": access_key_id.strip(),
        "secret_access_key": secret_access_key.strip(),
        "bucket_name": bucket_name.strip(),
    }

def encode_file_to_base64(file_path: str) -> str:
    """Encode file to base64 string."""
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    return base64.b64encode(p.read_bytes()).decode("utf-8")

def upload_to_runpod_s3(local_path: str, s3_key: str) -> str:
    """Upload to RunPod Network Volume S3, return worker-accessible path."""
    s3_cfg = get_s3_config()
    if not s3_cfg:
        raise RuntimeError("S3 config missing. Set url/region/access_key_id/bucket_name/secret_access_key in test.env")

    try:
        import boto3
        from botocore.client import Config
    except ImportError as e:
        raise RuntimeError("boto3 required for S3 upload: pip install boto3") from e

    client = boto3.client(
        "s3",
        endpoint_url=s3_cfg["endpoint_url"],
        aws_access_key_id=s3_cfg["access_key_id"],
        aws_secret_access_key=s3_cfg["secret_access_key"],
        region_name=s3_cfg["region"],
        config=Config(signature_version="s3v4"),
    )

    local_path_p = Path(local_path)
    if not local_path_p.exists():
        raise FileNotFoundError(f"File to upload not found: {local_path}")

    client.upload_file(str(local_path_p), s3_cfg["bucket_name"], s3_key)
    return f"/runpod-volume/{s3_key}"

def get_config():
    """Read API config from test.env or environment variables."""
    api_key = os.getenv("runpod_API_KEY") or os.getenv("RUNPOD_API_KEY")
    endpoint_id = os.getenv("qwen_image_edit") or os.getenv("RUNPOD_ENDPOINT_ID")
    if not api_key or not endpoint_id:
        print("Required env vars: runpod_API_KEY, qwen_image_edit (endpoint ID)")
        print("Set in test.env or export before running.")
        return None, None
    return api_key.strip(), endpoint_id.strip()

def run_sync(api_key: str, endpoint_id: str, input_payload: dict, timeout: int = 300):
    """Call RunPod /runsync. timeout is client wait time in seconds."""
    wait_ms = min(300000, max(60000, timeout * 1000))
    url = f"https://api.runpod.ai/v2/{endpoint_id}/runsync?wait={wait_ms}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {"input": input_payload}
    r = requests.post(url, json=body, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.json()

def main():
    parser = argparse.ArgumentParser(description="Qwen Image Edit API Test")
    parser.add_argument("--json", "-j", help="Input JSON file path (input object or full { \"input\": {...} })")
    parser.add_argument("--image-url", help="Test image URL")
    examples_dir = Path(__file__).resolve().parent / "examples"
    default_input = examples_dir / "input" / "test_input.png"
    default_out_dir = examples_dir / "output"
    parser.add_argument("--image-file", default=str(default_input), help=f"Test image file (default: {default_input})")
    parser.add_argument("--mode", choices=["url", "base64", "s3"], default="url", help="Input mode: url | base64 | s3")
    parser.add_argument("--all", action="store_true", help="Test both base64 and s3 modes sequentially")
    parser.add_argument("--prompt", default="add watercolor style, soft pastel tones", help="Edit prompt")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed")
    parser.add_argument("--steps", type=int, default=4, help="Sampling steps")
    parser.add_argument("--cfg", type=float, default=1.0, help="CFG scale")
    parser.add_argument("--timeout", type=int, default=300, help="Wait timeout in seconds (default 300)")
    parser.add_argument("--out", "-o", help="Output image path (default: examples/output/out_test.png)")
    args = parser.parse_args()

    if args.out is None:
        args.out = str(default_out_dir / "out_test.png")

    api_key, endpoint_id = get_config()
    if not api_key or not endpoint_id:
        sys.exit(1)

    def build_common():
        return {
            "prompt": args.prompt,
            "seed": args.seed,
            "steps": args.steps,
            "cfg": args.cfg,
        }

    def call_once(input_payload: dict, out_path: str | None):
        printable = dict(input_payload)
        for k in ["image_base64", "image_base64_2", "image_base64_3"]:
            if k in printable and isinstance(printable[k], str):
                printable[k] = f"<base64:{len(printable[k])} chars>"
        print("Input:", json.dumps(printable, indent=2, ensure_ascii=False))
        print("\nCalling RunPod runsync...")

        try:
            result = run_sync(api_key, endpoint_id, input_payload, timeout=args.timeout)
        except requests.exceptions.RequestException as e:
            print("Request failed:", e)
            if hasattr(e, "response") and e.response is not None:
                try:
                    print("Response body:", e.response.text[:800])
                except Exception:
                    pass
            return False

        status = result.get("status")
        output = result.get("output")

        print("\nStatus:", status)
        if output:
            if isinstance(output, dict) and "error" in output:
                print("Error:", output["error"])
                return False
            if isinstance(output, dict) and "image" in output:
                img_b64 = output["image"]
                print("image field found, length:", len(img_b64) if isinstance(img_b64, str) else "N/A")
                if out_path and img_b64:
                    raw = base64.b64decode(img_b64)
                    out_p = Path(out_path)
                    out_p.parent.mkdir(parents=True, exist_ok=True)
                    out_p.write_bytes(raw)
                    print("Saved:", out_path)
                return True
            print("Output (partial):", json.dumps(output, indent=2, ensure_ascii=False)[:1200])
        else:
            print("Full response:", json.dumps(result, indent=2, ensure_ascii=False)[:1800])

        if status == "IN_QUEUE" or status == "IN_PROGRESS":
            print("\n(Note) Job not completed yet. Worker may be cold starting. Try again or use async /run + /status.")
            return False
        if status != "COMPLETED":
            return False
        print("\nTest passed: output returned successfully.")
        return True

    if args.json:
        with open(args.json, encoding="utf-8") as f:
            data = json.load(f)
        input_payload = data.get("input", data)
        ok = call_once(input_payload, args.out)
        sys.exit(0 if ok else 1)

    if args.all:
        # Test 1: base64
        print("=== Test 1/2: base64 input ===")
        img_b64 = encode_file_to_base64(args.image_file)
        payload_b64 = build_common()
        payload_b64["image_base64"] = img_b64
        out1 = args.out
        ok1 = call_once(payload_b64, out1)

        # Test 2: s3 upload + image_path
        print("\n=== Test 2/2: S3 upload + image_path ===")
        ext = Path(args.image_file).suffix or ".png"
        s3_key = f"qwen_edit_tests/{uuid.uuid4().hex}{ext}"
        try:
            remote_path = upload_to_runpod_s3(args.image_file, s3_key)
            payload_s3 = build_common()
            payload_s3["image_path"] = remote_path
            out2 = None
            if args.out:
                p = Path(args.out)
                out2 = str(p.with_name(p.stem + "_s3" + p.suffix))
            ok2 = call_once(payload_s3, out2)
        except Exception as e:
            print("S3 test prep failed:", e)
            ok2 = False

        sys.exit(0 if (ok1 and ok2) else 1)

    # Single mode
    mode = args.mode
    if mode == "url":
        image_url = args.image_url or os.getenv("TEST_IMAGE_URL")
        if not image_url:
            print("--image-url or TEST_IMAGE_URL required (or use --json/--all)")
            sys.exit(1)
        input_payload = build_common()
        input_payload["image_url"] = image_url
        ok = call_once(input_payload, args.out)
        sys.exit(0 if ok else 1)

    if mode == "base64":
        img_b64 = encode_file_to_base64(args.image_file)
        input_payload = build_common()
        input_payload["image_base64"] = img_b64
        ok = call_once(input_payload, args.out)
        sys.exit(0 if ok else 1)

    if mode == "s3":
        ext = Path(args.image_file).suffix or ".png"
        s3_key = f"qwen_edit_tests/{uuid.uuid4().hex}{ext}"
        remote_path = upload_to_runpod_s3(args.image_file, s3_key)
        input_payload = build_common()
        input_payload["image_path"] = remote_path
        ok = call_once(input_payload, args.out)
        sys.exit(0 if ok else 1)

    print("Unsupported mode:", mode)
    sys.exit(1)

if __name__ == "__main__":
    main()