"""Download the pinned GPT-OSS 120B GGUF with resumable HTTP ranges."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

REPO = 'ggml-org/gpt-oss-120b-GGUF'
REVISION = '238abdd290bb874b90a5da1b4549881b7d05c091'
FILENAME = 'gpt-oss-120b-MXFP4.gguf'
SIZE = 63387346208
SHA256 = '582bd40f6886200101f4c4ed9f25f3fe80cc14c86e9e2b37746cd8904a0c622d'
DEFAULT_DIR = Path(__file__).resolve().parent / 'models' / 'gpt-oss-120b-GGUF'
CHUNK = 64 << 20


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while block := f.read(8 << 20):
            h.update(block)
    return h.hexdigest()


def download(directory=DEFAULT_DIR, workers=8):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / FILENAME
    if target.exists():
        if target.stat().st_size == SIZE and digest(target) == SHA256:
            print(f'Already verified: {target}', flush=True)
            return target
        raise RuntimeError(f'Existing model failed verification: {target}')
    part = target.with_suffix('.gguf.part')
    state_path = target.with_suffix('.gguf.progress.json')
    state = json.loads(state_path.read_text()) if state_path.exists() and part.exists() else {}
    if state and state.get('sha256') != SHA256:
        raise RuntimeError('Resume metadata belongs to a different model')
    done = set(state.get('chunks', []))
    count = (SIZE + CHUNK - 1) // CHUNK
    with part.open('ab') as f:
        f.truncate(SIZE)
    url = f'https://huggingface.co/{REPO}/resolve/{REVISION}/{FILENAME}'

    def fetch(index):
        lo, hi = index * CHUNK, min((index + 1) * CHUNK, SIZE) - 1
        expected_range = f'bytes {lo}-{hi}/{SIZE}'
        for attempt in range(5):
            try:
                with requests.get(url + f'?download=true&part={index}',
                                  headers={'Range': f'bytes={lo}-{hi}', 'Accept-Encoding': 'identity'},
                                  stream=True, timeout=(30, 90)) as response:
                    response.raise_for_status()
                    if response.status_code != 206 or response.headers.get('Content-Range') != expected_range:
                        raise RuntimeError(f'Incorrect range response for chunk {index}')
                    pos = lo
                    with part.open('r+b') as f:
                        f.seek(lo)
                        for block in response.iter_content(1 << 20):
                            if pos + len(block) > hi + 1:
                                raise RuntimeError('Range response exceeds requested bytes')
                            f.write(block)
                            pos += len(block)
                    if pos != hi + 1:
                        raise RuntimeError(f'Incomplete chunk {index}')
                return index
            except (requests.RequestException, RuntimeError):
                if attempt == 4:
                    raise
                time.sleep(min(2 ** attempt, 10))

    start = time.perf_counter()
    initial = len(done)
    print(f'Downloading {SIZE / 1e9:.1f} GB; resuming {len(done)}/{count} chunks', flush=True)
    last_report = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch, i) for i in range(count) if i not in done]
        for future in as_completed(futures):
            done.add(future.result())
            state_path.write_text(json.dumps({'sha256': SHA256, 'chunks': sorted(done)}))
            elapsed = time.perf_counter() - start
            if elapsed - last_report >= 15 or len(done) == count:
                rate = (len(done) - initial) * CHUNK / max(elapsed, 1) / 1e6
                print(f'{len(done)}/{count} chunks ({min(len(done) * CHUNK, SIZE) / 1e9:.1f} GB), '
                      f'{rate:.1f} MB/s', flush=True)
                last_report = elapsed
    print('Verifying SHA-256 ...', flush=True)
    if digest(part) != SHA256:
        raise RuntimeError('SHA-256 mismatch; model was not activated')
    part.replace(target)
    print(f'Verified: {target}', flush=True)
    return target


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--directory', type=Path, default=DEFAULT_DIR)
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()
    if args.workers < 1:
        ap.error('--workers must be positive')
    download(args.directory, args.workers)
