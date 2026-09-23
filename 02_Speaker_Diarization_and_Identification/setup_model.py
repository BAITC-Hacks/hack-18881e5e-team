"""Download the official quantized model and portable CPU runtime (stdlib only)."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / 'models' / 'Qwen3-8B-Q4_K_M.gguf'
MODEL_SHA = 'd98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785'
MODEL_URL = 'https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/7c41481f57cb95916b40956ab2f0b139b296d974/Qwen3-8B-Q4_K_M.gguf'
RELEASE = 'b11125'

def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def download(url, target, sha=None):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and (not sha or digest(target) == sha):
        print(f'Already installed: {target.name}', flush=True)
        return
    part = target.with_suffix(target.suffix + '.part')
    offset = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(url, headers={'Range': f'bytes={offset}-'} if offset else {})
    with urllib.request.urlopen(req, timeout=120) as response:
        if response.status != 206:
            offset = 0
        total = offset + int(response.headers.get('Content-Length', 0))
        done = offset
        reported = done // (250 * 1024**2)
        with part.open('ab' if offset else 'wb') as out:
            while block := response.read(4 * 1024**2):
                out.write(block)
                done += len(block)
                if done // (250 * 1024**2) > reported:
                    reported = done // (250 * 1024**2)
                    print(f'{target.name}: {done / 1e9:.2f}/{total / 1e9:.2f} GB', flush=True)
    if sha and digest(part) != sha:
        raise RuntimeError(f'SHA256 mismatch: {part}')
    part.replace(target)
    print(f'Download verified: {target.name}', flush=True)

def runtime():
    url = f'https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/{RELEASE}'
    with urllib.request.urlopen(url, timeout=60) as r:
        release = json.load(r)
    asset = next(a for a in release['assets'] if a['name'] == f'llama-{RELEASE}-bin-win-cpu-x64.zip')
    archive = ROOT / 'runtime' / asset['name']
    sha = (asset.get('digest') or '').removeprefix('sha256:') or None
    download(asset['browser_download_url'], archive, sha)
    with zipfile.ZipFile(archive) as z:
        z.extractall(ROOT / 'runtime')
    print('llama.cpp runtime ready', flush=True)

if __name__ == '__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(download, MODEL_URL, MODEL, MODEL_SHA), pool.submit(runtime)]
        for f in futures:
            f.result()
