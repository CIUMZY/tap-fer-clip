"""Phase 5 step 1: download the OpenAI ViT-L-14 checkpoint used by features_vitl14."""
import hashlib, json, sys, time, urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
OUT_DIR = Path('D:/ResearchVault/99system/models/open_clip')
URL = ('https://openaipublic.azureedge.net/clip/models/'
       'b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt')
EXPECTED = 'b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836'
DEST = OUT_DIR / 'ViT-L-14.pt'
REPORT = Path('D:/ResearchVault/07code/rb-tta-fer-fg2027/results/token_module_tailor_20261006/phase5_download.json')


def sha_of(p):
    h = hashlib.sha256()
    with p.open('rb') as fh:
        for b in iter(lambda: fh.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if DEST.exists():
        got = sha_of(DEST)
        rec = {'url': URL, 'expected_sha256': EXPECTED, 'actual_sha256': got,
               'bytes': DEST.stat().st_size, 'action': 'already present, verified',
               'sha256_matches': got == EXPECTED}
        REPORT.write_text(json.dumps(rec, indent=2), encoding='utf-8')
        print(json.dumps(rec, indent=2))
        return 0 if got == EXPECTED else 1
    part = DEST.with_suffix('.pt.part')
    last = None
    for attempt in range(1, 4):
        try:
            t0 = time.perf_counter()
            h = hashlib.sha256()
            done = 0
            req = urllib.request.Request(URL, headers={'User-Agent': 'open_clip-fetch'})
            with urllib.request.urlopen(req, timeout=60) as resp, part.open('wb') as fh:
                total = int(resp.headers.get('Content-Length') or 0)
                while True:
                    chunk = resp.read(1 << 22)
                    if not chunk:
                        break
                    fh.write(chunk)
                    h.update(chunk)
                    done += len(chunk)
                    if done // (200 << 20) != (done - len(chunk)) // (200 << 20):
                        print('  %.0f MB / %.0f MB (%.1fs)'
                              % (done / 1e6, total / 1e6, time.perf_counter() - t0), flush=True)
            got = h.hexdigest()
            secs = round(time.perf_counter() - t0, 1)
            ok = (got == EXPECTED)
            rec = {'url': URL, 'expected_sha256': EXPECTED, 'actual_sha256': got,
                   'bytes': part.stat().st_size, 'seconds': secs, 'attempt': attempt,
                   'sha256_matches': ok,
                   'note': 'URL path hash equals the file sha256 (same convention verified on ViT-B-16)'}
            if ok:
                part.replace(DEST)
                rec['action'] = 'downloaded and verified, renamed to ViT-L-14.pt'
                REPORT.write_text(json.dumps(rec, indent=2), encoding='utf-8')
                print(json.dumps(rec, indent=2), flush=True)
                return 0
            rec['action'] = 'sha256 mismatch, partial kept for inspection'
            print(json.dumps(rec, indent=2), flush=True)
            last = 1
        except Exception as e:
            print('attempt %d failed: %s %s' % (attempt, type(e).__name__, e), flush=True)
            last = 2
            time.sleep(3)
    REPORT.write_text(json.dumps({'url': URL, 'expected_sha256': EXPECTED, 'failed': True,
                                  'returncode': last}, indent=2), encoding='utf-8')
    return last


if __name__ == '__main__':
    raise SystemExit(main())
