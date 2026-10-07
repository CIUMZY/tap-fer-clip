"""Phase 5 step 1 (v2): resumable download of the OpenAI ViT-L-14 checkpoint.

v1 streamed in one shot and the connection was closed early (55 MB, then 134 MB of about
1.7 GB). v2 resumes with HTTP Range requests, checks the byte count reached Content-Length
before hashing, and only then renames the part file.
"""
import hashlib, json, sys, time, urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
OUT_DIR = Path('D:/ResearchVault/99system/models/open_clip')
URL = ('https://openaipublic.azureedge.net/clip/models/'
       'b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt')
EXPECTED = 'b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836'
DEST = OUT_DIR / 'ViT-L-14.pt'
PART = OUT_DIR / 'ViT-L-14.pt.part'
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
        print(json.dumps(rec, indent=2), flush=True)
        return 0 if got == EXPECTED else 1

    t0 = time.perf_counter()
    total = None
    rounds = 0
    history = []
    while rounds < 200:
        rounds += 1
        offset = PART.stat().st_size if PART.exists() else 0
        headers = {'User-Agent': 'open-clip-fetch'}
        if offset:
            headers['Range'] = 'bytes=%d-' % offset
        try:
            req = urllib.request.Request(URL, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as resp:
                status = resp.status
                if offset and status != 206:
                    print('server ignored Range (status %s); restarting' % status, flush=True)
                    PART.unlink(missing_ok=True)
                    offset = 0
                    continue
                if total is None:
                    if status == 206:
                        total = int(resp.headers['Content-Range'].split('/')[-1])
                    else:
                        total = int(resp.headers['Content-Length'])
                mode = 'ab' if offset else 'wb'
                with PART.open(mode) as fh:
                    while True:
                        chunk = resp.read(1 << 22)
                        if not chunk:
                            break
                        fh.write(chunk)
            size = PART.stat().st_size
            history.append({'round': rounds, 'offset': offset, 'size': size, 'total': total})
            print('  round %d: %d / %d bytes (%.1fs)' % (rounds, size, total, time.perf_counter() - t0), flush=True)
            if size >= total:
                break
        except Exception as e:
            history.append({'round': rounds, 'offset': offset,
                            'error': '%s %s' % (type(e).__name__, e)})
            print('  round %d error: %s %s' % (rounds, type(e).__name__, e), flush=True)
            time.sleep(2)

    size = PART.stat().st_size if PART.exists() else 0
    rec = {'url': URL, 'expected_sha256': EXPECTED, 'bytes': size, 'content_length': total,
           'rounds': rounds, 'history': history[-6:], 'seconds': round(time.perf_counter() - t0, 1)}
    if total is None or size < total:
        rec.update({'sha256_matches': False, 'action': 'incomplete download',
                    'reason': 'connection closed before Content-Length was reached'})
        REPORT.write_text(json.dumps(rec, indent=2), encoding='utf-8')
        print(json.dumps(rec, indent=2), flush=True)
        return 3
    got = sha_of(PART)
    rec['actual_sha256'] = got
    rec['sha256_matches'] = (got == EXPECTED)
    if got == EXPECTED:
        PART.replace(DEST)
        rec['action'] = 'downloaded, length-complete and sha256-verified; renamed to ViT-L-14.pt'
        REPORT.write_text(json.dumps(rec, indent=2), encoding='utf-8')
        print(json.dumps(rec, indent=2), flush=True)
        return 0
    rec['action'] = 'length-complete but sha256 mismatch'
    REPORT.write_text(json.dumps(rec, indent=2), encoding='utf-8')
    print(json.dumps(rec, indent=2), flush=True)
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
