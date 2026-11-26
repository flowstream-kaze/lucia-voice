#!/usr/bin/env python3
"""Fetch the pinned Kokoro assets and prove them by SHA256. Stdlib only.

Assets land next to tts_server.py, which is the folder it loads them from.
Proxies are honoured through the usual HTTP(S)_PROXY variables; no credentials
are read or sent.
"""
import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

MANIFEST = Path(__file__).resolve().with_name('model-manifest.json')
CHUNK = 1024 * 1024


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(CHUNK), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dir', type=Path, default=Path(__file__).resolve().parent,
                        help='destination for the assets (default: beside this script)')
    parser.add_argument('--only', action='append',
                        help='asset name to fetch, repeatable (default: every required asset)')
    parser.add_argument('--include-optional', action='store_true')
    args = parser.parse_args()

    spec = json.loads(MANIFEST.read_text(encoding='utf-8'))
    items = list(spec['required'])
    if args.include_optional:
        items += list(spec.get('optional', []))
    if args.only:
        wanted = set(args.only)
        items = [item for item in items if item['name'] in wanted]
        unknown = wanted - {item['name'] for item in items}
        if unknown:
            parser.error('not in manifest: ' + ', '.join(sorted(unknown)))

    args.dir.mkdir(parents=True, exist_ok=True)
    for item in items:
        target = args.dir / item['name']
        if target.is_file() and target.stat().st_size == item['bytes'] and sha256(target) == item['sha256']:
            print(f'ok      {item["name"]} already verified')
            continue
        partial = target.with_name(target.name + '.part')
        print(f'get     {item["name"]} {item["bytes"]} bytes', flush=True)
        with urllib.request.urlopen(item['url'], timeout=60) as response, partial.open('wb') as out:
            if response.status != 200:
                raise SystemExit(f'HTTP {response.status} for {item["url"]}')
            while block := response.read(CHUNK):
                out.write(block)
        if partial.stat().st_size != item['bytes'] or sha256(partial) != item['sha256']:
            partial.unlink()
            raise SystemExit(f'verify failed: {item["name"]} - delete it and retry, or pin a new release')
        partial.replace(target)
        print(f'ok      {item["name"]} sha256 {item["sha256"][:12]}...', flush=True)

    missing = [item['name'] for item in spec['required'] if not (args.dir / item['name']).is_file()]
    if not missing:
        print(f'ready: {args.dir}')
    elif args.only:
        print(f'note: still absent from {args.dir}: ' + ', '.join(missing))
    else:
        raise SystemExit('still missing: ' + ', '.join(missing))


if __name__ == '__main__':
    main()
