#!/usr/bin/env python3
"""
Regenerate MANIFEST.sha256.

The manifest is what lets a reader confirm that the tree they cloned is the tree
that produced the reported numbers. It was maintained by hand across rounds 2 and
3, which is why it went stale the moment a file was added or removed; the round-3
release edit had to touch it separately, and the round-4 cleanup would have left
thirteen entries pointing at files that no longer exist.

The rule is simple and stated here so it can be checked rather than inferred:
every file git tracks, except the manifest itself and git's own metadata files,
sorted by path, as `sha256  path`.

    python make_manifest.py            # rewrite MANIFEST.sha256
    python make_manifest.py --check    # verify the tree against it, exit 1 on drift
"""
import argparse
import hashlib
import os
import subprocess
import sys

MANIFEST = 'MANIFEST.sha256'
EXCLUDE = {MANIFEST, '.gitattributes'}


def tracked():
    out = subprocess.run(['git', 'ls-files', '-z'], capture_output=True, text=True, check=True)
    paths = [p for p in out.stdout.split('\0') if p]
    return sorted(p for p in paths if p not in EXCLUDE and os.path.exists(p))


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args()

    files = tracked()
    if a.check:
        if not os.path.exists(MANIFEST):
            sys.exit('%s does not exist' % MANIFEST)
        listed, bad, missing = {}, [], []
        for line in open(MANIFEST):
            line = line.strip()
            if not line:
                continue
            digest, path = line.split(None, 1)
            listed[path] = digest
        for path, digest in sorted(listed.items()):
            if not os.path.exists(path):
                missing.append(path)
            elif sha(path) != digest:
                bad.append(path)
        extra = [p for p in files if p not in listed]
        for p in missing:
            print('MISSING   %s' % p)
        for p in bad:
            print('MISMATCH  %s' % p)
        for p in extra:
            print('UNLISTED  %s' % p)
        print('%d listed | %d verified | %d mismatched | %d missing | %d unlisted'
              % (len(listed), len(listed) - len(bad) - len(missing), len(bad),
                 len(missing), len(extra)))
        return 1 if (bad or missing or extra) else 0

    with open(MANIFEST, 'w') as fh:
        for path in files:
            fh.write('%s  %s\n' % (sha(path), path))
    print('wrote %s: %d files' % (MANIFEST, len(files)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
