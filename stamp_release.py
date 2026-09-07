#!/usr/bin/env python3
"""
Write the release identifier into Methods M21.

    python stamp_release.py v1.2-round4 <full 40-character commit hash>

WHY THIS IS A SCRIPT
--------------------
M21 names the tagged release that the reported run belongs to, and close-out
Section G requires that the commit hash be stated and postdate every fix. The
hash cannot be known until the commit exists, so the sentence has to be written
after the commit — which in round 3 was done by hand, on the repository copy
only. The copy that went forward for submission was the one written before the
push, so it still carried the unfilled placeholder, and it also missed the
reproducibility sentence the same edit added. That is a filing accident that a
script prevents and a convention does not.

This script is the last step before submission. It refuses to run on anything
that is not a real tag and a real 40-character hash, it refuses to leave a
placeholder in place, and it prints the resulting sentence so it can be read
before the file is sent.

Idempotent: re-stamping with the same values is a no-op; re-stamping with new
values replaces whatever identifier is currently there.
"""
import os
import re
import sys

from docx import Document

M_PATH = os.path.join('docs_out', 'Group 3_Methods Section.docx')
ANCHOR = 'the version identifier for that run is the tagged release'
PLACEHOLDER = '[RELEASE TAG AND COMMIT HASH TO BE INSERTED AT PUSH — see CHANGELOG]'
PATTERN = re.compile(
    re.escape(ANCHOR) + r'\s+(?:' + re.escape(PLACEHOLDER) + r'|\S+\s*\([0-9a-f]{7,40}\))')


def set_text(p, text):
    runs = p.runs
    if not runs:
        p.add_run(text)
        return p
    runs[0].text = text
    for r in runs[1:]:
        r._element.getparent().remove(r._element)
    return p


def main(tag, commit):
    if not re.fullmatch(r'[0-9a-f]{40}', commit):
        sys.exit('commit must be the full 40-character hash, not %r. '
                 '`git rev-parse HEAD` gives it.' % commit)
    if not re.fullmatch(r'v[0-9]+\.[0-9]+[A-Za-z0-9._-]*', tag):
        sys.exit('tag looks wrong: %r (expected something like v1.2-round4)' % tag)

    # M21 says the commit "postdates every fix described in this manuscript".
    # In round 3 it did not: the tag was cut before the M21 edit, so the released
    # tree held a different Methods file from the one submitted. Check it here
    # rather than trust it.
    try:
        import subprocess
        head = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True,
                              text=True, check=True).stdout.strip()
        tagged = subprocess.run(['git', 'rev-list', '-n1', tag], capture_output=True,
                                text=True).stdout.strip()
        if tagged and tagged != commit:
            sys.exit('tag %s points at %s, not %s. Re-tag, then stamp.'
                     % (tag, tagged[:12], commit[:12]))
        if head != commit:
            print('NOTE: HEAD is %s and you are stamping %s. That is correct only if '
                  'this stamp is being committed on top of %s.'
                  % (head[:12], commit[:12], commit[:12]))
    except (OSError, subprocess.CalledProcessError):
        print('NOTE: not a git checkout — the commit could not be checked.')

    doc = Document(M_PATH)
    for p in doc.paragraphs:
        if ANCHOR in p.text:
            new = PATTERN.sub('%s %s (%s)' % (ANCHOR, tag, commit), p.text)
            if new == p.text:
                sys.exit('M21 does not carry a recognisable release identifier or the '
                         'placeholder. Refusing to guess where to write it.')
            set_text(p, new)
            doc.save(M_PATH)
            sentence = [s for s in new.split('. ') if ANCHOR in s]
            print('stamped %s\n\n  ...%s.\n' % (M_PATH, sentence[0].split(', and ')[-1]))
            if PLACEHOLDER in new:
                sys.exit('placeholder survived the edit — do not submit this file')
            return 0
    sys.exit('M21 anchor not found in %s' % M_PATH)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit(__doc__.strip().splitlines()[2].strip())
    sys.exit(main(sys.argv[1], sys.argv[2]))
