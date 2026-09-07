#!/usr/bin/env python3
"""
Remove Word review comments from a .docx.

The Methods file carried one comment anchored on the title, left over from an
earlier round and by now out of date — it argued from a site count of UNRECORDED,
which has been 4 since the school identifier was recovered. A stale comment in a
submitted file is a defect whether or not anyone reads it, and python-docx has no
API for comments, so this operates on the package directly.

It removes the comment parts, their relationships, their content-type overrides,
and the anchors and references left behind in the document body. Runs that exist
only to carry a comment reference are removed; runs that also carry text are
kept and only the reference is taken out.

    python strip_doc_comments.py "docs_out/Group 3_Methods Section.docx" [...]

Idempotent: a file with no comments is reported and left byte-identical.
"""
import os
import shutil
import sys
import tempfile
import zipfile

from lxml import etree

W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
PR = '{http://schemas.openxmlformats.org/package/2006/relationships}'
CT = '{http://schemas.openxmlformats.org/package/2006/content-types}'

COMMENT_PARTS = ('word/comments.xml', 'word/commentsExtended.xml',
                 'word/commentsIds.xml', 'word/commentsExtensible.xml')
COMMENT_REL_TYPES = ('/comments', '/commentsExtended', '/commentsIds', '/commentsExtensible')


def strip(path):
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        blob = {n: z.read(n) for n in names}

    present = [n for n in names if n in COMMENT_PARTS]
    doc = etree.fromstring(blob['word/document.xml'])

    anchors = doc.findall('.//' + W + 'commentRangeStart') \
        + doc.findall('.//' + W + 'commentRangeEnd')
    refs = doc.findall('.//' + W + 'commentReference')

    if not present and not anchors and not refs:
        print('%s: no comments' % os.path.basename(path))
        return 0

    for el in anchors:
        el.getparent().remove(el)
    for ref in refs:
        run = ref.getparent()
        ref.getparent().remove(ref)
        # A run left holding nothing but run properties existed only to anchor the
        # comment; one that still carries text is part of the sentence and stays.
        if run is not None and run.tag == W + 'r' \
                and not any(c.tag != W + 'rPr' for c in run):
            run.getparent().remove(run)
    blob['word/document.xml'] = etree.tostring(doc, xml_declaration=True,
                                               encoding='UTF-8', standalone=True)

    rels = etree.fromstring(blob['word/_rels/document.xml.rels'])
    for rel in list(rels):
        if any(rel.get('Type', '').endswith(t) for t in COMMENT_REL_TYPES):
            rels.remove(rel)
    blob['word/_rels/document.xml.rels'] = etree.tostring(
        rels, xml_declaration=True, encoding='UTF-8', standalone=True)

    types = etree.fromstring(blob['[Content_Types].xml'])
    for ov in list(types):
        if ov.get('PartName', '').lstrip('/') in COMMENT_PARTS:
            types.remove(ov)
    blob['[Content_Types].xml'] = etree.tostring(types, xml_declaration=True,
                                                 encoding='UTF-8', standalone=True)

    keep = [n for n in names if n not in COMMENT_PARTS]
    fd, tmp = tempfile.mkstemp(suffix='.docx')
    os.close(fd)
    with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as out:
        for n in keep:
            out.writestr(n, blob[n])
    shutil.move(tmp, path)
    print('%s: removed %d comment part(s), %d anchor(s), %d reference(s)'
          % (os.path.basename(path), len(present), len(anchors), len(refs)))
    return len(refs) or len(present)


if __name__ == '__main__':
    targets = sys.argv[1:] or [
        os.path.join('docs_out', 'Group 3_Methods Section.docx'),
        os.path.join('docs_out', 'Group 3_Results and Analysis Section.docx')]
    for t in targets:
        strip(t)
