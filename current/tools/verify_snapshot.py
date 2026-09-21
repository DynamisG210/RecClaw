"""Verify the published subset, syntax, JSON and maintained documentation links."""
from pathlib import Path
import ast
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / 'SOURCE_MANIFEST.json').read_bytes())
    expected = set()
    for row in manifest['files']:
        path = ROOT / row['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256'], row['path']
        if not row['changes']:
            assert row['sha256'] == row['source_sha256'], row['path']
        expected.add(row['path'])
    actual = {p.relative_to(ROOT).as_posix() for d in ('runtime', 'variants')
              for p in (ROOT / d).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    assert actual == expected, sorted(actual ^ expected)
    counts = {'source_files': len(expected), 'python': 0, 'json': 0, 'doc_links': 0}
    for path in ROOT.rglob('*'):
        if not path.is_file() or '__pycache__' in path.parts:
            continue
        if path.suffix == '.py':
            ast.parse(path.read_bytes(), filename=str(path)); counts['python'] += 1
        elif path.suffix == '.json':
            json.loads(path.read_bytes()); counts['json'] += 1
    docs = [ROOT / 'README.md', ROOT / 'THIRD_PARTY.md', *sorted((ROOT / 'docs').glob('*.md'))]
    for path in docs:
        text = path.read_text(encoding='utf-8')
        assert sum(line.startswith('```') for line in text.splitlines()) % 2 == 0, path
        for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)', text):
            if '://' in target or target.startswith('#'):
                continue
            assert (path.parent / target.split('#')[0]).exists(), (path, target)
            counts['doc_links'] += 1
    # Checks only publication inputs, not historic files elsewhere in the repo.
    suspicious = re.compile(r'\b(?:sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{20,})\b|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----')
    for name in expected:
        path = ROOT / name
        text = path.read_text(encoding='utf-8')
        assert not suspicious.search(text), 'Possible secret in ' + name
        assert not any(value in text for value in ('/NAS2020/', '/ssd/tingrangan/', '/Users/gtrho/')), 'Private locator in ' + name
    print(json.dumps(dict(status='PASS', **counts), indent=2))


if __name__ == '__main__':
    main()
