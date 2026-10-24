"""Export committed source without repository history or hosting metadata."""
import argparse
import io
from pathlib import Path
import re
import subprocess
import tarfile
import zipfile


def export(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if subprocess.check_output(['git','status','--porcelain'], cwd=root):
        raise ValueError('Commit or set aside local changes before exporting')
    raw = subprocess.check_output(['git','archive','--format=tar','HEAD'], cwd=root)
    entries = []
    with tarfile.open(fileobj=io.BytesIO(raw)) as source:
        for entry in source.getmembers():
            if not entry.isfile() or entry.name.startswith('.github/'):
                continue
            if Path(entry.name).suffix in ('.api-config','.pt','.pth') or entry.name == '.env':
                raise ValueError('Private/generated file in export: '+entry.name)
            data = source.extractfile(entry).read()
            text = data.decode('utf-8', errors='ignore')
            if re.search(r'(?:ghp_|sk-proj-)[A-Za-z0-9_-]{20,}|(?:C:[/\\]Users[/\\]|/home/)[A-Za-z0-9_.-]+[/\\]', text):
                raise ValueError('Credential or local path in '+entry.name)
            entries.append((entry.name, data))
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in entries:
            info = zipfile.ZipInfo('RecRSI/'+name, date_time=(2026,1,1,0,0,0))
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info,data)
    return len(entries)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print('Exported',export(Path(__file__).resolve().parents[1],args.output),'files')
