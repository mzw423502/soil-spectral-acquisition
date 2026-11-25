"""Download source files locally from their original archives; never redistribute them."""
import argparse
import hashlib
import shutil
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen
from zipfile import ZipFile

RECORDS = {
    'neospectra': ('13122321', [
        ('Neospectra_WoodwellKSSL_soil+site+NIR.csv','ce131d7796de557b7f55ccb8a0b1f0e8','Neospectra_WoodwellKSSL_NIR.csv'),
        ('Neospectra_WoodwellKSSL_MIR.csv','4cdfde936aaaaf363d5ea193577de86f','Neospectra_WoodwellKSSL_MIR.csv'),
        ('Neospectra_database_column_names.csv','9e7bcda869f38d466c030e2e40461f59','Neospectra_database_column_names.csv')]),
    'icraf': ('23815431',[('OSSL_v1.3_ICRAF_ISRIC.zip','495dbde86d7ce37ba8427b371e22d832','OSSL_v1.3_ICRAF_ISRIC.zip')]),
    'lucas': ('23815431',[('OSSL_v1.3_LUCAS.zip','94bd2e63050b3d403cdd77d6ac76550e','OSSL_v1.3_LUCAS.zip')]),
    'agroscope': ('11204174',[
        ('00_metadata_Agroscope_SSL.txt','7093412f6fac3ced1635499fae457453','00_metadata_Agroscope_SSL.txt'),
        ('01_laboratory_analyses.csv','4e6421fb2be2baada3ca1a587086783c','01_laboratory_analyses.csv'),
        ('02_spectraA_psr.csv','2ce2ebc4e3609cbef4be57c886b0a69a','02_spectraA_psr.csv'),
        ('03_spectraA_neo.csv','729cea1408c73412a17c04da21abcd2b','03_spectraA_neo.csv')])
}
PARQUETS = [f'ossl_{x}_v1.3.parquet' for x in ('mir','visnir','soilsite','soillab')]

def digest(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f,'md5').hexdigest()

def fetch(url, target, expected):
    if target.exists():
        if digest(target) != expected:
            raise RuntimeError(f'Existing file has a different checksum: {target}; inspect or move it before retrying')
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + '.partial')
    with urlopen(url, timeout=120) as fin, partial.open('wb') as fout:
        shutil.copyfileobj(fin,fout,1024*1024)
    if digest(partial) != expected:
        raise RuntimeError(f'Checksum mismatch: {partial}; file has not been accepted')
    partial.replace(target)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument('--dataset', choices=['all', *RECORDS], default='all')
    ap.add_argument('--plan', action='store_true',help='Print source URLs and checksums without downloading')
    args = ap.parse_args()
    for name, (record, files) in RECORDS.items():
        if args.dataset not in ('all', name):
            continue
        folder = args.root / ('data/neospectra' if name=='neospectra' else 'audit/agro_data' if name=='agroscope' else f'manuscript_study/data/{name}')
        for original, md5, local in files:
            url = f'https://zenodo.org/records/{record}/files/{quote(original,safe="")}?download=1'
            print(name, url, md5, folder/local)
            if args.plan:
                continue
            fetch(url, folder/local, md5)
            if original.endswith('.zip'):
                with ZipFile(folder/local) as z:
                    for base in PARQUETS:
                        candidates = [n for n in z.namelist() if n.rsplit('/',1)[-1]==base]
                        if len(candidates)!=1:
                            raise RuntimeError(f'Expected one archive member {base}; found {candidates}. Inspect the source archive layout.')
                        # Extract only the known file to an explicit local path.
                        with z.open(candidates[0]) as fi, (folder/base).open('wb') as fo:
                            shutil.copyfileobj(fi,fo)

if __name__=='__main__':
    main()
