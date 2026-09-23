"""STEP 0b -- Download EPA's DSSTox chemical-structure dump (DTXSID -> SMILES,
name, CASRN). The ToxCast MySQL dump has chemical identifiers but no SMILES;
this file is what 01b_prepare_structures.py joins in to fill that gap.
File size is verified against a known-good value so a truncated/interrupted
download is never silently treated as complete."""
import json
import zipfile
import requests
from core.paths import DATA_SOURCE

FILE_ID='69529775e4b0731a616efc4b'
NAME='DSSTox_CCD_dump_12092025_CSVs.zip'
SIZE=289824966


def main():
    """Download the DSSTox zip if it's missing or the wrong size, then list its
    contents and record provenance (source URL, release, size) alongside it."""
    destination=DATA_SOURCE/'dsstox'/NAME
    destination.parent.mkdir(parents=True,exist_ok=True)
    url=f'https://clowder.edap-cluster.com/api/files/{FILE_ID}/blob'
    if not destination.exists() or destination.stat().st_size!=SIZE:
        part=destination.with_suffix('.zip.part')
        with requests.get(url,stream=True,timeout=(30,300)) as response:
            response.raise_for_status()
            with part.open('wb') as out:
                for chunk in response.iter_content(8*1024*1024): out.write(chunk)
        if part.stat().st_size!=SIZE: raise IOError('Incomplete DSSTox download; rerun.')
        part.replace(destination)
    with zipfile.ZipFile(destination) as archive:
        print('Downloaded:',destination)
        for item in archive.infolist(): print(item.filename,item.file_size)
    destination.with_suffix('.provenance.json').write_text(json.dumps({'url':url,'release':'December 2025','size':SIZE},indent=2))


if __name__=='__main__': main()
