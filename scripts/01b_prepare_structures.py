"""STEP 1b -- Join official DSSTox structures (from 00b) onto the ToxCast
chemical list, producing the final, RDKit-canonicalized chemicals.csv (the
ToxCast MySQL dump itself has no SMILES). Also resolves MEA workbook compound
names to DTXSID/SMILES wherever exactly one DSSTox chemical matches a name or
synonym, updating the MEA-to-CompTox mapping template used downstream by
06_prepare_mea_data.py and 08_build_knowledge_graph.py."""
import csv
import json
import zipfile
from collections import defaultdict
import pandas as pd
from rdkit import Chem
from core.paths import DATA_SOURCE,DATA_NORMALIZED,MEA_PROCESSED
from core.mysql_io import connect,DATABASE


def exact_name(value):
    """Normalize a chemical name for exact-match comparison: casefold and
    collapse whitespace, so trivial formatting differences don't block a match."""
    return ' '.join(str(value).strip().casefold().split())


def main():
    """CLI entry point: stream the (large) DSSTox CSV out of its zip in chunks,
    keep only rows that are either an existing ToxCast chemical or match an
    MEA compound name/synonym, validate each SMILES with RDKit, then write
    chemicals.csv plus the updated MEA mapping/audit/structure files."""
    DATA_NORMALIZED.mkdir(parents=True,exist_ok=True)
    identifiers=DATA_NORMALIZED/'chemical_identifiers.csv'
    if not identifiers.exists():
        with connect(DATABASE) as connection:
            with connection.cursor() as cur:
                cur.execute('SELECT dsstox_substance_id,chnm,casn FROM chemical WHERE dsstox_substance_id LIKE %s',('DTXSID%',))
                with identifiers.open('w') as out:
                    writer=csv.writer(out); writer.writerow(['DTXSID','preferred_name','CASRN']); writer.writerows(cur.fetchall())
    ids=set(pd.read_csv(identifiers).DTXSID.dropna())
    mapping_path=MEA_PROCESSED/'mea_to_comptox_mapping_template.csv'
    mapping=pd.read_csv(mapping_path,dtype=str).fillna('') if mapping_path.exists() else pd.DataFrame()
    names=set(mapping.base_compound_name.map(exact_name)) if not mapping.empty else set()
    matches=defaultdict(dict); rows=[]; invalid=0
    source=DATA_SOURCE/'dsstox/DSSTox_CCD_dump_12092025_CSVs.zip'
    with zipfile.ZipFile(source) as archive:
        with archive.open('DSSTox_CCD_dump_12092025/DSSToxCCDdump.csv') as handle:
            for chunk in pd.read_csv(handle,chunksize=25000,dtype=str):
                for r in chunk.itertuples(index=False):
                    aliases={exact_name(r.PREFERRED_NAME)}
                    if isinstance(r.IDENTIFIER,str): aliases.update(exact_name(x) for x in r.IDENTIFIER.split('|'))
                    matched=names & aliases
                    if r.DTXSID not in ids and not matched: continue
                    mol=Chem.MolFromSmiles(r.SMILES) if isinstance(r.SMILES,str) else None
                    if mol is None or mol.GetNumAtoms()==0: invalid+=1; continue
                    row={'DTXSID':r.DTXSID,'smiles':Chem.MolToSmiles(mol),
                         'preferred_name':r.PREFERRED_NAME,'CASRN':r.CASRN,'source':'EPA DSSTox December 2025'}
                    rows.append(row)
                    for name in matched: matches[name][r.DTXSID]=row
    chemicals=pd.DataFrame(rows).drop_duplicates('DTXSID')
    if chemicals.empty: raise ValueError('No valid ToxCast structures resolved')
    temp=DATA_NORMALIZED/'chemicals.csv.part'; chemicals.to_csv(temp,index=False); temp.replace(DATA_NORMALIZED/'chemicals.csv')
    audit=[]; structures=[]
    for index,r in mapping.iterrows():
        candidates=matches[exact_name(r.base_compound_name)]
        if not r.DTXSID and len(candidates)==1:
            resolved=next(iter(candidates.values()))
            mapping.loc[index,'DTXSID']=resolved['DTXSID']; mapping.loc[index,'SMILES']=resolved['smiles']
            mapping.loc[index,'mapping_source']='unique exact DSSTox preferred name/synonym; December 2025'
        sid=mapping.loc[index,'DTXSID']
        found=chemicals[chemicals.DTXSID.eq(sid)]
        if len(found)==1:
            structures.append({'compound_id':r.compound_id,'base_compound_name':r.base_compound_name,
                'DTXSID':sid,'SMILES':found.iloc[0].smiles,'source':'EPA DSSTox December 2025'})
        audit.append({'compound_id':r.compound_id,'candidate_ids':';'.join(sorted(candidates)),
            'DTXSID':sid,'status':'mapped' if sid else ('ambiguous' if candidates else 'unresolved')})
    if not mapping.empty:
        mapping.to_csv(mapping_path,index=False)
        pd.DataFrame(audit).to_csv(MEA_PROCESSED/'mea_mapping_audit.csv',index=False)
        pd.DataFrame(structures,columns=['compound_id','base_compound_name','DTXSID','SMILES','source']).to_csv(MEA_PROCESSED/'mea_structure_pubchem.csv',index=False)
    report={'structures':len(chemicals),'toxcast_identifiers':len(ids),'toxcast_with_structure':int(chemicals.DTXSID.isin(ids).sum()),
        'invalid_or_missing_structures':invalid,'mea_mapped':sum(x['status']=='mapped' for x in audit),'source':str(source)}
    (DATA_NORMALIZED/'structure_report.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))


if __name__=='__main__': main()
