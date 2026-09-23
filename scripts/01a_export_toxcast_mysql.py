"""STEP 1a -- Export measured ToxCast/Tox21 endpoints from the curated project
MySQL database (comptox_invitrodb_v4_3) into data/normalized/bioactivity.csv
and chemical_identifiers.csv. This is the real source of bioactivity.csv used
by the rest of the pipeline (00a's own export is only a smaller first pass).

Reads via a server-side streaming cursor (SSCursor, see core/mysql_io.py) so
the ~2.9M-row join never has to sit fully in Python memory at once.
"""
import argparse
import csv
import json
import math
import os
import yaml
from core.mysql_io import connect,DATABASE,MYSQL_HOME
from core.paths import DATA_NORMALIZED,CONFIG


def main():
    """CLI entry point. Joins mc5 (hit calls) -> mc4 -> sample -> chemical and
    assay_component_endpoint -> assay_component -> assay -> assay_source to
    resolve each measurement's chemical and assay identity, converts AC50 to
    micromolar, and writes the result plus chemical_identifiers.csv."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    state=json.loads((MYSQL_HOME/'import_status.json').read_text())
    if state.get('status')!='complete': raise RuntimeError('SQL import is unfinished. Resume step 00a first.')
    connection=connect(DATABASE,streaming=True)
    threshold=yaml.safe_load(CONFIG.read_text())['data']['toxcast_active_threshold']
    DATA_NORMALIZED.mkdir(parents=True,exist_ok=True)
    with connection.cursor() as cur:
        cur.execute('SELECT name, description FROM mc5_model_type ORDER BY model_type')
        print('Model types:',cur.fetchall())
        if args.check: connection.close(); return
        cur.execute('SELECT dsstox_substance_id,chnm,casn FROM chemical WHERE dsstox_substance_id LIKE %s',('DTXSID%',))
        with (DATA_NORMALIZED/'chemical_identifiers.csv').open('w') as out:
            writer=csv.writer(out); writer.writerow(['DTXSID','preferred_name','CASRN']); writer.writerows(cur)
        # mc5_chid marks the representative series per chemical/endpoint.
        # tcplFit2 v4 concentrations (including ac50) are in regular units.
        sql='''SELECT c.dsstox_substance_id, src.assay_source_name,
          e.assay_component_endpoint_name, m.hitc, p.ac50, p.top,
          m.m5id,m.model_type,s.tested_conc_unit
        FROM mc5 m JOIN mc5_chid rep ON rep.m5id=m.m5id AND rep.chid_rep=1
        JOIN mc4 b ON b.m4id=m.m4id
        JOIN sample s ON s.spid=b.spid
        JOIN chemical c ON c.chid=s.chid
        JOIN assay_component_endpoint e ON e.aeid=m.aeid
        JOIN assay_component ac ON ac.acid=e.acid
        JOIN assay a ON a.aid=ac.aid
        JOIN assay_source src ON src.asid=a.asid
        LEFT JOIN (SELECT m5id, MAX(CASE WHEN hit_param='ac50' THEN hit_val END) ac50,
          MAX(CASE WHEN hit_param='top' THEN hit_val END) top
          FROM mc5_param GROUP BY m5id) p ON p.m5id=m.m5id
        WHERE c.dsstox_substance_id LIKE 'DTXSID%%' AND m.hitc BETWEEN -1 AND 1
          AND e.export_ready=1 AND e.data_usability=1'''
        cur.execute(sql)
        destination=DATA_NORMALIZED/'bioactivity.csv'; temporary=destination.with_suffix('.csv.part')
        count=0; unknown_units=0
        with temporary.open('w') as out:
            writer=csv.writer(out)
            writer.writerow(['DTXSID','program','assay','hitcall','ac50_uM','efficacy','signed_hitcall','m5id','model_type','concentration_unit'])
            for sid,program,assay,hit,ac50,top,m5id,model_type,unit in cur:
                factor={'um':1.,'µm':1.,'μm':1.,'micromolar':1.,'nm':.001,'mm':1000.,'m':1e6}.get(str(unit).lower().strip())
                active=hit>=threshold
                if factor is None: unknown_units+=1
                potency=ac50*factor if active and factor is not None and ac50 is not None and ac50>0 and math.isfinite(ac50) else None
                # Negative intended-direction hit calls become inactive BCE targets,
                # while the unmodified value is retained as signed_hitcall.
                writer.writerow([sid,program,assay,max(0.,hit),potency,top if active else None,hit,m5id,model_type,unit])
                count+=1
                if count%100000==0: print('Exported',count,flush=True)
        if count==0: raise ValueError('No bioactivity rows exported; inspect imported schema and units.')
        os.replace(temporary,destination)
    connection.close()
    (DATA_NORMALIZED/'mysql_export_report.json').write_text(json.dumps({'database':DATABASE,'rows':count,'unknown_unit_rows':unknown_units,'threshold':threshold,'source':state['source']},indent=2))
    print('Exported rows:',count,'; potency masked for unknown units:',unknown_units)


if __name__=='__main__': main()
