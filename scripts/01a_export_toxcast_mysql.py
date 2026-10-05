"""Step 1a - Export ToxCast/Tox21 measurements from MySQL

Exports the measured ToxCast/Tox21 endpoints from the curated MySQL database (`comptox_invitrodb_v4_3`) to
data/normalized/bioactivity.csv, with a matching chemical_identifiers.csv. This is the real source of
`bioactivity.csv` for the rest of the pipeline (step 00a only writes a smaller first pass).

How the export works:
  * measurements come from table mc5 (curve-fit results); only the representative curve per chemical and
    endpoint is used, restricted to usable, export-ready endpoints;
  * mc5 hit calls are continuous in [-1, 1]; values are kept in `signed_hitcall`, and `hitcall` clips negatives
    to 0 (a curve in the unintended direction counts as inactive);
  * AC50 is converted to micromolar from the sample's concentration unit; it is kept only for active calls
    (hit call >= the threshold in config.yaml), and masked when the unit is unknown;
  * rows are streamed from a server-side cursor, so millions of rows never sit in Python memory at once.

Connection: unix socket at data/mysql/mysql.sock by default. Override with the environment variables
COMPTOX_MYSQL_SOCKET, COMPTOX_MYSQL_USER and COMPTOX_MYSQL_PASSWORD (no password is stored in this repo).

Output: data/normalized/bioactivity.csv, chemical_identifiers.csv, mysql_export_report.json
Run   : python scripts/01a_export_toxcast_mysql.py --check
        python scripts/01a_export_toxcast_mysql.py
"""

# %%
import argparse
import csv
import json
import math
import os
from pathlib import Path

import yaml

# %% [markdown]
# ## 1. Paths and connection

# %%
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT_ROOT / "config.yaml"
NORMALIZED = PROJECT_ROOT / "data" / "normalized"
MYSQL_HOME = PROJECT_ROOT / "data" / "mysql"
DATABASE = "comptox_invitrodb_v4_3"

# Concentration-unit conversion to micromolar.
UNIT_TO_UM = {"um": 1.0, "µm": 1.0, "μm": 1.0, "micromolar": 1.0, "nm": 0.001, "mm": 1000.0, "m": 1e6}


def connect(database=None, streaming=False):
    """A pymysql connection over the local unix socket (server-side cursor if `streaming`)."""
    import pymysql
    return pymysql.connect(
        unix_socket=os.environ.get("COMPTOX_MYSQL_SOCKET", str(MYSQL_HOME / "mysql.sock")),
        user=os.environ.get("COMPTOX_MYSQL_USER", "root"),
        password=os.environ.get("COMPTOX_MYSQL_PASSWORD", ""),
        database=database, charset="utf8mb4", autocommit=True,
        cursorclass=pymysql.cursors.SSCursor if streaming else pymysql.cursors.Cursor,
        read_timeout=3600, write_timeout=3600)

# %% [markdown]
# ## 2. The export query
# mc5 (hit calls) is joined through mc4 and sample to the chemical, and through assay_component_endpoint,
# assay_component, assay and assay_source to the assay's name and source. `mc5_param` supplies the AC50 and the
# curve top (efficacy). Only `chid_rep = 1` (the representative series per chemical and endpoint) is kept.

# %%
EXPORT_SQL = """SELECT c.dsstox_substance_id, src.assay_source_name,
  e.assay_component_endpoint_name, m.hitc, p.ac50, p.top,
  m.m5id, m.model_type, s.tested_conc_unit
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
  AND e.export_ready=1 AND e.data_usability=1"""

# %% [markdown]
# ## 3. Run

# %%
def export_toxcast(check=False):
    state = json.loads((MYSQL_HOME / "import_status.json").read_text())
    if state.get("status") != "complete":
        raise RuntimeError("The SQL import is unfinished. Complete step 00a first.")
    threshold = yaml.safe_load(CONFIG.read_text())["data"]["toxcast_active_threshold"]
    NORMALIZED.mkdir(parents=True, exist_ok=True)
    connection = connect(DATABASE, streaming=True)
    with connection.cursor() as cursor:
        cursor.execute("SELECT name, description FROM mc5_model_type ORDER BY model_type")
        print("Model types:", cursor.fetchall())
        if check:
            connection.close()
            return
        cursor.execute("SELECT dsstox_substance_id, chnm, casn FROM chemical WHERE dsstox_substance_id LIKE %s", ("DTXSID%",))
        with (NORMALIZED / "chemical_identifiers.csv").open("w") as out:
            writer = csv.writer(out)
            writer.writerow(["DTXSID", "preferred_name", "CASRN"])
            writer.writerows(cursor)

        cursor.execute(EXPORT_SQL)
        destination = NORMALIZED / "bioactivity.csv"
        temporary = destination.with_suffix(".csv.part")
        count = unknown_units = 0
        with temporary.open("w") as out:
            writer = csv.writer(out)
            writer.writerow(["DTXSID", "program", "assay", "hitcall", "ac50_uM", "efficacy", "signed_hitcall",
                             "m5id", "model_type", "concentration_unit"])
            for sid, program, assay, hit, ac50, top, m5id, model_type, unit in cursor:
                factor = UNIT_TO_UM.get(str(unit).lower().strip())
                active = hit >= threshold
                if factor is None:
                    unknown_units += 1
                potency = ac50 * factor if active and factor is not None and ac50 is not None and ac50 > 0 and math.isfinite(ac50) else None
                writer.writerow([sid, program, assay, max(0.0, hit), potency, top if active else None, hit, m5id, model_type, unit])
                count += 1
                if count % 100000 == 0:
                    print("Exported", count, flush=True)
        if count == 0:
            raise ValueError("No bioactivity rows were exported; check the database schema and units.")
        os.replace(temporary, destination)
    connection.close()
    (NORMALIZED / "mysql_export_report.json").write_text(json.dumps(
        {"database": DATABASE, "rows": count, "unknown_unit_rows": unknown_units, "threshold": threshold,
         "source": state["source"]}, indent=2))
    print("Exported rows:", count, "| potency masked for unknown units:", unknown_units)

# %% Command line
def main():
    parser = argparse.ArgumentParser(description="Export ToxCast/Tox21 measurements from MySQL.")
    parser.add_argument("--check", action="store_true", help="only test the connection")
    export_toxcast(parser.parse_args().check)


if __name__ == "__main__":
    main()
