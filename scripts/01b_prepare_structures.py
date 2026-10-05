"""Step 1b - Attach structures to the ToxCast chemical list

The ToxCast database knows chemicals only by DTXSID and name. This step joins EPA's DSSTox structure dump
(step 00b) onto that list and writes the final `chemicals.csv` with RDKit-canonical SMILES. It also resolves the
compound names in the MEA workbook to DTXSIDs.

How the MEA matching works: an MEA compound is mapped to a DTXSID only when its name matches, exactly (case- and
whitespace-insensitive), the preferred name or a synonym of **exactly one** DSSTox chemical. Names matching
several chemicals are left unresolved (marked ambiguous in the audit file) instead of being guessed. Entries
that already have a DTXSID (e.g. filled in by hand) are never overwritten.

The DSSTox file is large, so it is streamed from the zip in chunks. Only rows that are a ToxCast chemical or an
MEA name match are kept, and every SMILES is validated with RDKit (invalid or empty structures are dropped).

Input : data/source/dsstox/DSSTox_CCD_dump_12092025_CSVs.zip (step 00b), the ToxCast database (MySQL) or
        data/normalized/chemical_identifiers.csv (step 01a), data/mea_processed/mea_to_comptox_mapping_template.csv (step 06)
Output: data/normalized/chemicals.csv, structure_report.json;
        data/mea_processed/mea_to_comptox_mapping_template.csv (updated), mea_mapping_audit.csv, mea_structure_pubchem.csv
Run   : python scripts/01b_prepare_structures.py
"""

# %%
import csv
import json
import os
import zipfile
from collections import defaultdict
from pathlib import Path

import pandas as pd
from rdkit import Chem

# %% [markdown]
# ## 1. Paths and connection

# %%
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DSSTOX_ZIP = PROJECT_ROOT / "data" / "source" / "dsstox" / "DSSTox_CCD_dump_12092025_CSVs.zip"
DSSTOX_MEMBER = "DSSTox_CCD_dump_12092025/DSSToxCCDdump.csv"
NORMALIZED = PROJECT_ROOT / "data" / "normalized"
MEA_PROCESSED = PROJECT_ROOT / "data" / "mea_processed"
MYSQL_HOME = PROJECT_ROOT / "data" / "mysql"
DATABASE = "comptox_invitrodb_v4_3"
SOURCE_LABEL = "EPA DSSTox December 2025"


def connect(database=None):
    """A pymysql connection over the local unix socket (override with COMPTOX_MYSQL_* environment variables)."""
    import pymysql
    return pymysql.connect(
        unix_socket=os.environ.get("COMPTOX_MYSQL_SOCKET", str(MYSQL_HOME / "mysql.sock")),
        user=os.environ.get("COMPTOX_MYSQL_USER", "root"),
        password=os.environ.get("COMPTOX_MYSQL_PASSWORD", ""),
        database=database, charset="utf8mb4", autocommit=True)

# %% [markdown]
# ## 2. Helpers

# %%
def exact_name(value):
    """Case-folded name with whitespace collapsed, for exact-match comparison."""
    return " ".join(str(value).strip().casefold().split())


def toxcast_identifiers():
    """DTXSIDs of the ToxCast chemicals (exported from MySQL on first use)."""
    path = NORMALIZED / "chemical_identifiers.csv"
    if not path.exists():
        with connect(DATABASE) as connection, connection.cursor() as cursor:
            cursor.execute("SELECT dsstox_substance_id, chnm, casn FROM chemical WHERE dsstox_substance_id LIKE %s", ("DTXSID%",))
            with path.open("w") as out:
                writer = csv.writer(out)
                writer.writerow(["DTXSID", "preferred_name", "CASRN"])
                writer.writerows(cursor.fetchall())
    return set(pd.read_csv(path).DTXSID.dropna())

# %% [markdown]
# ## 3. Run
# Steps:
# 1. Load the ToxCast DTXSIDs and the MEA compound names.
# 2. Stream the DSSTox CSV in chunks of 25,000 rows. For each chemical collect its names (preferred name plus
#    the `|`-separated synonyms). Keep it if it is a ToxCast chemical or matches an MEA name; validate and
#    canonicalise its SMILES.
# 3. Write `chemicals.csv`.
# 4. For every MEA compound: if it has no DTXSID yet and exactly one DSSTox chemical carries its name, fill in
#    that DTXSID and SMILES. Record an audit row (`mapped`, `ambiguous` or `unresolved`) and, for mapped
#    compounds, a structure row.
# 5. Save the updated mapping, audit and structure files and a summary report.

# %%
def prepare_structures():
    NORMALIZED.mkdir(parents=True, exist_ok=True)
    ids = toxcast_identifiers()
    mapping_path = MEA_PROCESSED / "mea_to_comptox_mapping_template.csv"
    mapping = pd.read_csv(mapping_path, dtype=str).fillna("") if mapping_path.exists() else pd.DataFrame()
    mea_names = set(mapping.base_compound_name.map(exact_name)) if not mapping.empty else set()

    matches = defaultdict(dict)   # MEA name -> {DTXSID: row}
    rows, invalid = [], 0
    with zipfile.ZipFile(DSSTOX_ZIP) as archive, archive.open(DSSTOX_MEMBER) as handle:
        for chunk in pd.read_csv(handle, chunksize=25000, dtype=str):
            for r in chunk.itertuples(index=False):
                aliases = {exact_name(r.PREFERRED_NAME)}
                if isinstance(r.IDENTIFIER, str):
                    aliases.update(exact_name(x) for x in r.IDENTIFIER.split("|"))
                matched = mea_names & aliases
                if r.DTXSID not in ids and not matched:
                    continue
                mol = Chem.MolFromSmiles(r.SMILES) if isinstance(r.SMILES, str) else None
                if mol is None or mol.GetNumAtoms() == 0:
                    invalid += 1
                    continue
                row = {"DTXSID": r.DTXSID, "smiles": Chem.MolToSmiles(mol), "preferred_name": r.PREFERRED_NAME,
                       "CASRN": r.CASRN, "source": SOURCE_LABEL}
                rows.append(row)
                for name in matched:
                    matches[name][r.DTXSID] = row

    chemicals = pd.DataFrame(rows).drop_duplicates("DTXSID")
    if chemicals.empty:
        raise ValueError("No valid ToxCast structures were resolved.")
    temporary = NORMALIZED / "chemicals.csv.part"
    chemicals.to_csv(temporary, index=False)
    temporary.replace(NORMALIZED / "chemicals.csv")

    audit, structures = [], []
    for index, r in mapping.iterrows():
        candidates = matches[exact_name(r.base_compound_name)]
        if not r.DTXSID and len(candidates) == 1:
            resolved = next(iter(candidates.values()))
            mapping.loc[index, "DTXSID"] = resolved["DTXSID"]
            mapping.loc[index, "SMILES"] = resolved["smiles"]
            mapping.loc[index, "mapping_source"] = "unique exact DSSTox preferred name/synonym; December 2025"
        sid = mapping.loc[index, "DTXSID"]
        found = chemicals[chemicals.DTXSID.eq(sid)]
        if len(found) == 1:
            structures.append({"compound_id": r.compound_id, "base_compound_name": r.base_compound_name,
                               "DTXSID": sid, "SMILES": found.iloc[0].smiles, "source": SOURCE_LABEL})
        audit.append({"compound_id": r.compound_id, "candidate_ids": ";".join(sorted(candidates)), "DTXSID": sid,
                      "status": "mapped" if sid else ("ambiguous" if candidates else "unresolved")})
    if not mapping.empty:
        mapping.to_csv(mapping_path, index=False)
        pd.DataFrame(audit).to_csv(MEA_PROCESSED / "mea_mapping_audit.csv", index=False)
        pd.DataFrame(structures, columns=["compound_id", "base_compound_name", "DTXSID", "SMILES", "source"]).to_csv(
            MEA_PROCESSED / "mea_structure_pubchem.csv", index=False)

    report = {"structures": len(chemicals), "toxcast_identifiers": len(ids),
              "toxcast_with_structure": int(chemicals.DTXSID.isin(ids).sum()), "invalid_or_missing_structures": invalid,
              "mea_mapped": sum(a["status"] == "mapped" for a in audit), "source": str(DSSTOX_ZIP)}
    (NORMALIZED / "structure_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

# %% Command line
if __name__ == "__main__":
    prepare_structures()
