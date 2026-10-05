"""Step 0a - Import the ToxCast MySQL dump

The ToxCast / invitrodb release is a ~16 GB compressed MySQL dump. This step streams it into a local MySQL
server (decompressing on the fly, so no second multi-GB file is written) and then exports a first-pass
chemicals / bioactivity CSV pair. It needs the `mysql` command-line client and a running MySQL server.

Credentials come from your normal MySQL client configuration; no password is ever passed on the command line.
Create a login path once (outside this repository):
    mysql_config_editor set --login-path=comptox --host=localhost --user=YOUR_USER --password

The rest of the pipeline reads a curated database (see step 01a); once that exists you normally do not need to
repeat this import. The first-pass CSVs written here are superseded by step 01a (bioactivity) and step 01b
(structures).

Input : data/source/toxcast/toxcast_clowder_dataset.zip_extracted/invitrodb_v4_3.sql.gz  (from step 00, extracted by step 01)
Output: a MySQL database `invitrodb_v4_3`, data/normalized/chemicals.csv, data/normalized/bioactivity.csv
Run   : python scripts/00a_import_toxcast_sql.py --check --login-path comptox
        python scripts/00a_import_toxcast_sql.py --login-path comptox [--skip-import]
"""

# %%
import argparse
import csv
import gzip
import os
import subprocess
from pathlib import Path

# %% [markdown]
# ## 1. Paths

# %%
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DUMP = (PROJECT_ROOT / "data" / "source" / "toxcast" / "toxcast_clowder_dataset.zip_extracted"
                / "invitrodb_v4_3.sql.gz")
NORMALIZED = PROJECT_ROOT / "data" / "normalized"
DATABASE = "invitrodb_v4_3"

# %% [markdown]
# ## 2. Talking to MySQL
# Everything goes through the `mysql` command-line client. `mysql_command` builds its common prefix from the
# connection settings; `query` runs a small statement; `export_query` streams a large SELECT straight to a CSV
# file without holding the rows in Python; `import_dump` pipes the decompressed dump into MySQL.

# %%
def mysql_command(mysql="mysql", host="", port=None, socket="", user="", login_path=""):
    command = [mysql]
    for flag, value in (("--host", host), ("--port", port), ("--socket", socket), ("--user", user),
                        ("--login-path", login_path)):
        if value:
            command += [flag, str(value)]
    return command


def query(connection, sql):
    """Run one statement and return its tab-separated output."""
    command = mysql_command(**connection) + ["--batch", "--raw", "--skip-column-names", "-e", sql]
    return subprocess.run(command, check=True, text=True, capture_output=True).stdout


def export_query(connection, sql, path, columns):
    """Stream a SELECT to `path` as CSV with the given header (mysql's own output is tab-separated)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    command = mysql_command(**connection) + ["--batch", "--raw", "-e", sql]
    raw = path.with_suffix(path.suffix + ".tmp")
    with raw.open("w", newline="", encoding="utf-8") as out:
        subprocess.run(command, check=True, text=True, stdout=out, stderr=subprocess.PIPE)
    with raw.open(encoding="utf-8", newline="") as src, path.open("w", encoding="utf-8", newline="") as dst:
        reader = csv.reader(src, delimiter="\t")
        writer = csv.writer(dst)
        writer.writerow(columns)
        writer.writerows(reader)
    raw.unlink()


def import_dump(connection, dump):
    dump = Path(dump)
    if not dump.exists():
        raise FileNotFoundError(dump)
    subprocess.run(mysql_command(**connection) + ["-e", f"CREATE DATABASE IF NOT EXISTS {DATABASE}"], check=True)
    client = subprocess.Popen(mysql_command(**connection), stdin=subprocess.PIPE)
    with gzip.open(dump, "rb") as source:
        while block := source.read(8 * 1024 * 1024):
            client.stdin.write(block)
    client.stdin.close()
    if client.wait() != 0:
        raise RuntimeError("The MySQL import failed.")

# %% [markdown]
# ## 3. Run
# * `check=True` only prints the MySQL version and whether the database exists.
# * Otherwise the dump is imported (unless `skip_import=True`) and two first-pass tables are exported:
#   **chemicals** (DTXSID, name, CASRN; the SMILES column is left empty because the ToxCast database has no
#   structures) and **bioactivity** (one row per measurement with assay name and hit call).

# %%
def import_toxcast(dump=DEFAULT_DUMP, check=False, skip_import=False, mysql="mysql", host="", port=None,
                   socket="", user="", login_path=""):
    connection = dict(mysql=mysql, host=host, port=port, socket=socket, user=user, login_path=login_path)
    if check:
        print(query(connection, "SELECT VERSION()").strip())
        print(query(connection, f"SHOW DATABASES LIKE '{DATABASE}'").strip() or "database missing")
        return
    if not skip_import:
        import_dump(connection, dump)
    NORMALIZED.mkdir(parents=True, exist_ok=True)
    export_query(connection,
                 f"USE {DATABASE}; SELECT DISTINCT dsstox_substance_id, '' AS smiles, chnm, casn FROM chemical "
                 "WHERE dsstox_substance_id IS NOT NULL",
                 NORMALIZED / "chemicals.csv", ["DTXSID", "smiles", "preferred_name", "CASRN"])
    export_query(connection,
                 f"USE {DATABASE}; SELECT c.dsstox_substance_id, 'ToxCast/Tox21', ace.assay_component_endpoint_name, "
                 "mc5.hitc, NULL, NULL FROM mc5 JOIN mc4 ON mc4.m4id=mc5.m4id JOIN sample s ON s.spid=mc4.spid "
                 "JOIN chemical c ON c.chid=s.chid JOIN assay_component_endpoint ace ON ace.aeid=mc5.aeid "
                 "WHERE c.dsstox_substance_id IS NOT NULL",
                 NORMALIZED / "bioactivity.csv", ["DTXSID", "program", "assay", "hitcall", "log10_ac50_uM", "efficacy"])
    print("Chemicals and bioactivity tables written to", NORMALIZED)

# %% Command line
def main():
    parser = argparse.ArgumentParser(description="Import the ToxCast MySQL dump and export first-pass tables.")
    parser.add_argument("--dump", default=str(DEFAULT_DUMP))
    parser.add_argument("--mysql", default="mysql")
    parser.add_argument("--user", default=os.getenv("MYSQL_USER", ""))
    parser.add_argument("--host", default=os.getenv("MYSQL_HOST", ""))
    parser.add_argument("--port", type=int, default=int(os.getenv("MYSQL_PORT", "0")) or None)
    parser.add_argument("--socket", default=os.getenv("MYSQL_SOCKET", ""))
    parser.add_argument("--login-path", default=os.getenv("MYSQL_LOGIN_PATH", ""))
    parser.add_argument("--skip-import", action="store_true", help="the database is already imported")
    parser.add_argument("--check", action="store_true", help="only test the connection")
    a = parser.parse_args()
    import_toxcast(a.dump, a.check, a.skip_import, a.mysql, a.host, a.port, a.socket, a.user, a.login_path)


if __name__ == "__main__":
    main()
