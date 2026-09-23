"""STEP 0a -- Import the raw EPA invitrodb (ToxCast) MySQL dump into a local
MySQL server, then export a first-pass chemicals/bioactivity CSV pair.

The dump is too large for pandas (about 16 GB compressed), so this step uses
the installed MySQL client/server and streams only the columns needed by the
prototype.  Credentials are read from the normal MySQL client configuration;
no password is accepted on the command line.

Note: this script's import_dump() loads the *entire* raw dump into a database
named invitrodb_v4_3 -- useful for exploring the full schema, but the rest of
the pipeline (01a_export_toxcast_mysql.py onward) reads from a smaller,
curated database instead (see core/mysql_io.py). Once that curated database
exists, you normally don't need to re-run the full import.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import os
import subprocess
from pathlib import Path

from core.paths import DATA_NORMALIZED, DATA_SOURCE

DEFAULT_DUMP = DATA_SOURCE / "toxcast" / "toxcast_clowder_dataset.zip_extracted" / "invitrodb_v4_3.sql.gz"


def mysql_args(args):
    """Build the common `mysql` CLI prefix (host/port/socket/user/login-path)
    shared by every query/import call below, from the parsed CLI args."""
    out = [args.mysql]
    if args.host:
        out += ["--host", args.host]
    if args.port:
        out += ["--port", str(args.port)]
    if args.socket:
        out += ["--socket", args.socket]
    if args.user:
        out += ["--user", args.user]
    if args.login_path:
        out += ["--login-path", args.login_path]
    return out


def query(args, sql):
    """Run one SQL statement and return its raw tab-separated stdout (used for
    small lookups like --check, not for bulk exports)."""
    cmd = mysql_args(args) + ["--batch", "--raw", "--skip-column-names", "-e", sql]
    return subprocess.run(cmd, check=True, text=True, capture_output=True).stdout


def export_query(args, sql, path, columns):
    """Run a SELECT and write its results straight to a CSV file, without ever
    holding the full result set in Python -- the mysql client streams its
    tab-separated output directly to `path`, which is then reformatted to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = mysql_args(args) + ["--batch", "--raw", "-e", sql]
    with path.open("w", newline="", encoding="utf-8") as out:
        proc = subprocess.run(cmd, check=True, text=True, stdout=out, stderr=subprocess.PIPE)
    # mysql's tabular output is TSV; normalize it to CSV without loading all rows.
    tmp = path.with_suffix(path.suffix + ".tmp")
    path.replace(tmp)
    with tmp.open(encoding="utf-8", newline="") as src, path.open("w", encoding="utf-8", newline="") as dst:
        reader = csv.reader(src, delimiter="\t")
        writer = csv.writer(dst)
        writer.writerow(columns)
        writer.writerows(reader)
    tmp.unlink()


def import_dump(args):
    """Stream the gzip-compressed SQL dump straight into `mysql`, decompressing
    on the fly so no second multi-GB temporary file is ever written to disk."""
    dump = Path(args.dump)
    if not dump.exists():
        raise FileNotFoundError(dump)
    create = mysql_args(args) + ["-e", "CREATE DATABASE IF NOT EXISTS invitrodb_v4_3"]
    subprocess.run(create, check=True)
    # mysql accepts gzip on stdin only after decompression; stream it without a
    # second 16 GB temporary file.
    client = subprocess.Popen(mysql_args(args), stdin=subprocess.PIPE)
    with gzip.open(dump, "rb") as source:
        while True:
            block = source.read(8 * 1024 * 1024)
            if not block:
                break
            client.stdin.write(block)
    client.stdin.close()
    if client.wait() != 0:
        raise RuntimeError("MySQL SQL import failed")


def main():
    """CLI entry point: optionally import the raw dump, then always export a
    first-pass chemicals.csv/bioactivity.csv (superseded later by 01a/01b's
    more complete versions)."""
    p = argparse.ArgumentParser()
    p.add_argument("--dump", default=str(DEFAULT_DUMP))
    p.add_argument("--mysql", default="mysql")
    p.add_argument("--user", default=os.getenv("MYSQL_USER", ""))
    p.add_argument("--host", default=os.getenv("MYSQL_HOST", ""))
    p.add_argument("--port", type=int, default=int(os.getenv("MYSQL_PORT", "0")) or None)
    p.add_argument("--socket", default=os.getenv("MYSQL_SOCKET", ""))
    p.add_argument("--login-path", default=os.getenv("MYSQL_LOGIN_PATH", ""))
    p.add_argument("--skip-import", action="store_true")
    p.add_argument("--check", action="store_true")
    a = p.parse_args()
    if a.check:
        print(query(a, "SELECT VERSION()" ).strip())
        print(query(a, "SHOW DATABASES LIKE 'invitrodb_v4_3'" ).strip() or "database missing")
        return
    if not a.skip_import:
        import_dump(a)
    DATA_NORMALIZED.mkdir(parents=True, exist_ok=True)
    # chemical contains DSSTox IDs and structure fields in invitrodb v4.
    # The invitrodb SQL schema stores identifiers, but not molecular SMILES.
    # Keep the structure column so a later DSSTox/PubChem enrichment can fill it.
    export_query(a, "USE invitrodb_v4_3; SELECT DISTINCT dsstox_substance_id, '' AS smiles, chnm, casn FROM chemical WHERE dsstox_substance_id IS NOT NULL", DATA_NORMALIZED / "chemicals.csv", ["DTXSID", "smiles", "preferred_name", "CASRN"])
    # mc0 is the assay measurement table; join to assay_component_endpoint for
    # stable assay names and retain continuous hitcall/ac50/efficacy channels.
    export_query(a, "USE invitrodb_v4_3; SELECT c.dsstox_substance_id, 'ToxCast/Tox21', ace.assay_component_endpoint_name, mc5.hitc, NULL, NULL FROM mc5 JOIN mc4 ON mc4.m4id=mc5.m4id JOIN sample s ON s.spid=mc4.spid JOIN chemical c ON c.chid=s.chid JOIN assay_component_endpoint ace ON ace.aeid=mc5.aeid WHERE c.dsstox_substance_id IS NOT NULL", DATA_NORMALIZED / "bioactivity.csv", ["DTXSID", "program", "assay", "hitcall", "log10_ac50_uM", "efficacy"])
    print("Normalized chemistry and bioactivity tables written to", DATA_NORMALIZED)


if __name__ == "__main__":
    main()
