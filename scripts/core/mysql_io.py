"""Project-local MySQL connection (socket only; no passwords in source)."""
import os
from core.paths import DATA_DIR

MYSQL_HOME = DATA_DIR / 'mysql'
SOCKET = MYSQL_HOME / 'mysql.sock'
DATABASE = 'comptox_invitrodb_v4_3'


def connect(database=None, streaming=False):
    import pymysql
    return pymysql.connect(
        unix_socket=os.environ.get('COMPTOX_MYSQL_SOCKET', str(SOCKET)),
        user=os.environ.get('COMPTOX_MYSQL_USER', 'root'),
        password=os.environ.get('COMPTOX_MYSQL_PASSWORD', ''),
        database=database, charset='utf8mb4', autocommit=True,
        cursorclass=pymysql.cursors.SSCursor if streaming else pymysql.cursors.Cursor,
        read_timeout=3600, write_timeout=3600,
    )
