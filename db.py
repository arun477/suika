import sqlite3
from copy import deepcopy
import json

DATABASE_NAME = "suika_commands.db"
TABLE = "suika_commands"
VEC_ID_MAPPING_TABLE = "id_to_vec_pos"


def create_database():
    conn = sqlite3.connect(DATABASE_NAME)
    try:
        cursor = conn.cursor()
        cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE} (
            linux_cmd_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT NOT NULL,
            syntax TEXT NOT NULL,
            keywords TEXT NOT NULL,
            examples TEXT NOT NULL
        )
        """)
        conn.commit()
    finally:
        conn.close()


def insert_doc(doc, cursor):
    if not doc.get("name"):
        return
    # convert list into json string
    doc["linux_cmd_id"] = f"linux_{doc['name']}"
    doc["keywords"] = json.dumps(doc["keywords"])
    doc["examples"] = json.dumps(doc["examples"])
    try:
        cursor.execute(
            f"""
        INSERT INTO {TABLE} (name, linux_cmd_id, description, syntax, keywords, examples)
        VALUES (:name, :linux_cmd_id, :description, :syntax, :keywords, :examples)
        """,
            doc,
        )
    except sqlite3.IntegrityError as e:
        print(f"integrityError: skipping document {doc['linux_cmd_id']} due to unique constraint violation.")


def load_data(docs):
    try:
        conn = sqlite3.connect(DATABASE_NAME)
        cursor = conn.cursor()
        for doc in docs:
            doc = deepcopy(doc)
            insert_doc(doc, cursor)
        conn.commit()
    except sqlite3.OperationalError as e:
        print(f"operationalError: {e}")
    finally:
        conn.close()


def query_by_id(doc_id):
    print("doc_id", doc_id)
    conn = sqlite3.connect(DATABASE_NAME)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM suika_commands WHERE linux_cmd_id = ?", (doc_id,))
    res = cursor.fetchone()
    doc = dict(res)
    conn.close()
    doc["keywords"] = json.loads(doc["keywords"])
    doc["examples"] = json.loads(doc["examples"])
    return doc


def fetch_all_documents():
    try:
        conn = sqlite3.connect(DATABASE_NAME)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(f"SELECT * FROM {TABLE}")
        rows = cursor.fetchall()
        documents = [dict(row) for row in rows]
        for doc in documents:
            doc["keywords"] = json.loads(doc["keywords"])
            doc["examples"] = json.loads(doc["examples"])
    except sqlite3.OperationalError as e:
        print(f"OperationalError: {e}")
        documents = []
    finally:
        conn.close()
    return documents


def fetch_vec_id_mapping():
    try:
        conn = sqlite3.connect(DATABASE_NAME)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(f"SELECT * FROM {VEC_ID_MAPPING_TABLE}")
        rows = cursor.fetchall()
        documents = [dict(row) for row in rows]
        mapping = {ele["faiss_index_id"]: ele["linux_cmd_id"] for ele in documents}
    except sqlite3.OperationalError as e:
        print(f"OperationalError: {e}")
        mapping = {}
    finally:
        conn.close()
    return mapping
