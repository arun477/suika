import faiss
import db as _db

DB_NAME = "suika_commands_vector.index"

_vec_index = faiss.read_index(DB_NAME)
_vec_to_id_mapping = _db.fetch_vec_id_mapping()


def search(query_emb, k=1):
    score, vec_idx = _vec_index.search(query_emb[None], k=k)
    doc_idx = _vec_to_id_mapping.get(int(vec_idx[0].item()), "")
    doc = _db.query_by_id(doc_idx)
    return doc, score[0][0].item()
