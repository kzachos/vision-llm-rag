from sentence_transformers import CrossEncoder

def re_rank_cross_encoders(prompt, documents, metadata_list=None):
    relevant_text = ""
    relevant_text_ids = []
    relevant_metadata = []
    encoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    ranks = encoder_model.rank(prompt, documents, top_k=3)
    for rank in ranks:
        doc_id = rank["corpus_id"]
        relevant_text += documents[doc_id]
        relevant_text_ids.append(doc_id)
        if metadata_list and doc_id < len(metadata_list):
            relevant_metadata.append(metadata_list[doc_id])
    return relevant_text, relevant_text_ids, relevant_metadata 