from eb1a.corpus import chunk_text
from eb1a.metrics import criterion_gold_chunks


def test_criterion_gold_mapping_uses_numbered_headings():
    text = "\n".join([f"Criterion {i}: Official guidance for criterion {i}." for i in range(1, 11)])
    chunks = chunk_text(text, "u", "t", target_words=20, overlap_words=0)
    mapping = criterion_gold_chunks(chunks)
    assert len(mapping) == 10
    assert mapping["awards"].endswith("0000")
