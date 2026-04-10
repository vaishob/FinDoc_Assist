from app.services.chunking import Chunker
from app.schemas import ParsedPage


def test_chunker_creates_chunks():
    chunker = Chunker(target_words=10, overlap_words=2)
    pages = [ParsedPage(page_number=1, text=" ".join(["alpha"] * 26))]
    chunks = chunker.chunk("doc_1", pages)
    assert len(chunks) >= 2
    assert all(chunk.document_id == "doc_1" for chunk in chunks)
