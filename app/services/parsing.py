from __future__ import annotations

from pathlib import Path

from ..schemas import ParsedPage


class DocumentParser:
    def parse(self, file_path: Path, content_type: str) -> list[ParsedPage]:
        if content_type == "text/plain" or file_path.suffix.lower() == ".txt":
            text = file_path.read_text(encoding="utf-8", errors="ignore")
            return [ParsedPage(page_number=1, text=text)]
        if content_type == "application/pdf" or file_path.suffix.lower() == ".pdf":
            return self._parse_pdf(file_path)
        raise ValueError("Unsupported file type")

    def _parse_pdf(self, file_path: Path) -> list[ParsedPage]:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError("pypdf is required to parse PDF files") from exc

        reader = PdfReader(str(file_path))
        pages: list[ParsedPage] = []
        for index, page in enumerate(reader.pages, start=1):
            pages.append(ParsedPage(page_number=index, text=page.extract_text() or ""))
        return pages
