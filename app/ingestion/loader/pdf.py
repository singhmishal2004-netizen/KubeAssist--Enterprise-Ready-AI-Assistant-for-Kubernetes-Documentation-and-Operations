
from pathlib import Path

import logfire
from pypdf import PdfReader


def parse_pdf(file_path: str) -> str:
    """
    Extracts text from PDF documents using pypdf.
    Uses pdfplumber as a fallback for pages where
    pypdf fails to extract text.
    """

    path = Path(file_path)

    with logfire.span(
        "pdf.parse",
        filename=path.name
    ):
        try:
            if not path.exists():
                raise FileNotFoundError(
                    f"PDF not found: {path}"
                )

            # 1. Read PDF
            reader = PdfReader(str(path))
            total_pages = len(reader.pages)

            logfire.info(
                "PDF loaded",
                filename=path.name,
                total_pages=total_pages
            )

            text_parts: list[str] = []
            blank_pages: list[int] = []

            # 2. Extract text using pypdf
            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""

                if text.strip():
                    text_parts.append(text)
                else:
                    blank_pages.append(page_num)

            # 3. Fallback using pdfplumber
            if blank_pages:
                logfire.info(
                    "Retrying blank pages with pdfplumber",
                    blank_pages=blank_pages
                )

                try:
                    import pdfplumber

                    with pdfplumber.open(str(path)) as pdf:
                        for page_num in blank_pages:
                            page = pdf.pages[page_num - 1]
                            fallback_text = page.extract_text() or ""

                            if fallback_text.strip():
                                text_parts.append(fallback_text)

                except Exception:
                    logfire.exception(
                        "pdfplumber fallback failed",
                        filename=path.name
                    )

            # 4. Combine extracted text
            full_text = "\n".join(text_parts)

            # 5. Extraction result
            if not full_text.strip():
                logfire.warning(
                    "No text extracted from PDF",
                    filename=path.name,
                    total_pages=total_pages
                )
            else:
                logfire.info(
                    "PDF parsing completed",
                    filename=path.name,
                    extracted_characters=len(full_text),
                    blank_pages=len(blank_pages)
                )

            return full_text

        except Exception:
            logfire.exception(
                "PDF parsing failed",
                filename=path.name
            )
            raise
