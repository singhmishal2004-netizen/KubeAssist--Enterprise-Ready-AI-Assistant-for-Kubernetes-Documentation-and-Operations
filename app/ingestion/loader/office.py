
from pathlib import Path

import logfire
from unstructured.partition.auto import partition


def parse_office(file_path: str) -> str:
    """
    Parses Office documents (.docx, .pptx, .xlsx)
    using the Unstructured library.

    Extracts document elements and converts them
    into text for the RAG ingestion pipeline.
    """

    path = Path(file_path)

    with logfire.span(
        "office.parse",
        filename=path.name
    ):
        try:
            if not path.exists():
                raise FileNotFoundError(
                    f"File not found: {path}"
                )

            # 1. Automatically detect and partition document
            elements = partition(filename=str(path))

            # 2. Extract text from document elements
            full_text = "\n".join(
                str(element)
                for element in elements
                if str(element).strip()
            )

            # 3. Handle empty extraction
            if not full_text.strip():
                logfire.warning(
                    "Office document returned empty text",
                    filename=path.name
                )
                return ""

            # 4. Log extraction metrics
            logfire.info(
                "Office document parsed successfully",
                filename=path.name,
                element_count=len(elements),
                extracted_characters=len(full_text)
            )

            return full_text

        except Exception:
            logfire.exception(
                "Office document parsing failed",
                filename=path.name
            )
            raise
