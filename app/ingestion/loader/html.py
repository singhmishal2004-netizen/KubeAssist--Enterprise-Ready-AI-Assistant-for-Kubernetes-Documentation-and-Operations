
from pathlib import Path

from bs4 import BeautifulSoup
import logfire


def parse_html(file_path: str) -> str:
    """
    Parses HTML documents using BeautifulSoup.

    Removes unnecessary HTML elements and extracts
    clean readable text for the RAG pipeline.
    """

    path = Path(file_path)

    with logfire.span(
        "html.parse",
        filename=path.name
    ):
        try:
            # 1. Read HTML content
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")

            with path.open(
                "r",
                encoding="utf-8",
                errors="ignore"
            ) as file:
                content = file.read()

            # 2. Parse HTML
            soup = BeautifulSoup(content, "html.parser")

            # 3. Remove unnecessary elements
            for element in soup([
                "script",
                "style",
                "meta",
                "noscript",
                "iframe"
            ]):
                element.decompose()

            # 4. Extract readable text
            text = soup.get_text(separator="\n", strip=True)

            # 5. Normalize whitespace
            lines = [
                " ".join(line.split())
                for line in text.splitlines()
            ]

            text_clean = "\n".join(
                line for line in lines if line
            )

            # 6. Log extraction metrics
            logfire.info(
                "HTML parsing completed",
                filename=path.name,
                extracted_characters=len(text_clean)
            )

            return text_clean

        except Exception:
            logfire.exception(
                "HTML parsing failed",
                filename=path.name
            )
            raise
