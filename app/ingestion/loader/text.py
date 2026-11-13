
from pathlib import Path

import logfire


def parse_text(file_path: str) -> str:
    """
    Parses plain-text files such as .txt, .md, .log and .csv.
    """

    path = Path(file_path)

    with logfire.span(
        "text.parse",
        filename=path.name
    ):
        try:
            # 1. Validate file
            if not path.exists():
                raise FileNotFoundError(
                    f"File not found: {path}"
                )

            # 2. Read file content
            with path.open(
                "r",
                encoding="utf-8",
                errors="ignore"
            ) as file:
                content = file.read()

            # 3. Log extraction metrics
            logfire.info(
                "Text file parsed successfully",
                filename=path.name,
                extracted_characters=len(content)
            )

            return content

        except Exception:
            logfire.exception(
                "Text parsing failed",
                filename=path.name
            )
            raise
