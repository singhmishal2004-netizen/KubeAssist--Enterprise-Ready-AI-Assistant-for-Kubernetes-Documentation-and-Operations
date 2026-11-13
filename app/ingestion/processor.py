
import os
import sys
import json
import time
import uuid
import logfire

from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.config import settings

from app.services.retrieval.embedding import (
    embed_texts,
    get_embedding_dim,
)

from app.ingestion.chunking.splitter import chunk_text

from app.ingestion.loader.pdf import parse_pdf
from app.ingestion.loader.html import parse_html
from app.ingestion.loader.text import parse_text
from app.ingestion.loader.office import parse_office


# --------------------------------------------------
# LOGFIRE CONFIGURATION
# --------------------------------------------------

logfire.configure(
    service_name="kubeassist-ingestion-service",
    send_to_logfire="if-token-present",
)

if not settings.LOGFIRE_TOKEN:
    print(
        "Logfire cloud export is disabled: "
        "set LOGFIRE_TOKEN in .env."
    )


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

PROCESSED_DATA_DIR = "processed_data"

COLLECTION_NAME = settings.QDRANT_COLLECTION

INITIAL_DOCUMENT_VERSION = 1

CHUNK_SIZE = 1500

QDRANT_MAX_RETRIES = 3
QDRANT_RETRY_DELAY = 2


# --------------------------------------------------
# QDRANT CLIENT
# --------------------------------------------------

qdrant_client = QdrantClient(
    url=settings.QDRANT_CLUSTER_ENDPOINT,
    api_key=settings.QDRANT_API_KEY,
    timeout=120,
)


# --------------------------------------------------
# INGESTION STATISTICS
# --------------------------------------------------

def create_stats():
    return {
        "successful": [],
        "failed": [],
        "skipped": [],
    }


# --------------------------------------------------
# SAVE PROCESSED DATA LOCALLY
# --------------------------------------------------

def save_processed_locally(
    data: dict,
    source_type: str,
    filename: str,
) -> str:

    folder = os.path.join(
        PROCESSED_DATA_DIR,
        source_type,
    )

    os.makedirs(folder, exist_ok=True)

    file_stem = os.path.splitext(filename)[0]

    destination = os.path.join(
        folder,
        f"{file_stem}.json",
    )

    with open(destination, "w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )

    return destination


# --------------------------------------------------
# DOCUMENT TEXT EXTRACTION
# --------------------------------------------------

def extract_text(
    file_path: str,
    filename: str,
) -> str | None:

    extension = os.path.splitext(filename)[1].lower()

    if extension == ".pdf":
        return parse_pdf(file_path)

    elif extension in (".html", ".htm"):
        return parse_html(file_path)

    elif extension in (".txt", ".md", ".log"):
        return parse_text(file_path)

    elif extension in (".docx", ".pptx"):
        return parse_office(file_path)

    else:
        logfire.warning(
            f"Unsupported file type: {filename}"
        )
        return None


# --------------------------------------------------
# STABLE POINT ID GENERATION
# --------------------------------------------------

def generate_point_id(
    document_id: str,
    version: int,
    chunk_index: int,
) -> str:
    """
    Generate a deterministic UUID for each document chunk.

    Retrying the same document with the same version
    and chunk index produces the same Qdrant point ID.
    """

    unique_key = (
        f"{document_id}:"
        f"{version}:"
        f"{chunk_index}"
    )

    return str(
        uuid.uuid5(uuid.NAMESPACE_URL, unique_key)
    )


# --------------------------------------------------
# QDRANT UPSERT WITH RETRIES
# --------------------------------------------------

def upsert_with_retry(points: list) -> bool:
    """
    Retry Qdrant upsert when a temporary request failure occurs.
    """

    for attempt in range(1, QDRANT_MAX_RETRIES + 1):

        try:

            qdrant_client.upsert(
                collection_name=COLLECTION_NAME,
                points=points,
                wait=True,
            )

            return True

        except Exception as error:

            logfire.warning(
                f"Qdrant upsert attempt "
                f"{attempt}/{QDRANT_MAX_RETRIES} failed: {error}"
            )

            if attempt == QDRANT_MAX_RETRIES:
                logfire.exception(
                    "Qdrant upsert failed after all retries."
                )
                return False

            delay = QDRANT_RETRY_DELAY * (2 ** (attempt - 1))

            logfire.info(
                f"Retrying Qdrant upsert in {delay} seconds."
            )

            time.sleep(delay)

    return False


# --------------------------------------------------
# PROCESS INDIVIDUAL FILE
# --------------------------------------------------

def process_file(
    file_path: str,
    filename: str,
    source_type: str,
) -> bool:
    """
    Ingestion pipeline:

    Parse → Chunk → Save → Embed → Index in Qdrant
    """

    with logfire.span(
        "Processing File",
        file=filename,
        source=source_type,
    ):

        try:

            # 1. Extract document text

            full_text = extract_text(
                file_path,
                filename,
            )

            if not full_text or not full_text.strip():

                logfire.warning(
                    f"No text extracted from {filename}."
                )
                return False

            logfire.info(
                f"Successfully extracted text from {filename}."
            )

            # 2. Chunk document

            chunks = chunk_text(
                full_text,
                chunk_size=CHUNK_SIZE,
            )

            if not chunks:

                logfire.warning(
                    f"No chunks generated from {filename}."
                )
                return False

            logfire.info(
                f"Generated {len(chunks)} chunks from {filename}."
            )

            # 3. Document metadata

            document_id = f"{source_type}/{filename}"

            document_version = INITIAL_DOCUMENT_VERSION

            is_latest = True

            # 4. Save processed data locally

            processed_data = {
                "filename": filename,
                "document_id": document_id,
                "source_type": source_type,
                "version": document_version,
                "is_latest": is_latest,
                "chunk_count": len(chunks),
                "chunks": chunks,
            }

            local_path = save_processed_locally(
                processed_data,
                source_type,
                filename,
            )

            logfire.info(
                f"Processed data saved at {local_path}."
            )

            # 5. Generate Jina embeddings

            with logfire.span("Generating Jina Embeddings"):

                embeddings = embed_texts(chunks)

                if len(embeddings) != len(chunks):

                    raise ValueError(
                        f"Embedding count mismatch: "
                        f"{len(chunks)} chunks but "
                        f"{len(embeddings)} embeddings."
                    )

            # 6. Prepare Qdrant points

            points = []

            for index, (chunk, vector) in enumerate(
                zip(chunks, embeddings)
            ):

                point_id = generate_point_id(
                    document_id,
                    document_version,
                    index,
                )

                points.append(
                    models.PointStruct(
                        id=point_id,
                        vector=vector,
                        payload={
                            "text": chunk,
                            "source": filename,
                            "source_type": source_type,
                            "document_id": document_id,
                            "chunk_index": index,
                            "version": document_version,
                            "is_latest": is_latest,
                            "embedding_model": (
                                settings.JINA_EMBEDDING_MODEL
                            ),
                        },
                    )
                )

            # 7. Index in Qdrant with retries

            with logfire.span("Indexing in Qdrant"):

                success = upsert_with_retry(points)

                if not success:
                    raise RuntimeError(
                        f"Qdrant indexing failed for {filename}."
                    )

            logfire.info(
                f"Successfully indexed {len(points)} chunks "
                f"from {filename} into {COLLECTION_NAME}."
            )

            return True

        except Exception as error:

            logfire.exception(
                f"Failed to process {filename}: {error}"
            )

            return False


# --------------------------------------------------
# PROCESS DIRECTORY
# --------------------------------------------------

def process_directory(
    dir_path: str,
    source_type: str,
    stats: dict,
):
    """Process every supported file inside a directory."""

    with logfire.span(
        "Scanning Directory",
        path=dir_path,
        source=source_type,
    ):

        if not os.path.isdir(dir_path):

            logfire.warning(
                f"Directory does not exist: {dir_path}"
            )
            return

        files = sorted([
            filename
            for filename in os.listdir(dir_path)
            if os.path.isfile(
                os.path.join(dir_path, filename)
            )
        ])

        logfire.info(
            f"Found {len(files)} files in {dir_path}."
        )

        for filename in files:

            file_path = os.path.join(
                dir_path,
                filename,
            )

            supported_extensions = {
                ".pdf",
                ".html",
                ".htm",
                ".txt",
                ".md",
                ".log",
                ".docx",
                ".pptx",
            }

            extension = os.path.splitext(filename)[1].lower()

            if extension not in supported_extensions:

                logfire.warning(
                    f"Skipping unsupported file: {filename}"
                )

                stats["skipped"].append(filename)
                continue

            success = process_file(
                file_path,
                filename,
                source_type,
            )

            if success:
                stats["successful"].append(filename)
            else:
                stats["failed"].append(filename)


# --------------------------------------------------
# QDRANT COLLECTION INITIALIZATION
# --------------------------------------------------

def initialize_collection(wipe: bool = False):
    """
    Initialize KubeAssist's dedicated Qdrant collection.

    Reuse an existing compatible collection.
    """

    with logfire.span(
        "Initializing Qdrant Collection",
        collection=COLLECTION_NAME,
    ):

        if wipe:

            if qdrant_client.collection_exists(COLLECTION_NAME):

                qdrant_client.delete_collection(
                    collection_name=COLLECTION_NAME,
                )

                logfire.warning(
                    f"Deleted collection {COLLECTION_NAME}."
                )

        if not qdrant_client.collection_exists(COLLECTION_NAME):

            dimension = get_embedding_dim()

            qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=dimension,
                    distance=models.Distance.COSINE,
                ),
            )

            logfire.info(
                f"Created collection {COLLECTION_NAME} "
                f"with dimension {dimension}."
            )

        else:

            collection_info = qdrant_client.get_collection(
                collection_name=COLLECTION_NAME,
            )

            vector_config = collection_info.config.params.vectors

            if not isinstance(vector_config, models.VectorParams):

                raise ValueError(
                    "Expected a single unnamed vector configuration "
                    "for KubeAssist."
                )

            actual_dimension = vector_config.size
            expected_dimension = get_embedding_dim()

            if actual_dimension != expected_dimension:

                raise ValueError(
                    f"Vector dimension mismatch. "
                    f"Existing: {actual_dimension}, "
                    f"Expected: {expected_dimension}."
                )

            logfire.info(
                f"Existing collection verified: "
                f"{COLLECTION_NAME}, dimension: {actual_dimension}."
            )


# --------------------------------------------------
# UNIVERSAL INGESTION
# --------------------------------------------------

def run_universal_ingestion(
    base_dir: str,
    explicit_source_type: str | None = None,
    wipe: bool = False,
):
    """
    Scan a directory and ingest its documents.
    """

    if not os.path.isdir(base_dir):

        raise NotADirectoryError(
            f"Invalid directory: {base_dir}"
        )

    with logfire.span(
        "Universal Ingestion Started",
        base_directory=base_dir,
    ):

        stats = create_stats()

        # 1. Initialize collection

        initialize_collection(wipe=wipe)

        # 2. Identify subdirectories

        subdirs = [
            directory
            for directory in os.listdir(base_dir)
            if os.path.isdir(
                os.path.join(base_dir, directory)
            )
        ]

        # 3. Process directory directly

        if not subdirs:

            if explicit_source_type:

                source_type = explicit_source_type

            else:

                base_name = os.path.basename(
                    os.path.normpath(base_dir)
                ).lower()

                if "true" in base_name:
                    source_type = "true"

                elif "noisy" in base_name:
                    source_type = "noisy"

                else:
                    source_type = "general"

            process_directory(
                base_dir,
                source_type,
                stats,
            )

        # 4. Process subdirectories separately

        else:

            for subdir in sorted(subdirs):

                folder_name = subdir.lower()

                if "true" in folder_name:
                    source_type = "true"

                elif "noisy" in folder_name:
                    source_type = "noisy"

                else:
                    source_type = subdir

                process_directory(
                    os.path.join(base_dir, subdir),
                    source_type,
                    stats,
                )

        return stats


# --------------------------------------------------
# PRINT FINAL REPORT
# --------------------------------------------------

def print_report(stats: dict):

    print("\n" + "=" * 55)
    print("KUBEASSIST INGESTION REPORT")
    print("=" * 55)

    print(
        f"\nSuccessful: {len(stats['successful'])}"
    )

    for filename in stats["successful"]:
        print(f"  [SUCCESS] {filename}")

    print(
        f"\nFailed: {len(stats['failed'])}"
    )

    for filename in stats["failed"]:
        print(f"  [FAILED]  {filename}")

    print(
        f"\nSkipped: {len(stats['skipped'])}"
    )

    for filename in stats["skipped"]:
        print(f"  [SKIPPED] {filename}")

    print("=" * 55)


# --------------------------------------------------
# MAIN ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":

    """
    Examples:

    Process all datasets:
        python -m app.ingestion.processor Data

    Process noisy data:
        python -m app.ingestion.processor Data/noisy_data noisy

    Process one failed document:
        python -m app.ingestion.processor "Data/noisy_sample_10/example.pdf" noisy

    Wipe and recreate collection (destructive):
        python -m app.ingestion.processor Data --wipe
    """

    wipe_requested = "--wipe" in sys.argv

    clean_args = [
        argument
        for argument in sys.argv[1:]
        if argument != "--wipe"
    ]

    target_path = (
        clean_args[0]
        if clean_args
        else "Data"
    )

    explicit_type = (
        clean_args[1]
        if len(clean_args) > 1
        else None
    )

    if not os.path.exists(target_path):

        print(
            f"Error: path '{target_path}' does not exist."
        )

        sys.exit(1)

    try:

        # Initialize collection once

        initialize_collection(wipe=wipe_requested)

        # Single-file ingestion

        if os.path.isfile(target_path):

            if not explicit_type:

                raise ValueError(
                    "Provide source type when processing "
                    "a single file, e.g. noisy or true."
                )

            filename = os.path.basename(target_path)

            success = process_file(
                target_path,
                filename,
                explicit_type,
            )

            stats = create_stats()

            if success:
                stats["successful"].append(filename)
            else:
                stats["failed"].append(filename)

        # Directory ingestion

        else:

            stats = run_universal_ingestion(
                base_dir=target_path,
                explicit_source_type=explicit_type,
                wipe=False,
            )

        print_report(stats)

        if stats["failed"]:

            print(
                "\nIngestion finished with failures."
            )

            sys.exit(1)

        print(
            "\nKubeAssist ingestion completed successfully."
        )

    except Exception as error:

        logfire.exception(
            f"Ingestion job failed: {error}"
        )

        print(f"\nIngestion error: {error}")

        sys.exit(1)
