from __future__ import annotations

from pathlib import Path
import re
from urllib.parse import urlencode
from urllib.request import urlopen

from src.preprocessing.fasta import parse_fasta
from src.preprocessing.validation import SequenceValidationError

NCBI_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
LOCAL_INPUT_SUFFIXES = {".fa", ".fasta", ".fna", ".txt"}


class InputSourceError(ValueError):
    pass


def _allowed_local_roots() -> list[Path]:
    return [Path.cwd().resolve(), Path("/tmp").resolve()]


def list_local_input_files() -> list[str]:
    files: set[str] = set()
    for root in _allowed_local_roots():
        if not root.exists():
            continue
        for candidate in root.rglob("*"):
            if candidate.is_file() and candidate.suffix.lower() in LOCAL_INPUT_SUFFIXES:
                files.add(str(candidate.resolve()))
    return sorted(files)


def _decode_uploaded_text(uploaded_bytes: bytes) -> str:
    try:
        return uploaded_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise InputSourceError("Uploaded file must be valid UTF-8 FASTA/text.") from exc


def load_records_from_text(text: str) -> list[tuple[str, str]]:
    try:
        records = parse_fasta(text)
    except SequenceValidationError as exc:
        raise InputSourceError(str(exc)) from exc
    if not records:
        raise InputSourceError("No valid sequence records found. Provide FASTA or a plain DNA sequence.")
    return records


def load_records_from_path(path: str) -> list[tuple[str, str]]:
    requested = path.strip()
    allowed_files = {candidate: Path(candidate) for candidate in list_local_input_files()}
    fasta_path = allowed_files.get(requested)
    if fasta_path is None:
        roots = ", ".join(str(root) for root in _allowed_local_roots())
        raise InputSourceError(f"Disk input must be an indexed FASTA/text file under: {roots}")
    return load_records_from_text(fasta_path.read_text(encoding="utf-8"))


def fetch_ncbi_fasta(accession: str, timeout: float = 20.0) -> str:
    accession = accession.strip()
    if not accession:
        raise InputSourceError("NCBI accession is empty.")
    query = urlencode({"db": "nuccore", "id": accession, "rettype": "fasta", "retmode": "text"})
    with urlopen(f"{NCBI_EFETCH_URL}?{query}", timeout=timeout) as response:
        payload = response.read().decode("utf-8")
    if not payload.strip() or not payload.lstrip().startswith(">"):
        raise InputSourceError(f"NCBI did not return FASTA for accession '{accession}'.")
    return payload


PRESET_ACCESSIONS = {
    "E. coli K-12 genome (NC_000913.3)": "NC_000913.3",
    "Human TP53 gene region (NG_017013.2)": "NG_017013.2",
    "Phage lambda genome (NC_001416.1)": "NC_001416.1",
}


def split_accessions(text: str) -> list[str]:
    """Split a comma/whitespace separated accession list, preserving order and dropping duplicates."""
    return list(dict.fromkeys(token for token in re.split(r"[\s,;]+", text) if token))


def load_records_from_accession(accession: str) -> list[tuple[str, str]]:
    accessions = split_accessions(accession)
    if not accessions:
        raise InputSourceError("NCBI accession is empty.")
    records: list[tuple[str, str]] = []
    for item in accessions:
        records.extend(load_records_from_text(fetch_ncbi_fasta(item)))
    return records


def load_sequence_records(
    *,
    pasted_text: str = "",
    file_path: str = "",
    accession: str = "",
    uploaded_bytes: bytes | None = None,
) -> tuple[list[tuple[str, str]], str]:
    if pasted_text.strip():
        return load_records_from_text(pasted_text), "Pasted sequence"
    if uploaded_bytes is not None:
        return load_records_from_text(_decode_uploaded_text(uploaded_bytes)), "Uploaded file"
    if file_path.strip():
        return load_records_from_path(file_path), "Disk file"
    if accession.strip():
        return load_records_from_accession(accession), f"NCBI accession: {accession.strip()}"
    raise InputSourceError("Provide pasted sequence text, an uploaded file, an indexed local file, or an NCBI accession.")
