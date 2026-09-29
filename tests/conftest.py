"""Shared fixtures. Unit tests are offline; tests marked `live` hit Ollama, Supabase and ipu.ac.in."""

import io
import sys
from pathlib import Path

import pytest
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]


def make_text_pdf(text: str) -> bytes:
    """A minimal valid one-page PDF with a real text layer."""
    content = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = b"%PDF-1.4\n"
    offsets = []
    for number, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + obj + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % off for off in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, xref)
    return out


def make_blank_pdf(pages: int = 2) -> bytes:
    from PyPDF2 import PdfWriter

    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=612, height=792)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


class FakeResponse:
    """Stands in for requests.Response (works as a context manager for stream=True)."""

    def __init__(self, body: bytes | str = b"", status: int = 200, headers: dict | None = None):
        self.content = body.encode() if isinstance(body, str) else body
        self.text = self.content.decode("utf-8", "replace")
        self.status_code = status
        self.headers = headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Client Error")

    def iter_content(self, chunk_size=1):
        for i in range(0, len(self.content), chunk_size):
            yield self.content[i : i + chunk_size]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeSession:
    """Replaces PoliteSession: maps URL -> FakeResponse (or an exception to raise)."""

    def __init__(self, pages: dict):
        self.pages = pages
        self.requested: list[str] = []

    def get(self, url, **kwargs):
        self.requested.append(url)
        result = self.pages.get(url, FakeResponse(status=404))
        if isinstance(result, Exception):
            raise result
        return result


@pytest.fixture
def text_pdf():
    return make_text_pdf("Datesheet for LLB examinations December 2026, theory papers start on 5 January 2027.")


@pytest.fixture
def blank_pdf():
    return make_blank_pdf()
