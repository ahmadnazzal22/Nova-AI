import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import pytest
from part2_rag.chunker import chunk_text, read_file, ingest_document, SUPPORTED_EXTS


class TestChunkText:
    def test_empty_text(self):
        assert chunk_text("", chunk_size=10, overlap=2) == []

    def test_whitespace_only(self):
        assert chunk_text("   \n\n  ", chunk_size=10, overlap=2) == []

    def test_single_chunk(self):
        text = "word " * 50
        chunks = chunk_text(text, chunk_size=100, overlap=10)
        assert len(chunks) == 1

    def test_multiple_chunks(self):
        text = "word " * 500
        chunks = chunk_text(text, chunk_size=100, overlap=10)
        assert len(chunks) > 1

    def test_no_overlap(self):
        text = "word " * 300
        chunks = chunk_text(text, chunk_size=100, overlap=0)
        assert 3 <= len(chunks) <= 4

    def test_full_overlap(self):
        text = "word " * 200
        chunks = chunk_text(text, chunk_size=100, overlap=99)
        assert len(chunks) > 2

    def test_chunks_not_empty(self):
        text = "hello world testing the chunker function"
        chunks = chunk_text(text, chunk_size=3, overlap=0)
        assert all(c.strip() for c in chunks)


class TestReadFile:
    def test_read_txt(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("Hello world")
            tmppath = f.name
        try:
            content = read_file(tmppath)
            assert content == "Hello world"
        finally:
            os.unlink(tmppath)

    def test_read_md_strips_formatting(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
            f.write("# Title\n**bold** text and [link](http://example.com)")
            tmppath = f.name
        try:
            content = read_file(tmppath)
            assert "Title" in content
            assert "bold" in content
            assert "link" in content
            assert "http://" not in content
        finally:
            os.unlink(tmppath)

    def test_unsupported_extension_reads_as_text(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".docx", delete=False, encoding="utf-8") as f:
            f.write("plain text content")
            tmppath = f.name
        try:
            content = read_file(tmppath)
            assert content == "plain text content"
        finally:
            os.unlink(tmppath)


class TestIngestDocument:
    def test_txt_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("word " * 1000)
            tmppath = f.name
        try:
            chunks = ingest_document(tmppath, chunk_size=100, overlap=10)
            assert len(chunks) > 1
            assert all(isinstance(c, str) for c in chunks)
        finally:
            os.unlink(tmppath)

    def test_md_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
            f.write("# Doc\n" + ("word " * 1000))
            tmppath = f.name
        try:
            chunks = ingest_document(tmppath, chunk_size=100, overlap=10)
            assert len(chunks) > 1
        finally:
            os.unlink(tmppath)

    def test_supported_extensions(self):
        assert ".txt" in SUPPORTED_EXTS
        assert ".md" in SUPPORTED_EXTS
        assert ".pdf" in SUPPORTED_EXTS
