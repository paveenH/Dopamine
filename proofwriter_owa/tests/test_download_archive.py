#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regression tests for data_proofwriter_owa.download_archive() (review finding,
2026-09-04): the previous implementation did a single blocking f.write(r.read())
with no progress output and no interrupt safety, so a Ctrl-C mid-download left
a truncated file at the FINAL path that a later run's os.path.exists() check
would silently treat as "already downloaded".

Uses a local http.server thread serving a small synthetic zip -- no real
network, no real ProofWriter data. Run with:
    python3 test_download_archive.py
"""

from __future__ import annotations

import http.server
import io
import os
import shutil
import sys
import tempfile
import threading
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data_proofwriter_owa as dpo  # noqa: E402

FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name}  {detail}")
        FAILURES.append(name)


def _make_zip_bytes(valid=True):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("depth-3/OWA/meta-test.jsonl", '{"id":"t1"}\n' * 50)
        zf.writestr("depth-5/OWA/meta-test.jsonl", '{"id":"t2"}\n' * 50)
    data = buf.getvalue()
    if not valid:
        # truncate mid-stream: still looks like SOME bytes came across, but
        # is not a valid zip (central directory is at the end).
        data = data[: len(data) // 2]
    return data


class _ZipHandler(http.server.BaseHTTPRequestHandler):
    payload = b""
    truncate_response = False  # simulate a connection that dies mid-transfer

    def do_GET(self):
        body = self.payload
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.truncate_response:
            self.wfile.write(body[: len(body) // 3])
            # deliberately do NOT close cleanly / write the rest -- the
            # client's r.read(chunk_size) loop will eventually hit a short
            # read; closing the connection here simulates a dropped network.
            self.close_connection = True
            return
        self.wfile.write(body)

    def log_message(self, *a):
        pass  # keep test output quiet


def _serve(payload: bytes, truncate=False):
    _ZipHandler.payload = payload
    _ZipHandler.truncate_response = truncate
    httpd = http.server.HTTPServer(("127.0.0.1", 0), _ZipHandler)
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return httpd, f"http://127.0.0.1:{port}/archive.zip"


def test_successful_download_lands_at_final_path_and_verifies():
    payload = _make_zip_bytes(valid=True)
    httpd, url = _serve(payload)
    tmp = tempfile.mkdtemp()
    try:
        path = dpo.download_archive(tmp, url=url, chunk_size=32)
        check("final path exists after a successful download",
              os.path.exists(path))
        check("no .part file left behind on success",
              not os.path.exists(path + ".part"))
        check("downloaded bytes match the served payload",
              open(path, "rb").read() == payload)
        # re-running against an already-complete, valid file must be a
        # no-op fast path (integrity-verified, not re-downloaded)
        path2 = dpo.download_archive(tmp, url=url, chunk_size=32)
        check("re-running on a complete file returns the same path",
              path2 == path)
    finally:
        httpd.shutdown()
        shutil.rmtree(tmp, ignore_errors=True)


def test_interrupted_download_leaves_only_a_part_file():
    """Simulates the exact failure mode from the review: a connection that
    dies mid-transfer must NOT leave anything at the final path. Before the
    fix, f.write(r.read()) would raise partway through urlopen's read and
    (depending on where) could leave a truncated file at the path used by a
    later os.path.exists() check; now the code always writes to `path.part`
    and only os.replace()s it after a successful integrity check, so an
    interrupted download can only ever leave `path.part`, never `path`."""
    payload = _make_zip_bytes(valid=True)
    httpd, url = _serve(payload, truncate=True)
    tmp = tempfile.mkdtemp()
    try:
        threw = False
        try:
            dpo.download_archive(tmp, url=url, chunk_size=16)
        except Exception:
            threw = True
        check("a truncated transfer raises rather than silently succeeding",
              threw)
        final_path = os.path.join(tmp, dpo.ARCHIVE_BASENAME)
        check("NOTHING is left at the final path after an interrupted download",
              not os.path.exists(final_path))
        check(".part file IS left behind for inspection",
              os.path.exists(final_path + ".part"))
    finally:
        httpd.shutdown()
        shutil.rmtree(tmp, ignore_errors=True)


def test_stale_part_file_is_removed_before_a_fresh_attempt():
    """A leftover .part from a previous crashed run must not be silently
    resumed/appended to (this implementation does not support resume) -- it
    should be discarded and the download started clean."""
    payload = _make_zip_bytes(valid=True)
    httpd, url = _serve(payload)
    tmp = tempfile.mkdtemp()
    try:
        final_path = os.path.join(tmp, dpo.ARCHIVE_BASENAME)
        stale_part = final_path + ".part"
        with open(stale_part, "wb") as f:
            f.write(b"garbage-from-a-previous-crashed-run")
        path = dpo.download_archive(tmp, url=url, chunk_size=32)
        check("download succeeds despite a stale .part file present",
              os.path.exists(path))
        check("final content is the FRESH download, not the stale garbage",
              open(path, "rb").read() == payload)
    finally:
        httpd.shutdown()
        shutil.rmtree(tmp, ignore_errors=True)


def test_existing_but_corrupt_zip_at_final_path_is_rejected():
    """Before this fix, the already-exists fast path only checked
    os.path.exists() + optionally sha256 -- it never verified the file was
    even a valid zip, so a corrupted cached archive from an old buggy run
    would be silently accepted and only fail much later inside
    _find_meta_files with a confusing 'structure differs' error."""
    tmp = tempfile.mkdtemp()
    try:
        final_path = os.path.join(tmp, dpo.ARCHIVE_BASENAME)
        with open(final_path, "wb") as f:
            f.write(_make_zip_bytes(valid=False))  # truncated, invalid zip
        died = False
        try:
            dpo.download_archive(tmp, url="http://unused.invalid/x.zip")
        except SystemExit:
            died = True
        check("a pre-existing corrupt zip at the final path is rejected, "
              "not silently accepted", died)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_existing_valid_zip_is_accepted_without_network():
    """The fast path must not touch the network at all when a valid,
    integrity-verified archive is already present."""
    tmp = tempfile.mkdtemp()
    try:
        final_path = os.path.join(tmp, dpo.ARCHIVE_BASENAME)
        payload = _make_zip_bytes(valid=True)
        with open(final_path, "wb") as f:
            f.write(payload)
        # a URL that would fail immediately if ever contacted (nothing is
        # listening there) -- proves the fast path never calls urlopen
        path = dpo.download_archive(
            tmp, url="http://127.0.0.1:1/should-never-be-hit.zip")
        check("existing valid zip accepted without touching the network",
              path == final_path)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    print("== successful download lands at final path ==")
    test_successful_download_lands_at_final_path_and_verifies()
    print("== interrupted download leaves only a .part file ==")
    test_interrupted_download_leaves_only_a_part_file()
    print("== stale .part file from a crashed run is discarded ==")
    test_stale_part_file_is_removed_before_a_fresh_attempt()
    print("== existing corrupt zip at final path is rejected ==")
    test_existing_but_corrupt_zip_at_final_path_is_rejected()
    print("== existing valid zip is accepted without network ==")
    test_existing_valid_zip_is_accepted_without_network()

    print()
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s): {FAILURES}")
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
