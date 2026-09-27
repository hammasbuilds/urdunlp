"""The `urdunlp` command on the files and pipes Windows actually produces.

Audit round 2, on Windows: a file written by PowerShell 5.1's `>` (UTF-16 with a
byte-order mark) and a cp1252 file both ended in a raw UnicodeDecodeError traceback;
a UTF-8 file with a BOM leaked U+FEFF into the output; and `urdunlp to-urdu bom.txt`
quietly transliterated the file *name*.
"""

import io
import subprocess
import sys
from pathlib import Path

import pytest

import urdunlp as U
from urdunlp.cli import main

URDU = "میں ٹھیک ہوں"
ROMAN = "main theek hoon"


class _Stdin:
    """A stand-in for sys.stdin whose .buffer yields raw bytes, as a pipe does."""

    def __init__(self, data: bytes) -> None:
        self.buffer = io.BufferedReader(io.BytesIO(data))  # type: ignore[arg-type]

    def isatty(self) -> bool:
        return False


def _file(tmp_path: Path, data: bytes, name: str = "in.txt") -> str:
    path = tmp_path / name
    path.write_bytes(data)
    return str(path)


@pytest.mark.parametrize(
    "data",
    [
        URDU.encode("utf-8"),
        b"\xef\xbb\xbf" + URDU.encode("utf-8"),  # UTF-8 with a BOM: Notepad, Excel
        URDU.encode("utf-16"),  # UTF-16 with a BOM: PowerShell 5.1's `>` and Out-File
        b"\xfe\xff" + URDU.encode("utf-16-be"),
        URDU.encode("utf-32"),
    ],
    ids=["utf-8", "utf-8-bom", "utf-16-le-bom", "utf-16-be-bom", "utf-32"],
)
def test_input_files_in_every_unicode_encoding(tmp_path, capsys, data):
    assert main(["to-roman", "-i", _file(tmp_path, data)]) == 0
    out = capsys.readouterr().out
    assert out == ROMAN + "\n"  # and no U+FEFF leaked into it


def test_utf16_without_a_bom_is_recognised(tmp_path, capsys):
    assert main(["to-urdu", "-i", _file(tmp_path, "kal milte hain".encode("utf-16-le"))]) == 0
    assert capsys.readouterr().out == "کل ملتے ہیں\n"


def test_a_file_in_another_encoding_is_one_line_not_a_traceback(tmp_path, capsys):
    path = _file(tmp_path, "café au lait".encode("cp1252"))
    assert main(["to-urdu", "-i", path]) == 1
    err = capsys.readouterr().err
    assert err.count("\n") == 1 and "Traceback" not in err
    assert "line 1" in err and "0xE9" in err and "--encoding" in err
    # ...and naming the encoding reads it
    assert main(["to-urdu", "-i", path, "--encoding", "cp1252"]) == 0
    assert capsys.readouterr().out.startswith("café")


def test_undecodable_standard_input_is_one_line(monkeypatch, capsys):
    monkeypatch.setattr(sys, "stdin", _Stdin("kal\ncafé".encode("cp1252")))
    assert main(["to-urdu"]) == 1
    captured = capsys.readouterr()
    assert captured.out == "کل\n"  # the good line before it was still processed
    assert "standard input, line 2" in captured.err and "Traceback" not in captured.err


def test_standard_input_in_utf16(monkeypatch, capsys):
    monkeypatch.setattr(sys, "stdin", _Stdin(URDU.encode("utf-16")))
    assert main(["to-roman"]) == 0
    assert capsys.readouterr().out == ROMAN + "\n"


def test_output_file_is_utf8_and_bom_is_optional(tmp_path, capsys):
    out = tmp_path / "out.txt"
    assert main(["to-urdu", "ye kitab hai?", "-o", str(out)]) == 0
    assert capsys.readouterr().out == ""
    assert out.read_bytes() == "یہ کتاب ہے؟\n".encode()
    assert main(["to-urdu", "ye kitab hai?", "-o", str(out), "--bom"]) == 0
    assert out.read_bytes() == b"\xef\xbb\xbf" + "یہ کتاب ہے؟\n".encode()


def test_a_file_name_given_as_text_gets_a_note(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _file(tmp_path, b"kal", "bom.txt")
    assert main(["to-urdu", "bom.txt"]) == 0
    captured = capsys.readouterr()
    assert "-i bom.txt" in captured.err


def test_missing_input_file(capsys):
    assert main(["to-urdu", "-i", "no/such/file.txt"]) == 1
    assert "cannot read no/such/file.txt" in capsys.readouterr().err


def test_text_and_input_file_together_is_an_error(tmp_path):
    with pytest.raises(SystemExit) as exit_:
        main(["to-urdu", "kal", "-i", _file(tmp_path, b"kal")])
    assert exit_.value.code == 2


def test_question_marks_from_the_shell_get_a_warning(monkeypatch, capsys):
    """PowerShell 5.1 pipes Urdu as `???`: say so instead of echoing it back silently."""
    monkeypatch.setattr(sys, "stdin", _Stdin(b"??? ???? ???\n"))
    assert main(["to-roman"]) == 0
    err = capsys.readouterr().err
    assert "question marks" in err and "$OutputEncoding" in err and "-i input.txt" in err


def test_the_real_command_reads_a_utf16_file_through_a_pipe(tmp_path):
    env_path = str(Path(U.__file__).resolve().parent.parent)
    done = subprocess.run(
        [sys.executable, "-m", "urdunlp", "to-roman"],
        input=URDU.encode("utf-16"),
        capture_output=True,
        env={**__import__("os").environ, "PYTHONPATH": env_path},
        timeout=120,
        check=True,
    )
    assert done.stdout.decode("utf-8").strip() == ROMAN
    assert done.stderr == b""
