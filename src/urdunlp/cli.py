"""The `urdunlp` command: the library's main functions from a shell.

    urdunlp to-urdu "mera naam ali hai"      # میرا نام علی ہے
    urdunlp to-roman "میں ٹھیک ہوں"            # main theek hoon
    urdunlp langid "هي ڪتاب منهنجو آهي"        # sd  Sindhi ...
    urdunlp normalize "كتاب"                 # کتاب
    echo "kal milte hain" | urdunlp to-urdu  # reads standard input, line by line
    urdunlp to-roman -i news.txt -o news-roman.txt

Also `python -m urdunlp`.

Files and standard input are read as UTF-8, with or without a byte-order mark, or as
UTF-16 when they start with one - which is what Windows PowerShell 5.1 writes with
`>` - and anything else stops with one line saying so, not a traceback. Output is
UTF-8; `--bom` adds the mark Excel and old Notepad look for.
"""

from __future__ import annotations

import argparse
import codecs
import contextlib
import io
import json
import os
import sys
import tempfile
from collections.abc import Callable, Iterator
from typing import Any, BinaryIO, TextIO

from . import __version__
from .langid import identify_language
from .normalize import normalize
from .tokenize import words
from .translit import transliterate_to_roman, transliterate_with_confidence

_Factory = Callable[[argparse.Namespace], Callable[[str], str]]


def _emit(args: argparse.Namespace, line: str, text: str, record: dict[str, Any]) -> str:
    """The output line: `text`, or with --json one JSON object for the input line."""
    if not args.json:
        return text
    return json.dumps({"input": line, **record}, ensure_ascii=False)


def _to_urdu(args: argparse.Namespace) -> Callable[[str], str]:
    def run(line: str) -> str:
        result = transliterate_with_confidence(
            line,
            keep_english=args.keep_english,
            urdu_punctuation=not args.latin_punctuation,
        )
        text = result.text
        if args.sources:
            detail = " ".join(f"{token}/{source}" for token, source in result.sources)
            text = f"{text}\t{detail}"
        record = {
            "text": result.text,
            "sources": [list(pair) for pair in result.sources],
            "rule_share": result.rule_share,
        }
        return _emit(args, line, text, record)

    return run


def _to_roman(args: argparse.Namespace) -> Callable[[str], str]:
    def run(line: str) -> str:
        text = transliterate_to_roman(line, method=args.method)
        return _emit(args, line, text, {"text": text})

    return run


def _langid(args: argparse.Namespace) -> Callable[[str], str]:
    def run(line: str) -> str:
        guess = identify_language(line)
        if guess.language is None:
            text = "-\tno Perso-Arabic letters"
        else:
            note = "\tshort text, a guess" if guess.short else ""
            text = f"{guess.language}\t{guess.name}\tmargin {guess.margin}{note}"
        record = {
            "language": guess.language,
            "name": guess.name,
            "margin": guess.margin,
            "short": guess.short,
        }
        return _emit(args, line, text, record)

    return run


def _normalize(args: argparse.Namespace) -> Callable[[str], str]:
    def run(line: str) -> str:
        text = normalize(line, normalize_digits=args.digits)
        return _emit(args, line, text, {"text": text})

    return run


def _words(args: argparse.Namespace) -> Callable[[str], str]:
    def run(line: str) -> str:
        tokens = words(line, keep_punctuation=args.punctuation)
        return _emit(args, line, " | ".join(tokens), {"words": tokens})

    return run


_POWERSHELL_HELP = """\
Windows PowerShell 5.1 passes text to programs as ASCII by default, which turns
every Urdu letter into '?' before urdunlp sees it, and reads their output in the
console's code page. Either run this once in the session:
    $OutputEncoding = [Text.UTF8Encoding]::new()
    [Console]::InputEncoding = [Console]::OutputEncoding = $OutputEncoding
or skip the pipe and let urdunlp read and write the files itself:
    urdunlp to-roman -i input.txt -o output.txt"""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="urdunlp",
        description="Urdu and Roman Urdu text processing. Each command takes TEXT as an "
        "argument, or reads a file (-i) or standard input one line at a time.",
        epilog="On Windows PowerShell 5.1, see `urdunlp COMMAND --help` for the encoding "
        "setup, or use -i and -o.",
    )
    parser.add_argument("--version", action="version", version=f"urdunlp {__version__}")
    commands = parser.add_subparsers(dest="command", metavar="COMMAND", required=True)

    def command(name: str, summary: str, make: _Factory) -> argparse.ArgumentParser:
        sub = commands.add_parser(
            name,
            help=summary,
            description=summary,
            epilog=_POWERSHELL_HELP,
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        sub.add_argument("text", nargs="*", help="text to process (default: -i FILE or stdin)")
        sub.add_argument(
            "-i", "--input", metavar="FILE", help="read FILE, one line at a time, instead of TEXT"
        )
        sub.add_argument(
            "-o", "--output", metavar="FILE", help="write to FILE (UTF-8) instead of stdout"
        )
        sub.add_argument(
            "--encoding",
            metavar="NAME",
            help="encoding of the input, when it is neither UTF-8 nor UTF-16 with a "
            "byte-order mark (for example cp1252)",
        )
        sub.add_argument(
            "--bom",
            action="store_true",
            help="start the output with a UTF-8 byte-order mark, for Excel and old Notepad",
        )
        sub.add_argument(
            "--json",
            action="store_true",
            help="one JSON object per input line (JSON Lines), with the input and every field",
        )
        sub.set_defaults(make=make)
        return sub

    sub = command("to-urdu", "Roman Urdu to Urdu script", _to_urdu)
    sub.add_argument("--keep-english", action="store_true", help="leave English words in Latin")
    sub.add_argument(
        "--latin-punctuation", action="store_true", help="leave ? , ; . as typed, not ؟ ، ؛ ۔"
    )
    sub.add_argument(
        "--sources", action="store_true", help="also print which stage resolved each token"
    )
    sub = command("to-roman", "Urdu script to Roman Urdu", _to_roman)
    sub.add_argument(
        "--method",
        choices=("learned", "rules"),
        default="learned",
        help="learned spellings (default) or the letter-by-letter rules",
    )
    command("langid", "which of 11 Perso-Arabic-script languages", _langid)
    sub = command("normalize", "unify Arabic/Urdu letters, strip diacritics", _normalize)
    sub.add_argument("--digits", action="store_true", help="also turn Urdu digits into 0-9")
    sub = command("words", "split into word tokens, separated by ' | '", _words)
    sub.add_argument("--punctuation", action="store_true", help="keep punctuation tokens")
    return parser


class InputError(Exception):
    """Input that cannot be read as text: reported in one line, not as a traceback."""


def _sniff(head: bytes) -> str:
    """The encoding of a byte stream from its first bytes: a BOM, or UTF-16 by its NULs."""
    if head.startswith(codecs.BOM_UTF32_LE) or head.startswith(codecs.BOM_UTF32_BE):
        return "utf-32"
    if head.startswith(codecs.BOM_UTF16_LE) or head.startswith(codecs.BOM_UTF16_BE):
        return "utf-16"  # PowerShell 5.1's `>` and Out-File write this
    # UTF-16 without a mark: ASCII text has a NUL in every other byte.
    even, odd = head[0:64:2], head[1:64:2]
    if len(head) >= 4 and odd.count(0) > len(odd) * 0.6 and not even.count(0):
        return "utf-16-le"
    if len(head) >= 4 and even.count(0) > len(even) * 0.6 and not odd.count(0):
        return "utf-16-be"
    return "utf-8-sig"  # UTF-8, and a BOM is removed rather than leaked into the output


def _decoded_lines(binary: BinaryIO, name: str, encoding: str | None) -> Iterator[str]:
    peek = getattr(binary, "peek", None)
    head = peek(64)[:64] if peek is not None else b""
    chosen = encoding or _sniff(head)
    if chosen.lower().replace("_", "-") in ("utf-8", "utf8"):
        chosen = "utf-8-sig"
    try:
        stream = io.TextIOWrapper(binary, encoding=chosen, errors="strict", newline=None)
    except LookupError:
        raise InputError(f"unknown encoding {encoding!r}") from None
    number = 0
    try:
        for line in stream:
            number += 1
            yield line.rstrip("\n")
    except UnicodeDecodeError as error:
        byte = error.object[error.start : error.start + 1].hex().upper()
        hint = (
            "save it as UTF-8, or name its encoding with --encoding (for example cp1252)"
            if encoding is None
            else f"it is not {encoding}"
        )
        raise InputError(
            f"{name}, line {number + 1}: byte 0x{byte} is not {chosen.replace('-sig', '')}; {hint}"
        ) from None
    finally:
        stream.detach()


def _lines(args: argparse.Namespace) -> Iterator[str]:
    if args.text:
        yield " ".join(args.text)
        return
    if args.input:
        if os.path.isdir(args.input):
            raise InputError(f"cannot read {args.input}: it is a directory")
        try:
            handle = open(args.input, "rb")  # noqa: SIM115 - closed below, lines are lazy
        except OSError as error:
            raise InputError(f"cannot read {args.input}: {error.strerror}") from None
        with handle:
            yield from _decoded_lines(handle, args.input, args.encoding)
        return
    binary = getattr(sys.stdin, "buffer", None)
    if binary is None:  # stdin replaced by a text stream, as an embedding app may do
        for line in sys.stdin:
            yield line.rstrip("\r\n").lstrip("﻿")
        return
    yield from _decoded_lines(binary, "standard input", args.encoding)


def _looks_mangled(line: str) -> bool:
    """Whether a line looks like text a shell already replaced with question marks."""
    stripped = line.replace(" ", "")
    return stripped.count("?") >= 3 and not stripped.strip("?.,!؟۔0123456789")


class _Output:
    """Where the results go: standard output, or -o FILE written in full, then moved.

    The file is written beside its destination under a temporary name and moved over
    it only once every line is done. Opening FILE for writing straight away, as the
    first version did, truncated it before a single line was read - so
    `urdunlp normalize -i f.txt -o f.txt` left f.txt empty and exited 0. It also
    means an error half way through leaves the old FILE as it was.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        self.path: str | None = args.output
        self.temporary: str | None = None
        if self.path is None:
            if args.bom:
                sys.stdout.write("\ufeff")
            self.stream: TextIO = sys.stdout
            return
        if os.path.isdir(self.path):
            raise InputError(f"cannot write {self.path}: it is a directory")
        folder = os.path.dirname(os.path.abspath(self.path))
        try:
            handle, self.temporary = tempfile.mkstemp(prefix=".urdunlp-", suffix=".tmp", dir=folder)
        except OSError as error:
            raise InputError(f"cannot write {self.path}: {error.strerror}") from None
        self.stream = open(  # noqa: SIM115 - closed in finish/abandon
            handle, "w", encoding="utf-8-sig" if args.bom else "utf-8", newline="\n"
        )

    def finish(self) -> None:
        if self.temporary is None or self.path is None:
            return
        self.stream.close()
        try:
            os.replace(self.temporary, self.path)
        except OSError as error:
            self.abandon()
            raise InputError(f"cannot write {self.path}: {error.strerror}") from None
        self.temporary = None

    def abandon(self) -> None:
        if self.temporary is None:
            return
        self.stream.close()
        with contextlib.suppress(OSError):
            os.remove(self.temporary)
        self.temporary = None


def _use_utf8(stream: TextIO) -> None:
    """Write Urdu whatever the console or redirect's code page.

    Done before the arguments are parsed, because `--help` prints ؟ ، ؛ ۔ too: with
    output redirected on Windows the stream is cp1252, and `urdunlp to-urdu --help >
    help.txt` ended in UnicodeEncodeError.
    """
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is not None:
        with contextlib.suppress(ValueError, OSError):
            reconfigure(encoding="utf-8", errors="replace")


def _stdin_is_terminal() -> bool:
    """Whether standard input is a person at a keyboard rather than a pipe or file.

    On Windows `isatty()` is also true for the NUL device, so `urdunlp to-urdu < NUL`
    was told to pipe something in. There a console is told apart by asking for its
    console mode, which only a real console has.
    """
    try:
        if not sys.stdin.isatty():
            return False
    except (AttributeError, ValueError):
        return False
    # A `sys.platform` comparison, not `os.name`: mypy narrows on the former (and
    # so skips `ctypes.windll` entirely when checking this file for a non-Windows
    # target) but not the latter, where the attribute would need an ignore that is
    # wrong on exactly one of the two platforms CI runs on.
    if sys.platform != "win32":
        return True
    try:
        import ctypes
        import msvcrt

        handle = msvcrt.get_osfhandle(sys.stdin.fileno())
        mode = ctypes.c_uint32()
        return bool(ctypes.windll.kernel32.GetConsoleMode(handle, ctypes.byref(mode)))
    except (OSError, ValueError, AttributeError, io.UnsupportedOperation):
        return True


def main(argv: list[str] | None = None) -> int:
    _use_utf8(sys.stdout)
    _use_utf8(sys.stderr)
    parser = _parser()
    args = parser.parse_args(argv)
    if args.text and args.input:
        parser.error("give TEXT or -i FILE, not both")
    if not args.text and not args.input and _stdin_is_terminal():
        # Waiting silently on a terminal looks like a hang.
        parser.error(f"give TEXT, -i FILE, or pipe it in: echo TEXT | urdunlp {args.command}")
    if len(args.text) == 1 and os.path.isfile(args.text[0]):
        print(
            f"urdunlp: note: processing the text {args.text[0]!r}, which is also a file "
            f"name; to read the file, use -i {args.text[0]}",
            file=sys.stderr,
        )
    run = args.make(args)
    out: _Output | None = None
    warned = False
    try:
        out = _Output(args)
        for line in _lines(args):
            if not warned and _looks_mangled(line):
                warned = True
                print(
                    "urdunlp: warning: the input arrived as question marks - the shell "
                    "replaced the text before urdunlp saw it.\n" + _POWERSHELL_HELP,
                    file=sys.stderr,
                )
            out.stream.write(run(line) + "\n")
            if out.temporary is None:
                out.stream.flush()  # a pipe reader sees each line as it is done
        out.finish()
    except InputError as error:
        print(f"urdunlp: error: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    except BrokenPipeError:  # `urdunlp to-urdu < big.txt | head`
        return 0
    finally:
        if out is not None:
            out.abandon()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
