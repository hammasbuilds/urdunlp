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
import io
import os
import sys
from collections.abc import Callable, Iterator
from typing import BinaryIO, TextIO

from . import __version__
from .langid import identify_language
from .normalize import normalize
from .tokenize import words
from .translit import transliterate_to_roman, transliterate_with_confidence

_Factory = Callable[[argparse.Namespace], Callable[[str], str]]


def _to_urdu(args: argparse.Namespace) -> Callable[[str], str]:
    def run(line: str) -> str:
        result = transliterate_with_confidence(
            line,
            keep_english=args.keep_english,
            urdu_punctuation=not args.latin_punctuation,
        )
        if not args.sources:
            return result.text
        detail = " ".join(f"{token}/{source}" for token, source in result.sources)
        return f"{result.text}\t{detail}"

    return run


def _to_roman(args: argparse.Namespace) -> Callable[[str], str]:
    return lambda line: transliterate_to_roman(line, method=args.method)


def _langid(args: argparse.Namespace) -> Callable[[str], str]:
    def run(line: str) -> str:
        guess = identify_language(line)
        if guess.language is None:
            return "-\tno Perso-Arabic letters"
        note = "\tshort text, a guess" if guess.short else ""
        return f"{guess.language}\t{guess.name}\tmargin {guess.margin}{note}"

    return run


def _normalize(args: argparse.Namespace) -> Callable[[str], str]:
    return lambda line: normalize(line, normalize_digits=args.digits)


def _words(args: argparse.Namespace) -> Callable[[str], str]:
    return lambda line: " | ".join(words(line, keep_punctuation=args.punctuation))


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


def _output(args: argparse.Namespace) -> TextIO:
    if args.output:
        try:
            return open(  # noqa: SIM115 - closed by main
                args.output, "w", encoding="utf-8-sig" if args.bom else "utf-8", newline="\n"
            )
        except OSError as error:
            raise InputError(f"cannot write {args.output}: {error.strerror}") from None
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        # Urdu on a Windows console or a redirect: the default code page cannot
        # encode it, and print() would raise UnicodeEncodeError on the first letter.
        reconfigure(encoding="utf-8")
    if args.bom:
        sys.stdout.write("﻿")
    return sys.stdout


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.text and args.input:
        parser.error("give TEXT or -i FILE, not both")
    if not args.text and not args.input and sys.stdin.isatty():
        # Waiting silently on a terminal looks like a hang.
        parser.error(f"give TEXT, -i FILE, or pipe it in: echo TEXT | urdunlp {args.command}")
    if len(args.text) == 1 and os.path.isfile(args.text[0]):
        print(
            f"urdunlp: note: processing the text {args.text[0]!r}, which is also a file "
            f"name; to read the file, use -i {args.text[0]}",
            file=sys.stderr,
        )
    run = args.make(args)
    out: TextIO | None = None
    warned = False
    try:
        out = _output(args)
        for line in _lines(args):
            if not warned and _looks_mangled(line):
                warned = True
                print(
                    "urdunlp: warning: the input arrived as question marks - the shell "
                    "replaced the text before urdunlp saw it.\n" + _POWERSHELL_HELP,
                    file=sys.stderr,
                )
            out.write(run(line) + "\n")
            out.flush()
    except InputError as error:
        print(f"urdunlp: error: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    except BrokenPipeError:  # `urdunlp to-urdu < big.txt | head`
        return 0
    finally:
        if out is not None and out is not sys.stdout:
            out.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
