"""The `urdunlp` command: the library's main functions from a shell.

    urdunlp to-urdu "mera naam ali hai"      # میرا نام علی ہے
    urdunlp to-roman "میں ٹھیک ہوں"            # main theek hoon
    urdunlp langid "هي ڪتاب منهنجو آهي"        # sd  Sindhi ...
    urdunlp normalize "كتاب"                 # کتاب
    echo "kal milte hain" | urdunlp to-urdu  # reads standard input, line by line

Also `python -m urdunlp`.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Iterable

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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="urdunlp",
        description="Urdu and Roman Urdu text processing. Each command takes TEXT as an "
        "argument, or reads standard input one line at a time.",
    )
    parser.add_argument("--version", action="version", version=f"urdunlp {__version__}")
    commands = parser.add_subparsers(dest="command", metavar="COMMAND", required=True)

    def command(name: str, summary: str, make: _Factory) -> argparse.ArgumentParser:
        sub = commands.add_parser(name, help=summary, description=summary)
        sub.add_argument("text", nargs="*", help="text to process (default: standard input)")
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


def _lines(text: list[str]) -> Iterable[str]:
    if text:
        yield " ".join(text)
        return
    for line in sys.stdin:
        yield line.rstrip("\r\n")


def main(argv: list[str] | None = None) -> int:
    # Urdu on a Windows console: the default code page cannot encode it, and print()
    # would raise UnicodeEncodeError on the first letter.
    for stream in (sys.stdin, sys.stdout):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")
    parser = _parser()
    args = parser.parse_args(argv)
    if not args.text and sys.stdin.isatty():
        # Waiting silently on a terminal looks like a hang.
        parser.error(f"give TEXT, or pipe it in: echo TEXT | urdunlp {args.command}")
    run = args.make(args)
    try:
        for line in _lines(args.text):
            print(run(line), flush=True)
    except KeyboardInterrupt:
        return 130
    except BrokenPipeError:  # `urdunlp to-urdu < big.txt | head`
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
