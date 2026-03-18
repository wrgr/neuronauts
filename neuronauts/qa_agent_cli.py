"""Entry point for the ``neuronauts-qa`` console script."""

import sys
from pathlib import Path

# When installed as a package the scripts/ directory won't be on sys.path,
# but scripts/qa_agent.py imports this package so we just re-export its main.

def main(argv=None) -> int:
    _scripts = Path(__file__).resolve().parent.parent / "scripts"
    if str(_scripts) not in sys.path:
        sys.path.insert(0, str(_scripts))

    from qa_agent import main as _main  # type: ignore[import]
    return _main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
