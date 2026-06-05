"""Allow `python -m pycut` invocation."""

import sys

from pycut.cli import console_main

if __name__ == "__main__":
    sys.exit(console_main())
