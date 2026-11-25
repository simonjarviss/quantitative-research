"""`python3 -m r2 --ping | --sql ... | --tables ...`: the CLI in r2.main."""
import sys

from r2 import main

sys.exit(main())
