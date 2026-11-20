"""Allow ``python -m rdv`` to behave like the ``rdv`` console script."""

from rdv.entrypoints.cli.app import main

if __name__ == "__main__":
    main()
