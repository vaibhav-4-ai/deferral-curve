import argparse


def main() -> None:
    parser = argparse.ArgumentParser(prog="dc")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("ingest", "eval", "calibrate", "serve"):
        sub.add_parser(name)
    args = parser.parse_args()
    if args.command == "eval":
        from dc.eval_retrieval import report

        report()
    else:
        print(f"{args.command}: not implemented")


if __name__ == "__main__":
    main()
