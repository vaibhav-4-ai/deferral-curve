import argparse


def main() -> None:
    parser = argparse.ArgumentParser(prog="dc")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("ingest", "eval", "run", "calibrate", "serve"):
        sub.add_parser(name)
    args = parser.parse_args()
    if args.command == "eval":
        from dc.eval_retrieval import report

        report()
    elif args.command == "run":
        from dc.run_eval import run

        run()
    else:
        print(f"{args.command}: not implemented")


if __name__ == "__main__":
    main()
