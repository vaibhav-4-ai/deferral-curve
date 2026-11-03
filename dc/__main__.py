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
    elif args.command == "calibrate":
        from dc.curve import calibrate

        calibrate()
    elif args.command == "run":
        from dc.run_eval import run

        run()
    elif args.command == "serve":
        import uvicorn

        uvicorn.run("app.main:app", host="0.0.0.0", port=7860)
    else:
        print(f"{args.command}: not implemented")


if __name__ == "__main__":
    main()
