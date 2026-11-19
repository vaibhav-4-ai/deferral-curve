import argparse


def main() -> None:
    parser = argparse.ArgumentParser(prog="dc")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("eval", help="measure how often retrieval finds the right help page")
    sub.add_parser("run", help="run the agent on every test question and save the results")
    sub.add_parser("calibrate", help="fit the hand-off rule and draw the risk-coverage curve")
    sub.add_parser("serve", help="start the web app")
    sub.add_parser("cmi-experiment", help="does the question's code-mixing (CMI) improve the hand-off?")
    sub.add_parser("fetch-codemix", help="download the Hinglish model for /codemix (at image build time)")
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
    elif args.command == "cmi-experiment":
        from dc.cmi_experiment import run

        run()
    elif args.command == "fetch-codemix":
        from dc.codemix import fetch

        fetch()
    elif args.command == "serve":
        import uvicorn

        uvicorn.run("app.main:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
