"""Shared argparse helper: every analyze.py takes a single --config flag."""


def add_config_arg(parser, default_config):
    parser.add_argument(
        "--config", default=str(default_config),
        help="Path to config.yml (single source of truth for paths & params).",
    )
    return parser
