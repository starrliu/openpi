"""Serve an OpenPi CogACT-baseline checkpoint to RobotControl."""

import argparse
import logging

from openpi.serving.robotcontrol_cogact import create_app
from openpi.serving.robotcontrol_cogact import load_policy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--config", default="pi05_cogact_baseline")
    parser.add_argument("--device", default=None)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5500)
    args = parser.parse_args()

    policy = load_policy(args.checkpoint_dir, config_name=args.config, pytorch_device=args.device)
    create_app(policy).run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
