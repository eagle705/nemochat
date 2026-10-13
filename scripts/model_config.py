"""Print the approved A/B dimensions and analytical parameter estimates."""

import argparse
import json
from dataclasses import asdict

from nemochat.models.dimensions import BRIDGE_REVISION, dimensions, parameter_estimate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["A", "B"], default="A")
    parser.add_argument("--build-provider", action="store_true",
                        help="Requires the pinned GPU SDK environment; does not create weights")
    args = parser.parse_args()
    model = dimensions(args.variant)
    print(json.dumps({"dimensions": asdict(model), "estimate": parameter_estimate(model),
                      "bridge_revision": BRIDGE_REVISION}, indent=2))
    if args.build_provider:
        from nemochat.models.bridge_provider import build_provider
        provider = build_provider(args.variant)
        print(provider)


if __name__ == "__main__":
    main()
