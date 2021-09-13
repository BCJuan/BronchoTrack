import argparse
from BronchoTrack.data import dataorg


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", dest="root", type=str, required=True)
    parser.add_argument("--new-root", dest="new_root", type=str, required=True)
    parser.add_argument("--split", dest="split", action="store_true")
    parser.add_argument("--clean", dest="clean", action="store_true")
    parser.add_argument("--split-size", dest="split_size", type=int, default=10)
    parser.add_argument("--compute-statistics", dest="compute_statistics", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse()
    organizer = dataorg.BronchoOrganizer(
        args.root,
        args.new_root,
        split=args.split,
        split_size=args.split_size,
        clean=args.clean,
    )
    if args.compute_statistics:
        organizer.compute_statistics()
    else:
        organizer.create_csvs()
