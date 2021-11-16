import argparse
from BronchoTrack.data import dataorg
from BronchoTrack.utils import fix_randseed

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", dest="root", type=str, required=True)
    parser.add_argument("--new-root", dest="new_root", type=str, required=True)
    parser.add_argument("--clean", dest="clean", action="store_true")
    parser.add_argument("--n-trajectories", dest="n_trajectories", type=int, default=75)
    parser.add_argument("--compute-statistics", dest="compute_statistics", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    fix_randseed(42)
    args = parse()
    organizer = dataorg.BronchoOrganizer(
        args.root,
        args.new_root,
        n_trajectories=args.n_trajectories,
        clean=args.clean,
    )
    if args.compute_statistics:
        organizer.compute_statistics()
    else:
        organizer.create_csvs()
