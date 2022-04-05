import argparse
from BronchoTrack.data import dataorg
from BronchoTrack.utils import fix_randseed


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", dest="root", type=str, required=True, help="root for all data, images and csv")
    parser.add_argument("--new-root", dest="new_root", type=str, required=True, help="where to organize new csv with sequences")
    parser.add_argument("--clean", dest="clean", action="store_true", help="to clean or not previously generated sequences")
    parser.add_argument("--n-trajectories", dest="n_trajectories", type=int, default=75, help="number of trajectories selected per lobe and patient")
    parser.add_argument("--test-patient", dest="test_pacient", type=str, choices=["P18", "P25", "P21", "P30", "P20"], default="P18", help="which patient to held out as validation")
    parser.add_argument("--only-val", dest="only_val", action="store_true", default=False, help="if true, we test in the validation set")
    parser.add_argument("--intra-patient", dest="intra_patient", action="store_true", default=False, help="if true, validation is performed with the same patients but different sequences")
    parser.add_argument("--length", dest="length", type=int, default=2, help="length of each sample fed to model")
    parser.add_argument("--rotate-patient", dest="rotate_patient", action="store_true", default=False, help="If dataset has been already created then we rotate the test patient")
    parser.add_argument("--save-indexes", dest="save_indexes", action="store_true", default=False, help="Stores indexes of trajectories to be able to rotate patients with the same trajectories")
    return parser.parse_args()


if __name__ == "__main__":
    fix_randseed(42)
    args = parse()
    organizer = dataorg.BronchoOrganizer(
        args.root,
        args.new_root,
        n_trajectories=args.n_trajectories,
        clean=args.clean,
        test_pacient=args.test_pacient,
        only_val=args.only_val,
        intra_patient=args.intra_patient,
        length=args.length,
        rotate_patient=args.rotate_patient,
        save_indexes=args.save_indexes
    )
    organizer.create_csvs()
    
