from BronchoTrack.data import data


if __name__ == "__main__":
    organizer = data.BronchoOrganizer("data/raw_data", "data/cleaned", split_size=10)
    organizer.create_csvs()