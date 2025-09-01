import argparse


def main():
    try:
        parser: argparse.ArgumentParser = argparse.ArgumentParser()
        parser.add_argument("data_dir",
                            type=str,
                            help="Path to a file that requires unique entries per line.")
        args = parser.parse_args()
        data_dir: str = args.data_dir
        validation_set: set = set()
        with open(data_dir, "r+") as data_paths_file:
            for line in data_paths_file:
                if line in validation_set:
                    print("[Output]: Has duplicate.")
                    return
                validation_set.add(line)
        print("[Output]: No duplicates found!")
    except Exception as exception:
        print(exception)
        exit(1)


if __name__ == "__main__":
    main()
