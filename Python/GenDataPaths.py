import argparse
import os
import re
from typing import List


def main():
    try:
        parser: argparse.ArgumentParser = argparse.ArgumentParser()
        parser.add_argument("data_dir",
                            type=str,
                            help="Path to a directory containing all the data files whose full paths are to be "
                                 "aggregated.")
        parser.add_argument("--output_name",
                            type=str,
                            help="Name of the output file, including extension.",
                            required=False,
                            default="DataPaths.txt")
        parser.add_argument("--selector_expression",
                            type=str,
                            help="A regular expression pattern to select valid data files.",
                            required=False,
                            default=".*")
        args = parser.parse_args()

        data_dir: str = args.data_dir
        if not os.path.isdir(data_dir):
            print(f"The given data directory: {data_dir} was not discoverable.")
            exit(1)

        output_name: str = args.output_name
        selector_expression: str = args.selector_expression
        data_paths: List[str] = []
        for (root, file_path, file_names) in os.walk(data_dir):
            for file_name in file_names:
                try:
                    if selector_expression == ".*" or re.match(selector_expression, file_name):
                        data_paths.append(os.path.join(root, file_name) + "\n")
                except Exception as exception:
                    print(f"[WARNING] {exception}")

        data_paths.sort()

        with open(output_name, "w+") as data_paths_file:
            data_paths_file.writelines(data_paths)
    except Exception as exception:
        print(exception)
        exit(1)


if __name__ == "__main__":
    main()
