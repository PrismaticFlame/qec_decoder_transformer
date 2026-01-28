import os
import sys


def file_read(file_name, search_term):
    try:
        for file in file_name:
            with open(file, 'r') as f:
                for (i, line) in enumerate(f):
                    if search_term in line:
                        print(f"{file} | {i}: {line}", end='')
    except Exception as e:
        print(f"Error: {e}")


def main():
    locations = file_read(file_name=sys.argv[1:-1], search_term=sys.argv[-1])

if __name__ == "__main__":
    main()