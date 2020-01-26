import pandas as pd
import numpy as np


def read_merged_csv_file():
    """
    Read the new csv file.
    :return: dataframe all data from CSV
    """
    try:
        all_data = pd.read_csv('all_ratings_merged.csv', delimiter='\t', encoding='latin1')

        return all_data

    except AttributeError:
        print("CSV files are not in proper format.")


if __name__ == '__main__':
    read_merged_csv_file()