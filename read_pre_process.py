"""
python version: 3.6.9
pip version: 19.0.3
To run you have install needed modules
    1)pip3 install pandas
    2)pip3 install nltk
"""

import pandas as pd
import numpy as np
from nltk.stem import PorterStemmer 


def read_data():
    """
    Read the data from CSV file. The type of the data is Dataframe pandas. Set used columns.
    :return: The data from the 3 CSV in 3 different Dataframe variables.
    """
    try:

        book_ratings = pd.read_csv('BX-CSV-Dump/BX-Book-Ratings.csv', delimiter=';', encoding='latin1')
        book_ratings.columns = ['userId', 'ISBN', 'bookRating']

        users = pd.read_csv('BX-CSV-Dump/BX-Users.csv', delimiter=';', encoding='latin1')
        users.columns = ['userId', 'location', 'age']

        books = pd.read_csv('BX-CSV-Dump/BX-Books.csv', delimiter=";", error_bad_lines=False, encoding='latin1')
        books.columns = ['ISBN', 'bookTitle', 'bookAuthor', 'yearOfPublication', 'publisher', 'imageURLS', 'imageURLM',
                         'imageURLL']

        books.drop(['imageURLS', 'imageURLM', 'imageURLL'], axis=1, inplace=True)

        return book_ratings, users, books

    except AttributeError:
        print("CSV files are not in proper format.")


def stem(cell):
    stemming = PorterStemmer()
    word_list = cell
    stemmed_list = stemming.stem(word_list)

    return stemmed_list


def pre_processing(book_ratings, users, books):
    """
    Delete books with less than 10 ratings and  users that have rate less than 5 books. Stem the the keywords
    :param book_ratings: dataframe (pd)
    :param users: dataframe (pd)
    :param books: dataframe (pd)
    :return: processed data
    """
    count_userId = book_ratings.groupby("userId")["userId"].transform(len)
    count_ISBN = book_ratings.groupby("ISBN")["ISBN"].transform(len)

    mask_for_book_ratings = (count_userId >= 10) & (count_ISBN >= 5)
    final_book_ratings = book_ratings[mask_for_book_ratings]

    new_books = books[(count_ISBN >= 5)]
    for index, value in new_books.iterrows():
        value.bookTitle = stem(value.bookTitle)

    new_users = users[(count_userId >= 10)]
    # user's age goes from nan to 244. Need to fix it. But I have to change and the book ratings **
    # new_users = new_users[(new_users.age >= 5) | (new_users.age <= 100)]

    return final_book_ratings, new_users, new_books


def merge_tables(final_book_ratings, new_users, new_books):
    """
    Using panda's merge to create one table.
    :param final_book_ratings: dataFrame
    :param new_users: dataFrame
    :param new_books: dataFrame
    :return: merged table
    """
    rating_book = pd.merge(final_book_ratings, new_books, on='ISBN')
    all_ratings = pd.merge(rating_book, new_users, on='userId')
    new_books.to_csv('books.csv', sep='\t', encoding='utf-8', index=False)
    all_ratings.to_csv('all_ratings_merged.csv', sep='\t', encoding='utf-8', index=False)


if __name__ == '__main__':
    book_ratings, users, books = read_data()
    final_book_ratings, new_users, new_books = pre_processing(book_ratings, users, books)
    merge_tables(final_book_ratings, new_users, new_books)
