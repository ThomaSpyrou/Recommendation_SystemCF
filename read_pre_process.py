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
import re
import nltk


def read_data():
    """
    Read the data from CSV file. The type of the data is Dataframe from pandas. Set used columns.
    :return: The data from the 3 CSV in 3 different Dataframe variables.
    """
    try:
        book_ratings = pd.read_csv('BX-CSV-Dump/BX-Book-Ratings.csv', delimiter=';', escapechar='\\', encoding='latin1')
        book_ratings.columns = ['userId', 'ISBN', 'bookRating']

        users = pd.read_csv('BX-CSV-Dump/BX-Users.csv', delimiter=';', escapechar='\\', encoding='latin1')
        users.columns = ['userId', 'location', 'age']

        books = pd.read_csv('BX-CSV-Dump/BX-Books.csv', delimiter=";", escapechar='\\', error_bad_lines=False, encoding='latin1')
        books.columns = ['ISBN', 'bookTitle', 'bookAuthor', 'yearOfPublication', 'publisher', 'imageURLS', 'imageURLM',
                         'imageURLL']

        books.drop(['imageURLS', 'imageURLM', 'imageURLL'], axis=1, inplace=True)

        return book_ratings, users, books

    except AttributeError:
        print("CSV files are not in proper format.")


def stem_and_token(list_of_words):
    """
    :param list_of_words: a list of strings
    :return: the list of string stemmed and tokenized
    """
    stemming = PorterStemmer()
    cleanString = re.sub(r'[^A-Za-z]', ' ', list_of_words)
    tokenized_list_of_string = nltk.word_tokenize(cleanString)
    word_list = []
    for item in tokenized_list_of_string:
        word_list.append(stemming.stem(item.lower()))

    return word_list


def pre_processing(book_ratings, users, books):
    """
    Delete books with less than 5 ratings and  users that have rate less than 10 books. Stem the the keywords
    :param book_ratings: dataframe (pd)
    :param users: dataframe (pd)
    :param books: dataframe (pd)
    :return: processed data
    """
    count_userId = book_ratings.groupby("userId")["userId"].transform(len)
    count_ISBN = book_ratings.groupby("ISBN")["ISBN"].transform(len)

    mask_for_book_ratings = (count_userId >= 5) & (count_ISBN >= 10)
    final_book_ratings = book_ratings[mask_for_book_ratings]
    new_books = books[(count_ISBN >= 10)]
    new_books['bookTitle'] = new_books['bookTitle'].apply(lambda title: stem_and_token(title))
    new_users = users[(count_userId >= 5)]

    return final_book_ratings, new_users, new_books


def merge_tables(final_book_ratings, new_users, new_books):
    """
    Using panda's merge to create one table. And write to csv file in order to use it in the other .py file.
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
    print('Pre-processing started!')
    book_ratings, users, books = read_data()
    final_book_ratings, new_users, new_books = pre_processing(book_ratings, users, books)
    merge_tables(final_book_ratings, new_users, new_books)
    print('Pre-processing just completed, two new CSV files have been created.\n --Run recommend.py--')
