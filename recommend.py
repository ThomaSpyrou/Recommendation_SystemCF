import pandas as pd
import numpy as np
import random
import ast
import itertools
from scipy.spatial import distance


def read_merged_csv_file():
    """
    Read the new csv file.
    :return: dataframe all data from CSV
    """
    try:
        all_data = pd.read_csv('all_ratings_merged.csv', delimiter='\t', encoding='utf-8')
        books = pd.read_csv('books.csv', delimiter='\t', encoding='utf-8')

        return all_data, books

    except AttributeError:
        print("CSV files are not in proper format.")


def recommend(merged_data, books):
    """
    :param merged_data: dataframe all the data merged
    :return:
    """
    random_indexes = random.sample(range(0, len(merged_data)), 5)
    users_id = []
    for item in random_indexes:
        users_id.append(merged_data.iloc[item].userId)

    users = (merged_data.loc[merged_data['userId'].isin(users_id)])
    users = users.sort_values(by=['userId', 'bookRating'], ascending=[True, False])  # sort them in order to the top exist the biggest rate

    # users with all book ratings
    users_with_all_book_ratings = users.groupby('userId').agg(lambda x: x.tolist()).reset_index()

    # users with the 3 bigger ratings
    users_after_reduction = users.groupby('userId', group_keys=False).apply(lambda c: c.nlargest(3, 'bookRating')) # choose the first 3 book ratings for each user
    users_after_reduction.bookTitle = users_after_reduction.bookTitle.apply(ast.literal_eval)

    recommend_list = users_after_reduction.groupby('userId').agg(lambda x: x.tolist()).reset_index()
    books.bookTitle = books.bookTitle.apply(ast.literal_eval)

    # recommend_list columns: userId, ISBN(b), bookRating, bookTitle(b),
    # bookAuthor(b), yearOfPublication(b), publisher(b), location, age

    recommend_list.bookTitle = recommend_list.bookTitle.apply(np.concatenate)
    jac_sim_list = []
    dice_sim_list = []
    for index, value in books.iterrows():
        jacc_similarity = 0
        dice_sim = 0
        for inner_index, inner_value in recommend_list.iterrows():
            if value.ISBN not in inner_value.ISBN:
                if value.bookAuthor in inner_value.bookAuthor:
                    jacc_similarity += 0.4
                    dice_sim += 0.3
                jacc_similarity += jac_similarity(value.bookTitle, inner_value.bookTitle) * 0.2
                dice_sim += dice_similarity(value.bookTitle, inner_value.bookTitle) * 0.5
                year_sim = 0
                for item in range(len(inner_value.yearOfPublication)):
                    temp_year_sim = 0
                    temp_year_sim = 1 - (abs(value.yearOfPublication - inner_value.yearOfPublication[item])/2005)
                    if temp_year_sim >= year_sim:
                        year_sim = temp_year_sim
                jacc_similarity += year_sim * 0.4
                dice_sim += year_sim * 0.2

                jac_sim_list.append([jacc_similarity, value.ISBN, inner_value.userId])
                dice_sim_list.append([dice_sim, value.ISBN, inner_value.userId])

                jacc_similarity = 0
                dice_sim = 0
            else:
                continue

    return jac_sim_list, dice_sim_list, users_id


def jac_similarity(books_list, to_recommend_list):
    s1 = set(books_list)
    s2 = set(to_recommend_list)
    return len(s1.intersection(s2)) / len(s1.union(s2)) - len(s1.intersection(s2))


def dice_similarity(books_list, to_recommend_list):
    s1 = set(books_list)
    s2 = set(to_recommend_list)
    return float(2 * len(s1.intersection(s2)) / (len(s1) + len(s2)))


def write_to_file(jac_sim_list, dice_sim_list, users):
    jac = pd.DataFrame(jac_sim_list, columns=['jaccard', 'ISBN', 'userId'])
    dice = pd.DataFrame(dice_sim_list, columns=['dice_sim', 'ISBN', 'userId'])

    # users = users.sort_values(by=['userId', 'bookRating'], ascending=[True, False])

    jac = jac.sort_values(by=['jaccard', 'userId'], ascending=[True, False])
    dice = dice.sort_values(by=['dice_sim', 'userId'], ascending=[True, False])

    final_jac_ten_per_user = pd.DataFrame(columns=['jaccard', 'ISBN', 'userId'])
    final_dice_ten_per_user = pd.DataFrame(columns=['dice_sim', 'ISBN', 'userId'])

    for item in users:
        jac_users = jac.loc[jac['userId'] == item]
        dice_users = dice.loc[dice['userId'] == item]
        user_jac_ten = jac_users.nlargest(10, ['jaccard'])
        user_dice_ten = dice_users.nlargest(10, ['dice_sim'])
        final_jac_ten_per_user = pd.concat([user_jac_ten, final_jac_ten_per_user])
        final_dice_ten_per_user = pd.concat([user_dice_ten, final_dice_ten_per_user])

    final_jac_ten_per_user.to_csv('recommended_by_jaccard2.csv', sep='\t', encoding='utf-8', index=False)
    final_dice_ten_per_user.to_csv('recommended_by_dice2.csv', sep='\t', encoding='utf-8', index=False)


if __name__ == '__main__':
    merged_data, books = read_merged_csv_file()
    jac_sim_list, dice_sim_list, users = recommend(merged_data, books)
    write_to_file(jac_sim_list, dice_sim_list, users)
