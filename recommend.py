import pandas as pd
import numpy as np
import random
import ast
from collections import Counter


def read_merged_csv_file():
    """
    Read the created CSV files, of which the data is processed.
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
    :param books: dataframe all books to recommend, values in cells processed.
    :return: estimate the recommendations for both similarities for each user. list of list pattern [[userId , jaccard/dice, isbn]]
    """
    random_indexes = random.sample(range(0, len(merged_data)), 5)
    users_id = []
    for item in random_indexes:
        users_id.append(merged_data.iloc[item].userId)

    users = (merged_data.loc[merged_data['userId'].isin(users_id)])
    # sort them in order to the top exist the biggest rate
    users = users.sort_values(by=['userId', 'bookRating'], ascending=[True, False])

    # users with all book ratings
    users_with_all_book_ratings = users.groupby('userId').agg(lambda x: x.tolist()).reset_index()
    users_with_all_book_ratings.rename(columns={'ISBN': 'read'}, inplace=True)

    # users with the 3 bigger ratings, choose the first 3 book ratings for each user
    users_after_reduction = users.groupby('userId', group_keys=False).apply(lambda c: c.nlargest(3, 'bookRating'))
    users_after_reduction.bookTitle = users_after_reduction.bookTitle.apply(ast.literal_eval)

    recommend_list = users_after_reduction.groupby('userId').agg(lambda x: x.tolist()).reset_index()
    recommend_list = recommend_list.join(users_with_all_book_ratings.read)
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
            if value.ISBN not in inner_value.read:
                if value.bookAuthor in inner_value.bookAuthor:
                    jacc_similarity += 0.4
                    dice_sim += 0.3
                jacc_similarity += jac_similarity(value.bookTitle, inner_value.bookTitle) * 0.2
                dice_sim += dice_similarity(value.bookTitle, inner_value.bookTitle) * 0.5
                year_sim = 0
                for item in range(len(inner_value.yearOfPublication)):
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
    """
    :param books_list: list
    :param to_recommend_list: list
    :return: jaccard similarity
    """
    s1 = set(books_list)
    s2 = set(to_recommend_list)
    return len(s1.intersection(s2)) / len(s1.union(s2))


def dice_similarity(books_list, to_recommend_list):
    """
    :param books_list: list
    :param to_recommend_list: list
    :return: dice similarity
    """
    s1 = set(books_list)
    s2 = set(to_recommend_list)
    return float(2 * len(s1.intersection(s2)) / (len(s1) + len(s2)))


def write_to_file(jac_sim_list, dice_sim_list, users):
    """
    :param jac_sim_list: list od lists
    :param dice_sim_list: list of lists
    :param users: list
    :return: write recommendations to CSV files return the 10 books per user in a Dataframe
    """
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

    final_jac_ten_per_user.to_csv('recommended_by_jaccard.csv', sep='\t', encoding='utf-8', index=False)
    final_dice_ten_per_user.to_csv('recommended_by_dice.csv', sep='\t', encoding='utf-8', index=False)

    return final_jac_ten_per_user, final_dice_ten_per_user


def overlap_between_sims(jaccard, dice):
    """
    :param jaccard: dataframe
    :param dice: dataframe
    :return: print output
    """
    jaccard_per_user = jaccard.groupby('userId').agg(lambda x : x.tolist()).reset_index()

    dice_per_user = dice.groupby('userId').agg(lambda x: x.tolist()).reset_index()
    dice_per_user.rename(columns={'ISBN': 'isbndice'}, inplace=True)

    joined_jac_dice = jaccard_per_user.join(dice_per_user.isbndice)

    # a = jaccard['ISBN'].isin(dice['ISBN']).value_counts()
    diff = {}
    for index, value in joined_jac_dice.iterrows():
        res = list((Counter(value['ISBN']) - Counter(value['isbndice'])).elements())
        diff[value['userId']] = len(res)

    for item in diff:
        print("User with the id:", item, "has", 10 - diff[item],  "same recommended books for jaccard and dice:")

    print(40 * '-')
    over_jac = []
    for index, value in jaccard_per_user.iterrows():
        for inner_index, inner_value in jaccard_per_user.iterrows():
            if value.userId != inner_value.userId:
                temp_over = len(set(value['ISBN']).intersection(inner_value['ISBN']))
                over_jac.append([value.userId, inner_value.userId, temp_over])

    for item in over_jac:
        print("With jaccard similarity. User:", item[0], "has", item[2], "out of ten in common with the user:", item[1])

    print(40 * '-')
    over_dice = []
    for index, value in jaccard_per_user.iterrows():
        for inner_index, inner_value in jaccard_per_user.iterrows():
            if value.userId != inner_value.userId:
                temp_over = len(set(value['ISBN']).intersection(inner_value['ISBN']))
                over_dice.append([value.userId, inner_value.userId, temp_over])

    for item in over_dice:
        print("With dice similarity. User:", item[0], "has", item[2], "out of ten in common with the user:", item[1])


if __name__ == '__main__':
    print("Recommendation started!")
    merged_data, books = read_merged_csv_file()
    jac_sim_list, dice_sim_list, users = recommend(merged_data, books)
    jacc, dice = write_to_file(jac_sim_list, dice_sim_list, users)
    overlap_between_sims(jacc, dice)
    print('Recommendations have been wrote in two CSV files.')
