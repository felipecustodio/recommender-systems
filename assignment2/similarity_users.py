from __future__ import division
import pandas
import numpy as np

def jaccard_index(first_user, second_user):

    number_of_attributes_first = len(first_user)
    number_of_attributes_second = len(second_user)

    intersection = len(list(set(first_user) & set(second_user)))

    # print(number_of_attributes_first)
    # print(number_of_attributes_second)
    # print(intersection)

    return intersection / (number_of_attributes_first + number_of_attributes_second - intersection)


# read dataset
users_data = pandas.read_csv("csv/users_data.csv")

similarity_matrix = np.zeros((users_data.shape[0], users_data.shape[0]))

for row in users_data.itertuples():

    user_index = getattr(row, "Index")
    user_attributes = []
    user_attributes.append(getattr(row, "gender"))
    user_attributes.append(getattr(row, "age"))
    user_attributes.append(getattr(row, "occupation"))
    user_attributes.append(getattr(row, "zip_code"))

    # print(user_attributes)
    
    new_users = users_data[user_index:]

    for new_row in new_users.itertuples():
        
        new_user_index = getattr(new_row, "Index")
        new_user_attributes = []
        new_user_attributes.append(getattr(new_row, "gender"))
        new_user_attributes.append(getattr(new_row, "age"))
        new_user_attributes.append(getattr(new_row, "occupation"))
        new_user_attributes.append(getattr(new_row, "zip_code"))

        # print(new_user_attributes)

        print(user_index, new_user_index)
        similarity_matrix[user_index][new_user_index] = jaccard_index(user_attributes, new_user_attributes)
        # print(similarity_matrix[user_index][new_user_index])

df = pandas.DataFrame(similarity_matrix)
df.to_csv("./csv/similarity_users.csv")
    