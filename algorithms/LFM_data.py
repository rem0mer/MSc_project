# coding: utf-8 -*-

import pandas as pd
import pickle
import os

class DataProcessing:
    def __init__(self):
        pass

    def process(self):
        print('Start converting user data（users.dat）...')
        self.process_user_data()
        print('Start converting movie data（movies.dat）...')
        self.process_movies_date()
        print('Start converting user-to-movie rating data（ratings.dat）...')
        self.process_rating_data()
        print('Over!')

    def process_user_data(self, file='../data/ml-1m/users.dat'):
        if not os.path.exists("data/users.csv"):
            fp = pd.read_table(file, sep='::', engine='python',names=['userID', 'Gender', 'Age', 'Occupation', 'Zip-code'])
            fp.to_csv('data/users.csv', index=False)

    def process_rating_data(self, file='../data/ml-1m/ratings.dat'):
        if not os.path.exists("data/ratings.csv"):
            fp = pd.read_table(file, sep='::', engine='python',names=['UserID', 'MovieID', 'Rating', 'Timestamp'])
            fp.to_csv('data/ratings.csv', index=False)

    def process_movies_date(self, file='../data/ml-1m/movies.dat'):
        if not os.path.exists("data/movies.csv"):
            fp = pd.read_table(file, sep='::', engine='python',names=['MovieID', 'Title', 'Genres'])
            fp.to_csv('data/movies.csv', index=False)

    # Data tagging of whether user have action on movie
    def get_pos_neg_item(self,file_path="data/ratings.csv"):
        if not os.path.exists("data/lfm_items.dict"):
            self.items_dict_path="data/lfm_items.dict"

            self.uiscores=pd.read_csv(file_path)
            self.user_ids=set(self.uiscores["UserID"].values)
            self.item_ids=set(self.uiscores["MovieID"].values)
            self.items_dict = {user_id: self.get_one(user_id) for user_id in list(self.user_ids)}

            fw = open(self.items_dict_path, 'wb')
            pickle.dump(self.items_dict, fw)
            fw.close()

    # Define positive and negative data for a single user
    # Positive: Movies that the user has rated; Negative: Movies that the user has not rated
    def get_one(self, user_id):
        print('Prepare positive and negative data for %s...' % user_id)
        pos_item_ids = set(self.uiscores[self.uiscores['UserID'] == user_id]['MovieID'])
        # xor
        neg_item_ids = self.item_ids ^ pos_item_ids
        neg_item_ids = list(neg_item_ids)[:len(pos_item_ids)]
        item_dict = {}
        for item in pos_item_ids: item_dict[item] = 1
        for item in neg_item_ids: item_dict[item] = 0
        return item_dict

if __name__ == '__main__':
    dp=DataProcessing()
    dp.process()
    dp.get_pos_neg_item()
