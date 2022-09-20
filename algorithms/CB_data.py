# coding: utf-8 -*-

import pandas as pd
import json
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
        if os.path.exists("data/users.csv"):
            print("user.csv already exists")
        fp = pd.read_table(file, sep='::', engine='python',names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'])
        fp.to_csv('data/users.csv', index=False)

    def process_rating_data(self, file='../data/ml-1m/ratings.dat'):
        if os.path.exists("data/ratings.csv"):
            print("ratings.csv already exists")
        fp = pd.read_table(file, sep='::', engine='python',names=['UserID', 'MovieID', 'Rating', 'Timestamp'])
        fp.to_csv('data/ratings.csv', index=False)

    def process_movies_date(self, file='../data/ml-1m/movies.dat'):
        if os.path.exists("data/movies.csv"):
            print("movies.csv already exists")
        fp = pd.read_table(file, sep='::', engine='python',names=['MovieID', 'Title', 'Genres'])
        fp.to_csv('data/movies.csv', index=False)

    # Get the feature information matrix of item
    def prepare_item_profile(self,file='data/movies.csv'):
        items=pd.read_csv(file)
        item_ids=set(items["MovieID"].values)
        self.item_dict={}
        genres_all=list()
        # Put the genre of each movie in item_dict
        for item in item_ids:
            genres=items[items["MovieID"]==item]["Genres"].values[0].split("|")
            self.item_dict.setdefault(item,[]).extend(genres)
            genres_all.extend(genres)
        self.genres_all=set(genres_all)
        # Store the feature information matrix of each movie in self.item_matrix
        self.item_matrix={}
        for item in self.item_dict.keys():
            self.item_matrix[str(item)]=[0] * len(set(self.genres_all))
            for genre in self.item_dict[item]:
                index=list(set(genres_all)).index(genre)
                self.item_matrix[str(item)][index]=1
        json.dump(self.item_matrix,
                  open('data/item_profile.json','w'))
        print("The item information calculation is completed, and the save path is：{}"
              .format('data/item_profile.json'))

    # Calculate the user's preference matrix
    def prepare_user_profile(self,file='data/ratings.csv'):
        users = pd.read_csv(file)
        user_ids = set(users["UserID"].values)
        # Convert users information into dict
        users_rating_dict={}
        for user in user_ids:
            users_rating_dict.setdefault(str(user),{})
        with open(file,"r") as fr:
            for line in fr.readlines():
                if not line.startswith("UserID"):
                    (user,item,rate)=line.split(",")[:3]
                    users_rating_dict[user][item]=int(rate)

        # Get user ratings for which movies under each genre
        self.user_matrix={}
        # iterate over each user
        for user in users_rating_dict.keys():
            print("user is {}".format(user))
            score_list=users_rating_dict[user].values()
            # avg
            avg=sum(score_list)/len(score_list)
            self.user_matrix[user]=[]
            # Traverse each type
            # (ensure that the types represented by each column in the item_profile
            # and user_profile information matrices are consistent)
            for genre in self.genres_all:
                score_all=0.0
                score_len=0
                for item in users_rating_dict[user].keys():
                    if genre in self.item_dict[int(item)]:
                        score_all += (users_rating_dict[user][item]-avg)
                        score_len+=1
                if score_len==0:
                    self.user_matrix[user].append(0.0)
                else:
                    self.user_matrix[user].append(score_all / score_len)
        json.dump(self.user_matrix,
                  open('data/user_profile.json', 'w'))
        print("user information calculation is completed, and the save path is：{}"
              .format('data/user_profile.json'))

if __name__ == '__main__':
    dp=DataProcessing()
    dp.process()
    dp.prepare_item_profile()
    dp.prepare_user_profile()
