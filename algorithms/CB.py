# coding: utf-8 -*-

import json
import pandas as pd
import numpy as np
import math
import random

class CBRecommend:
    # load data from CB-data
    def __init__(self,K):
        # Number of items recommended to users
        self.K = K
        self.item_profile=json.load(open("data/item_profile.json","r"))
        self.user_profile=json.load(open("data/user_profile.json","r"))

    # Get a list of items that the user has not rated
    def get_none_score_item(self,user):
        items=pd.read_csv("data/movies.csv")["MovieID"].values
        data = pd.read_csv("data/ratings.csv")
        have_score_items=data[data["UserID"]==user]["MovieID"].values
        none_score_items=set(items)-set(have_score_items)
        return none_score_items

    # Get the user's preference for the item
    def cosUI(self,user,item):
        Uia=sum(
            np.array(self.user_profile[str(user)])
            *
            np.array(self.item_profile[str(item)])
        )
        Ua=math.sqrt( sum( [ math.pow(one,2) for one in self.user_profile[str(user)]] ) )
        Ia=math.sqrt( sum( [ math.pow(one,2) for one in self.item_profile[str(item)]] ) )
        return  Uia / (Ua * Ia)

    # Movie recommendation for users
    def recommend(self,user):
        user_result={}
        item_list=self.get_none_score_item(user)
        for item in item_list:
            user_result[item]=self.cosUI(user,item)
        if self.K is None:
            result = sorted(
                user_result.items(), key= lambda k:k[1], reverse=True
            )
        else:
            result = sorted(
                user_result.items(), key= lambda k:k[1], reverse=True
            )[:self.K]
        print(result)

    # Recommendation system performance evaluation
    def evaluate(self):
        evas=[]
        data = pd.read_csv("data/ratings.csv")
        # Randomly select 20 users for effect evaluation
        for user in random.sample([one for one in range(1,6040)], 12):
            have_score_items=data[data["UserID"] == user]["MovieID"].values
            items=pd.read_csv("data/movies.csv")["MovieID"].values

            user_result={}
            for item in items:
                user_result[item]=self.cosUI(user,item)
            results = sorted(
                user_result.items(), key=lambda k: k[1], reverse=True
            )[:len(have_score_items)]
            rec_items=[]
            for one in results:
                rec_items.append(one[0])
            eva = len(set(rec_items) & set(have_score_items)) / len(have_score_items)
            evas.append( eva )
        return sum(evas) / len(evas)


if __name__=="__main__":
    cb=CBRecommend(K=10)
    #cb.recommend(1)
    print(cb.evaluate())