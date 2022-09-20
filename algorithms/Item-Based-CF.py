# -*- coding: utf-8 -*-

import random
import math
import os
import json

class ItemCFRec:
    def __init__(self,datafile,ratio):
        self.datafile = datafile
        self.ratio = ratio

        self.data = self.loadData()
        self.trainData,self.testData = self.splitData(3,47)
        self.items_sim = self.ItemSimilarityBest()

    def loadData(self):
        print("loading...")
        data=[]
        for line in open(self.datafile):
            userid,itemid,record,_ = line.split("::")
            data.append((userid,itemid,int(record)))
        return data


    def splitData(self,k,seed,M=9):
        print("Splitting training and testing datasets...")
        train,test = {},{}
        random.seed(seed)
        for user,item,record in self.data:
            if random.randint(0,M) == k:
                test.setdefault(user,{})
                test[user][item] = record
            else:
                train.setdefault(user,{})
                train[user][item] = record
        return train,test

    def ItemSimilarityBest(self):
        print("Start calculating similarity between items")
        if os.path.exists("data/item_sim.json"):
            print("Item similarity is loaded from file ...")
            itemSim = json.load(open("data/item_sim.json", "r"))
        else:
            itemSim = dict()
            item_user_count = dict()  # Get how many users have acted on each item
            count = dict()  # co-occurrence matrix
            for user, item in self.trainData.items():
                print("user is {}".format(user))
                for i in item.keys():
                    item_user_count.setdefault(i, 0)
                    if self.trainData[str(user)][i] > 0.0:
                        item_user_count[i] += 1
                    for j in item.keys():
                        count.setdefault(i, {}).setdefault(j, 0)
                        if self.trainData[str(user)][i] > 0.0 and self.trainData[str(user)][j] > 0.0 and i != j:
                            count[i][j] += 1
            # co-occurrence matrix -> similarity matrix
            for i, related_items in count.items():
                itemSim.setdefault(i, dict())
                for j, cuv in related_items.items():
                    itemSim[i].setdefault(j, 0)
                    itemSim[i][j] = cuv / math.sqrt(item_user_count[i] * item_user_count[j])
        json.dump(itemSim, open('data/item_sim.json', 'w'))
        return itemSim


    def recommend(self, user, k=8, nitems=40):
        result = dict()
        u_items = self.trainData.get(user, {})
        for i, pi in u_items.items():
            for j, wj in sorted(self.items_sim[i].items(), key=lambda x: x[1], reverse=True)[0:k]:
                if j in u_items:
                    continue
                result.setdefault(j, 0)
                result[j] += pi * wj

        return dict(sorted(result.items(), key=lambda x: x[1], reverse=True)[0:nitems])


    def precision(self, k=8,nitems=10):
        print("Start calculating precision ...")
        hit = 0
        precision = 0
        for user in self.testData.keys():
            u_items = self.testData.get(user, {})
            result = self.recommend(user, k=k, nitems=nitems)
            for item, rate in result.items():
                if item in u_items:
                    hit += 1
            precision += nitems
        return hit / (precision * 1.0)


if __name__ == "__main__":
    ib = ItemCFRec("../data/ml-1m/ratings.dat",[1,9])
    print("The result of user 1's recommendation is as follows：{}".format(ib.recommend("1")))
    print("The precision is： {}".format(ib.precision()))

