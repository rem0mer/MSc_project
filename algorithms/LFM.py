# coding: utf-8 -*-
import math
import random
import pickle
import pandas as pd
import numpy as np
from math import exp
import time

class LFM:

    def __init__(self):
        self.class_count = 5
        self.iter_count = 5
        self.lr = 0.02
        self.lam = 0.01
        self._init_model()


    def _init_model(self):
        file_path = 'data/ratings.csv'
        pos_neg_path = 'data/lfm_items.dict'

        self.uiscores = pd.read_csv(file_path)
        self.user_ids = set(self.uiscores['UserID'].values)  # 6040
        self.item_ids = set(self.uiscores['MovieID'].values) # 3706
        self.items_dict = pickle.load(open(pos_neg_path,'rb'))

        array_p = np.random.randn(len(self.user_ids), self.class_count)
        array_q = np.random.randn(len(self.item_ids), self.class_count)
        self.p = pd.DataFrame(array_p, columns=range(0, self.class_count), index=list(self.user_ids))
        self.q = pd.DataFrame(array_q, columns=range(0, self.class_count), index=list(self.item_ids))


    def _predict(self, user_id, item_id):
        p = np.mat(self.p.loc[user_id].values)
        q = np.mat(self.q.loc[item_id].values).T
        r = (p * q).sum()
        # use sigmoid function，transfer data
        logit = 1.0 / (1 + exp(-r))
        return logit

    # loss function(SSE)
    def _loss(self, user_id, item_id, y, step):
        e = y - self._predict(user_id, item_id)
        # print('Step: {}, user_id: {}, item_id: {}, y: {}, loss: {}'.format(step, user_id, item_id, y, e))
        return e

    def _optimize(self, user_id, item_id, e):
        gradient_p = -e * self.q.loc[item_id].values
        l2_p = self.lam * self.p.loc[user_id].values
        delta_p = self.lr * (gradient_p + l2_p)

        gradient_q = -e * self.p.loc[user_id].values
        l2_q = self.lam * self.q.loc[item_id].values
        delta_q = self.lr * (gradient_q + l2_q)

        self.p.loc[user_id] -= delta_p
        self.q.loc[item_id] -= delta_q

    def train(self):
        for step in range(0, self.iter_count):
            time.sleep(30)
            for user_id, item_dict in self.items_dict.items():
                print('Step: {}, user_id: {}'.format(step, user_id))
                item_ids = list(item_dict.keys())
                random.shuffle(item_ids)
                for item_id in item_ids:
                    e = self._loss(user_id, item_id, item_dict[item_id], step)
                    self._optimize(user_id, item_id, e)
            self.lr *= 0.9
        self.save()

    # Calculate the movies that the user has not rated, and return the top N to the user
    def predict(self, user_id, top_n=10):
        self.load()
        user_item_ids = set(self.uiscores[self.uiscores['UserID'] == user_id]['MovieID'])
        other_item_ids = self.item_ids ^ user_item_ids # xor
        interest_list = [self._predict(user_id, item_id) for item_id in other_item_ids]
        candidates = sorted(zip(list(other_item_ids), interest_list), key=lambda x: x[1], reverse=True)
        return candidates[:top_n]

    # save model
    def save(self):
        f = open('data/lfm.model', 'wb')
        pickle.dump((self.p, self.q), f)
        f.close()

    # load model
    def load(self):
        f = open('data/lfm.model', 'rb')
        self.p, self.q = pickle.load(f)
        f.close()

    # AE evaluate
    def evaluate(self):
        self.load()
        users=random.sample(self.user_ids,10)
        user_dict={}
        for user in users:
            user_item_ids = set(self.uiscores[self.uiscores['UserID'] == user]['MovieID'])
            _sum=0.0
            for item_id in user_item_ids:
                p = np.mat(self.p.loc[user].values)
                q = np.mat(self.q.loc[item_id].values).T
                _r = (p * q).sum()
                r=self.uiscores[(self.uiscores['UserID'] == user)
                                & (self.uiscores['MovieID']==item_id)]["Rating"].values[0]
                _sum+=math.pow(abs(r-_r),2)
            user_dict[user] = _sum/len(user_item_ids)
            print("userID：{},MSE：{}".format(user,user_dict[user]))

        return sum(user_dict.values())/len(user_dict.keys())

if __name__=="__main__":
    lfm=LFM()
    #lfm.train()
    #print(lfm.predict(6027,10))
    print(lfm.evaluate())




