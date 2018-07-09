# https://www.kaggle.com/c/whats-cooking

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set_style("dark")


file_directory = "C:\\Users\\B-dragon90\\Desktop\\Github\\Kaggle\\whats-cooking"
file_name = "train.json"

dat1 = pd.read_json(file_directory + "\\" + file_name)

dat1.head()

dat1.tail()

import collections
collections.Counter(dat1["cuisine"])

sns.countplot(y = "cuisine", data = dat1)


# 설명 변수 전처리

dat1["ingredients"][0]
#dat1["ingredients"]

# NLP(Natural Language Processing; 자연어 처리)

ing = dat1["ingredients"]
ing_new = list()
ing_new_2 = list()

for ing_i in ing:
    a_list = list()
    for i in range(len(ing_i)):
        a_list.append(ing_i[i].replace(" ", ""))
    ing_new.append(" ".join(a_list))
    ing_new_2.extend(a_list)
# 단어 빈도수 체크하고, 띄어쓰기 있는 단어들을 붙여주고,
# 리스트들을 띄어쓰기로 구분된 하나의 문장으로 바꿔줌


# Count Vector & TF-IDF

from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer()   # 단어 빈도 수 체크
bag = count.fit_transform(ing_new)

from sklearn.feature_extraction.text import TfidfVectorizer
# Term Frequency (TF; 단어 빈도) * Inverse Document Frequency (IDF; 역문서 빈도)
tfidv = TfidfVectorizer()  # IDF는 한 단어가 문서 집합 전체에서 얼마나 공통적으로 나타났는지
bag_tf = tfidv.fit_transform(ing_new)


# 로지스틱 회귀 모형 적합

from sklearn.linear_model import LogisticRegression

lr_cv = LogisticRegression(C = 10)
lr_cv.fit(bag, dat1["cuisine"])

lr_tf = LogisticRegression(C = 10)
lr_tf.fit(bag_tf, dat1["cuisine"])

print("Logistic Regression 모델 적합을 완료하였습니다.")

score_cv = round(lr_cv.score(bag, dat1["cuisine"]) * 100, 3)
print("Count Vector 방식으로 했을 때 Training Accuracy는 {} 입니다.".format(score_cv))

score_tf = round(lr_tf.score(bag_tf, dat1["cuisine"]) * 100, 3)
print("TF-IDF 방식으로 했을 때 Training Accuracy는 {} 입니다.".format(score_tf))

# 나이브 베이즈 모형 적합

from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB

nb_cv = BernoulliNB()
nb_cv.fit(bag, dat1["cuisine"])

nb_tf = GaussianNB()
nb_tf.fit(bag_tf.toarray(), dat1["cuisine"])

score_cv = round(nb_cv.score(bag, dat1["cuisine"]) * 100, 3)
print("Count Vector 방식으로 했을 때 Training Accuracy는 {}입니다.".format(score_cv))

score_tf = round(nb_tf.score(bag_tf, dat1["cuisine"]) * 100, 3)
print("TF-IDF 방식으로 했을 때 Training Accuracy는 {}입니다.".format(score_tf))


# Test 데이터 예측

file_name = 'test.json'

test = pd.read_json(file_directory + "\\" + file_name)

ing_test = test["ingredients"]
ing_new_test = list()

for ing_i in ing:
    a_list = list()
    for i in range(len(ing_i)):
        a_list.append(ing_i[i].replace(" ", ""))
    ing_new.append(" ".join(a_list))

count_test = CountVectorizer(vocabulary = count.vocabulary_)
bag_test = count_test.fit_transform(ing_new_test)

tfidv_test = TfidfVectorizer(vocabulary = tfidv.vocabulary_)
bag_test_tf = tfidv_test.fit_transform(ing_new_test)


# 제출용 CSV 라일 제작
submission_lr_cv = pd.DataFrame()
submission_lr_cv["id"] = test["id"]
submission_lr_cv["cuisine"] = lr_cv.predict(bag_test)
submission_lr_cv.to_csv(file_directory + "\\" + "submission_lr_cv.csv", index = False)

submission_lr_tf = pd.DataFrame()
submission_lr_tf["id"] = test["id"]
submission_lr_tf["cuisine"] = lr_tf.predict(bag_test_tf)
submission_lr_tf.to_csv(file_directory + "\\" + "submission_lr_tf.csv", index = False)

submission_nb_cv = pd.DataFrame()
submission_nb_cv["id"] = test["id"]
submission_nb_cv["cuisine"] = nb_cv.predict(bag_test)
submission_nb_cv.to_csv(file_directory + "\\" + "submission_nb_cv.csv", index = False)
