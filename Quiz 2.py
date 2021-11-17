import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()

ratings = pd.read_csv("gameratings.csv")
test = pd.read_csv("test_esrb.csv")
names = pd.read_csv("target_names.csv")

data_train = ratings.loc[:,"console":"violence"].values
data_test = test.loc[:,"console":"violence"].values
target_train = ratings.Target.values
target_test = test.Target.values

# train the model
knn.fit(X = data_train, y = target_train)

# test the model
predicted = knn.predict(X = data_test)
expected = target_test

# display wrong predicted and expected pairs
wrong = [(p,e) for (p,e) in zip(predicted, expected) if p != e]
print(wrong)

# produce a csv file of the name and predicted rating
game = test.loc[:,"title"].values
pred_ratings = [names.loc[x-1,"target_name"] for x in predicted]

name_rating = [(x,y) for (x,y) in zip(game,pred_ratings)]

outfile = open("game_and_rating.csv", 'w')
outfile.write('title' + ',' + 'prediction' + '\n')
for n in name_rating:
    outfile.write(str(n[0]) + ',' + (str(n[1]) + '\n'))


