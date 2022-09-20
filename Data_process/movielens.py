# -*-coding:utf-8-*-
import pandas as pd
import matplotlib.pyplot as plt

# display Chinese labels
plt.rcParams["font.sans-serif"] = ["SimHei"]
# Used to display the negative sign
plt.rcParams["axes.unicode_minus"] = False

def getRatings(file_path):
    rates = pd.read_table(
        file_path,
        header=None,
        sep="::",
        names=["userID", "movieID", "rate", "timestamp"],
    )
    print("The range of userIDs is: <{},{}>"
          .format(min(rates["userID"]), max(rates["userID"])))
    print("The range of movieIDs is: <{},{}>"
          .format(min(rates["movieID"]), max(rates["movieID"])))
    print("The range of rating values is: <{},{}>"
          .format(min(rates["rate"]), max(rates["rate"])))
    print("The total number of data is:\n{}".format(rates.count()))
    print("The first 5 records of the data are:\n{}".format(rates.head(5)))
    df = rates["userID"].groupby(rates["userID"])
    print("The minimum number of user rating records is：{}".format(df.count().min()))

    scores = rates["rate"].groupby(rates["rate"]).count()
    # Add numbers to the graph
    for x, y in zip(scores.keys(), scores.values):
        plt.text(x, y + 2, "%.0f" % y, ha="center", va="bottom", fontsize=12)
    plt.bar(scores.keys(), scores.values, fc="r", tick_label=scores.keys())
    plt.xlabel("rating")
    plt.ylabel("number of people")
    plt.title("rating correspond to people statistics")
    plt.show()

def getMovies(file_path):
    movies = pd.read_table(
        file_path,
        header=None,
        sep="::",
        names=["movieID", "title", "genres"]
    )

    print("The range of movieIDs is: <{},{}>"
          .format(min(movies["movieID"]), max(movies["movieID"])))
    print("The total number of data is:\n{}".format(movies.count()))
    moviesDict = dict()
    for line in movies["genres"].values:
        for one in line.split("|"):
            moviesDict.setdefault(one, 0)
            moviesDict[one] += 1

    print("The total number of movie types is:{}".format(len(moviesDict)))
    print("The movie types are:{}".format(moviesDict.keys()))
    print(moviesDict)

    newMD = sorted(moviesDict.items(), key=lambda x: x[1], reverse=True)
    # set label
    labels = [newMD[i][0] for i in range(len(newMD))]
    values = [newMD[i][1] for i in range(len(newMD))]
    explode = [x * 0.01 for x in range(len(newMD))]
    # Set the X-axis Y-axis scale
    plt.axes(aspect=1)
    # labeldistance represents the distance from the label to the center
    # pctdistance represents the distance from the center to the data
    # autopct means percentage format
    plt.pie(
        x=values,
        labels=labels,
        explode=explode,
        autopct="%3.1f %%",
        shadow=False,
        labeldistance=1.1,
        startangle=0,
        pctdistance=0.8,
        center=(-1, 0),
    )
    # Control position: In the bbox_to_anchor array, the former controls left and right movement
    # and the latter controls up and down
    # ncol controls the number of columns listed in the legend, defaults to 1
    plt.legend(loc=7, bbox_to_anchor=(1.3, 1.0), ncol=3, fancybox=True, shadow=True, fontsize=6)
    plt.show()


def getUsers(file_path):
    users = pd.read_table(
        file_path,
        header=None,
        sep="::",
        names=["userID", "gender", "age", "Occupation", "zip-code"],
    )
    print("The range of userIDs is: <{},{}>".format(min(users["userID"]), max(users["userID"])))
    print("The total number of data is:\n{}".format(users.count()))

    usersGender = users["gender"].groupby(users["gender"]).count()
    print(usersGender)

    plt.axes(aspect=1)
    plt.pie(x=usersGender.values, labels=usersGender.keys(), autopct="%3.1f %%")
    plt.legend(bbox_to_anchor=(1.0, 1.0))
    plt.show()

    usersAge = users["age"].groupby(users["age"]).count()
    print(usersAge)

    plt.plot(
        usersAge.keys(),
        usersAge.values,
        label="User age information display",
        linewidth=3,
        color="r",
        marker="o",
        markerfacecolor="blue",
        markersize=12,
    )
    for x, y in zip(usersAge.keys(), usersAge.values):
        plt.text(x, y+10, "%.0f" % y, ha="center", va="bottom", fontsize=12)
    plt.xlabel("User age")
    plt.ylabel("The number of people corresponding to the age group")
    plt.title("User age group statistics")
    plt.show()


if __name__ == "__main__":
    getRatings("../data/ml-1m/ratings.dat")
    getMovies("../data/ml-1m/movies.dat")
    getUsers("../data/ml-1m/users.dat")
