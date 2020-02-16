import numpy
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity


class DataSource(object):
    pass


class DummyData(DataSource):
    def wine(self, wine_id):
        return numpy.random.rand(1, 20)

    def all(self):
        return numpy.random.rand(30, 20)

    def liked_by(self, user_id, n):
        return numpy.random.randint(2, size=n)


class Wine(object):
    def __init__(self, data_source, id):
        self.data_source = data_source
        self.vector = data_source.wine(id)

    def similar_wines(self, n=10):
        wines = self.data_source.all()
        similarities = pd.DataFrame(cosine_similarity(self.vector, wines)) \
            .melt(var_name='wine') \
            .sort_values('value', ascending=False) \
            .head(n)
        return similarities


class User(object):
    def __init__(self, data_source, id):
        self.data_source = data_source
        self.id = id

    def liked_wines(self, n=10):
        return self.data_source.liked_by(self.id, n)


if __name__ == '__main__':
    wine = Wine(DummyData(), 0)
    print(wine.similar_wines())
