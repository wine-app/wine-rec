import pandas as pd

import numpy


class RandomProjectionHasher(object):
    def __init__(self, hash_size, num_features):
        self.hash_size = hash_size
        self.random_projections = numpy.random.randn(hash_size, num_features)

    def vector_hash(self, wine_vectors):
        """
        :param wine_vectors: A numpy array or pandas DataFrame of shape (n_wines, n_features)
        :return: A list of hashes representing corresponding to the wines
        """
        bools = (numpy.dot(wine_vectors, self.random_projections.T) > 0).astype('int')
        return [''.join(bits) for bits in bools.astype('str')]


def wine_hashes(wine_vectors, n=1, hash_size=5):
    """
    Get multiple hash values for each wine vector.
    :param wine_vectors: A pandas DataFrame of size (n_wines, n_features)
    :param n: The number of hash tables to produce
    :param hash_size: The size of each hash in bits
    :return: A pandas DataFrame containing one column for each hash created
    """
    hashes = [RandomProjectionHasher(hash_size, wine_vectors.shape[1]).vector_hash(wine_vectors) for _ in range(n)]
    return pd.DataFrame({'hash{}'.format(str(i)): hashes[i] for i in range(len(hashes))})


if __name__ == '__main__':
    print(wine_hashes(numpy.random.randn(10, 20), 5))
