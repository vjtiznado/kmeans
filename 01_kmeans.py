# k-means is a type of unsupervised machine learning method
# it can turn into a semi-supervosed method if we start with some initial
# clusters where, for example, half of the data is already labeled as
# member of a certain cluster and then we use these initial cluster to 
# estimate to which cluster each non-labeled data belongs to

import numpy as np
import matplotlib.pyplot as plt
import os

from tqdm import tqdm


def load_data(data_dir):
    # print(os.listdir(data_dir))
    img_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
    imgs = np.array([plt.imread(f) for f in img_files])

    return imgs


def preprocess(data):
    n, h, w, c = data.shape
    flattened = np.reshape(data, (n, h*w*c))
    # print(flattened.shape)

    new_data = flattened

    return new_data, h, w, c


class KMeansClassifier():
    def __init__(self, k=4, height=28, width=28, channel=1):
        self.k = k
        self.mean_images = None
        self.img_shape = (height, width, channel)
        self.avg_distances = list()

    def __str__(self):
        return f"KMeansClassifier Object: K = {self.k}"

    # takes in two numpy arrays and return euclidean distances
    def distance(self, img1, img2):
        deltas = img1 - img2
        squares = deltas ** 2
        sums = np.sum(squares)
        distance = np.sqrt(sums)
        return distance

    def fit(self, data, iters=1):
        # define our initial clusters
        # random init for the first clusters

        # shuffle the data because it is originally
        # sorted by the stream de come from
        np.random.shuffle(data)
        clusters = np.split(data, self.k)
        # make sure your data is divisible by k

        self.mean_images = [np.mean(cluster, axis=0) for cluster in clusters]
        # axis = 0 is for getting the mean across each image

        # iterations
        for i in tqdm(range(iters)):
            # initialize new clusters as empty list (list())
            new_clusters = [list() for c in range(self.k)]

            # iterate over the data
            # and find which cluster each img is
            # closest to, and then assign to that cluster
            sum_min_distances = 0
            for img in tqdm(data):
                distances = [self.distance(img, cluster_mean) for
                             cluster_mean in self.mean_images]

                index = np.argmin(distances)
                sum_min_distances += np.min(distances)

                new_clusters[index].append(img)

            self.avg_distances.append(sum_min_distances/len(data))
            # now recalculate the mean images after each iteration
            self.mean_images = [np.mean(cluster, axis=0)
                                for cluster in new_clusters]

    def display_means(self):
        plt.figure()
        for k, img in enumerate(self.mean_images):
            reshaped = np.reshape(img, self.img_shape)
            plt.imshow(reshaped)
            plt.title(f'Mean Image for Cluster #{k + 1}')
            plt.show()

    def dispaly_avgs(self):
        plt.figure()
        plt.plot(self.avg_distances)
        plt.title('Average distances over iterations')
        plt.xlabel('iteration number')
        plt.ylabel('average distance')
        plt.show()


if __name__ == '__main__':
    try:
        # load_data : directory -> numpy array containing our images
        data = load_data('stream_all')

        # preprocess: n x h x w x c array -> n x (h*w*c) array
        # possibly normalize data
        processed_data, h, w, c = preprocess(data)

        # kmeans: the classifier itself
        kmeans = KMeansClassifier(k=4, height=h, width=w, channel=c)

        # .fit: array of data -> None
        kmeans.fit(processed_data, iters=10)

        # .display_means: displays our mean images
        kmeans.display_means()

        # .display_avgs: plots distance averages along iterations
        kmeans.display_avgs()

        print(kmeans)

    except KeyboardInterrupt:
        print('User aborted.')
