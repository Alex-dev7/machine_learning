import numpy as np

def euclidean_distance(x1, x2):
    distance = np.sqrt(np.sum((x1-x2) **2))
    return distance
    

class KNN:
    def _init_ (self, k):
        self.k = k
        
    def fit(self, X, y):
        self.X_train = X
        self.Y_train = y
        
        
    
    def predict(self, X):
        predictions = [self.helper_predict(x) for x in X]
        return predictions
        
    def helper_predict(self, x):
        
        # euclidean distance
        distances = distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        
        # indices of k nearest neighbors
        sorted_distances =  np.argsort(distances)
        closest_k = sorted_distances[:self.k]
        
        # labels of k nearest neighbors
        k_nearest_labels = [self.y_train[i] for i in closest_k]
        
        # majority vote for classification
        if isinstance(self.y_train[0], (int, np.integer)):
            return np.bincount(k_nearest_labels).argmax()
        else:
            return np.mean(k_nearest_labels)