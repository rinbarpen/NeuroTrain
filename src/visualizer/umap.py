import umap
import numpy as np

class UMAPVisualizer:
    def __init__(self, n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean', random_state=42):
        self.reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=n_components,
            metric=metric,
            random_state=random_state
        )
        self.embedding_ = None

    def fit_transform(self, X):
        """
        对输入的特征X进行umap降维
        :param X: numpy.ndarray, shape=(n_samples, n_features)
        :return: 降维后的embedding, shape=(n_samples, n_components)
        """
        X = np.array(X)
        self.embedding_ = self.reducer.fit_transform(X)
        return self.embedding_

    def transform(self, X):
        """
        对新的特征数据进行umap映射
        :param X: numpy.ndarray, shape=(n_samples, n_features)
        :return: 映射后的embedding, shape=(n_samples, n_components)
        """
        X = np.array(X)
        return self.reducer.transform(X)

    def get_embedding(self):
        """
        获取最近一次降维结果
        """
        return self.embedding_


if __name__ == "__main__":
    # 示例：使用UMAPVisualizer对数据降维
    np.random.seed(42)
    X = np.random.rand(100, 10)  # 生成100个样本，每个样本10维
    visualizer = UMAPVisualizer(n_neighbors=10, min_dist=0.05, n_components=2)
    embedding = visualizer.fit_transform(X)
    print("UMAP降维结果 shape:", embedding.shape)
    # 对新数据进行umap映射
    X_new = np.random.rand(5, 10)
    embedding_new = visualizer.transform(X_new)
    print("新数据映射结果 shape:", embedding_new.shape)
