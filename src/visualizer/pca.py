import numpy as np
from sklearn.decomposition import PCA

class PCAVisualizer:
    def __init__(self, n_components=2, random_state=42):
        self.n_components = n_components
        self.random_state = random_state
        self.pca = PCA(n_components=self.n_components, random_state=self.random_state)
        self.embedding_ = None

    def fit_transform(self, X):
        """
        对输入的特征X进行PCA降维
        :param X: numpy.ndarray, shape=(n_samples, n_features)
        :return: 降维后的embedding, shape=(n_samples, n_components)
        """
        X = np.array(X)
        self.embedding_ = self.pca.fit_transform(X)
        return self.embedding_

    def transform(self, X):
        """
        对新的特征数据进行PCA映射
        :param X: numpy.ndarray, shape=(n_samples, n_features)
        :return: 映射后的embedding, shape=(n_samples, n_components)
        """
        X = np.array(X)
        return self.pca.transform(X)

    def get_embedding(self):
        """
        获取最近一次降维结果
        """
        return self.embedding_

# ========== 使用示例 ==========
if __name__ == "__main__":
    # 随机生成数据
    X = np.random.rand(100, 5)

    # 创建PCAVisualizer实例
    visualizer = PCAVisualizer(n_components=2)

    # 拟合并降维
    X_embedded = visualizer.fit_transform(X)
    print("降维后的数据shape：", X_embedded.shape)

    # 使用已有模型降维新数据
    X_new = np.random.rand(5, 5)
    X_new_embedded = visualizer.transform(X_new)
    print("新数据降维后shape：", X_new_embedded.shape)
