import numpy as np
from sklearn.manifold import TSNE

class TSNEVisualizer:
    def __init__(self, n_components=2, perplexity=30.0, learning_rate=200.0, n_iter=1000, random_state=42):
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.random_state = random_state
        self.tsne = TSNE(
            n_components=self.n_components,
            perplexity=self.perplexity,
            learning_rate=self.learning_rate,
            # sklearn 1.2.0及以前为n_iter参数，之后版本请用max_iter替换
            n_iter=self.n_iter,
            random_state=self.random_state
        )
        self.embedding_ = None

    def fit_transform(self, X):
        """
        对输入的特征X进行t-SNE降维
        :param X: numpy.ndarray, shape=(n_samples, n_features)
        :return: 降维后的embedding, shape=(n_samples, n_components)
        """
        X = np.array(X)
        self.embedding_ = self.tsne.fit_transform(X)
        return self.embedding_

    def get_embedding(self):
        """
        获取最近一次降维结果
        """
        return self.embedding_

# ========== 使用示例 ==========
if __name__ == "__main__":
    # 随机生成数据
    X = np.random.rand(100, 5)  # 100个样本，5维特征

    # 创建TSNEVisualizer实例
    visualizer = TSNEVisualizer(n_components=2, perplexity=40.0)

    # 拟合并降维
    X_embedded = visualizer.fit_transform(X)
    print("降维后的数据shape：", X_embedded.shape)

    # 获取最近一次降维结果
    print("最近一次降维结果：", visualizer.get_embedding())
