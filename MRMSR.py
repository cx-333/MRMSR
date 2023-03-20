# coding: utf-8

import numpy as np

#=========================================================================================================#
# Paper Name: Feature Seection With Maximal Relevance and Minimal Supervised Redundancy.
# Paper Author: Yadi Wang
# Code Author: Xin Chen
# Code Description: Feature Selection Algorithm && Dimension Reduction Algorithm for Classification Task.
#=========================================================================================================#
class MRMSR:
    def __init__(self, k: int) -> None:
        self.k = k
        self.F_hat = None

    def feature_selection(self, features, classes):
        """
        params:
            features: Feature set F = {f1, f2, ..., fp}, where fi stand for a column vector.
            classes: class labels set C = {y1, y2, ..., yN}, where yi is a scalar.
        return:
            A list for the selected columns of feature set or k-D array.
        """
        # step 1: initation
        F_hat = []
        m = 0
        fea_num = len(classes)
        # step 2: Calculate H(C) = - sum(p(yi)*log(p(yi))), i \in C
        H_C = H_entropy(classes)
        # step 3: Calculate R(Fi, C) = I(Fi; C) / H(C)
        p = features.shape[1]
        retain_columns = [i for i in range(p)]
        R = np.zeros(p)     # relevance
        for i in range(p):
            R[i] = C_Information_entropy(features[:, i].reshape(fea_num, 1), classes) / H_C
        # step 4: select F_hat
        while (m < self.k):
            if m == 0:
                column = np.argmax(R)   # l
                F_hat.append(column)    # first feature
                # first_F = features[:, column].reshape(fea_num, 1)
                m += 1
                retain_columns.remove(column)
            J_F = np.zeros(p)   # -1 <= J(Fi) <= 1
            Ti = 0
            for i in retain_columns:
                # Calculate I(Fl; C|Fi) and I(Fi; C|Fl)
                I1 = C_Information_entropy(features[:, [column, i]], classes)
                I2 = C_Information_entropy(features[:, [i, column]], classes)
                S_li = 1 - ((I1 + I2) / (2 * H_C))
                Ti += S_li
                J_F[i] = R[i] - ((1 / m) * Ti)
            J_F[F_hat] = -1
            column = np.argmax(J_F)
            retain_columns.remove(column)
            F_hat.append(column)    
            m += 1
        self.F_hat = F_hat
        return features[:, F_hat]
    
    def feature_columns(self):
        return self.F_hat


def H_entropy(x: list or np.ndarray) -> float:
    # Example: 
    #   data = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0])
    #   H_entropy(data) = 0.9402859586706311
    if type(x) == list:
        x = np.array(x) 
    C = list(set(x))
    count_xi = np.zeros(len(C), dtype=int)
    i = 0
    for ci in C:
        count_xi[i] = np.sum(x == ci)
        i += 1
    prob_xi = count_xi / len(x)
    result = - np.sum( prob_xi * np.log2(prob_xi + 1e-7) )   # guarantee stability.

    return result

def C_Information_entropy(A: np.ndarray, C: list or np.ndarray) -> float:
    # Conditional mutual information entropy
    # A: N x 2 dimensional array,  A[:, 0] -- Fi, A[:, 1] -- Fj
    # Example:
    #   A = np.array([[0, 0], [0, 0], [1, 0], [2, 1], [2, 2], [2, 2], [1, 2],
    #                 [0, 1], [0, 2], [2, 1], [0, 1], [1, 1], [1, 0], [2, 1]])
    #   A[:, [0, 1]] = A[:, [1, 0]] 
    #   C = [0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0]
    #   C_Information_entropy(A, C) = H(C|Fj) - H(C|Fi, Fj) = 0.21104414354105505
    if type(C) == list:
        C = np.array(C)
    if A.shape[1] == 2:
        # I(Fi; C|Fj) = H(C|Fj) - H(C|Fi, Fj)
        temp = A[:, 1].reshape(len(C), 1)
        info = C_H_entropy(temp, C) - C_H_entropy(A, C)
        return info
    else:
        # I(Fi; C) = H(C) - H(C|Fi)
        info = H_entropy(C) - C_H_entropy(A, C)
        return info


def C_H_entropy(A: np.ndarray, C: list or np.ndarray) -> float:
    # Conditional entropy
    # Example 1:
    #   A = np.array([[0], [0], [1], [2], [2], [2], [1], [0], [0], [2], [0], [1], [1], [2]])
    #   C = [0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0]
    #   C_H_entropy(A, C) = 0.6935714...
    # Example 2:
    #   A = np.array([[0, 0], [0, 0], [1, 0], [2, 1], [2, 2], [2, 2], [1, 2],
    #                 [0, 1], [0, 2], [2, 1], [0, 1], [1, 1], [1, 0], [2, 1]])
    #   C = [0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0]
    #   C_H_entropy(A, C) = 0.482491964...
    if type(C) == list:
        C = np.array(C)
    D = A.shape[0]
    if A.shape[1] == 2:
        A1 = list(set(A[:, 0]))
        A2 = list(set(A[:, 1]))
        prob_Ai = np.zeros(len(A1) * len(A2))
        i = 0
        for c1 in A1:
            for c2 in A2:
                prob_Ai[i] = (np.sum((A[:, 0] == c1) * (A[:, 1] == c2)) / D) * \
                             H_entropy(C[(A[:, 0] == c1) * (A[:, 1] == c2)])
                i += 1
        return np.sum(prob_Ai)
    else:
        A3 = list(set(A[:, 0]))
        prob_Ai = np.zeros(len(A))
        i = 0
        for Ai in A3:
            prob_Ai[i] = (np.sum(A[:, 0] == Ai) / D) * H_entropy(C[A[:, 0] == Ai])
            i += 1  
        return np.sum(prob_Ai)





if __name__ == '__main__':
    print("Testing .... ")
    from sklearn.linear_model import BayesianRidge
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_iris, load_wine
    from sklearn.decomposition import  PCA 

    data = load_iris(return_X_y=True)
    X_train, X_test, Y_train, Y_test = train_test_split(data[0], data[1], test_size=0.1, stratify=data[1])

    reg1 = BayesianRidge()
    reg2 = BayesianRidge()
    pca = PCA(n_components=2)
    mrmsr = MRMSR(k=2)

    x_pca_train = pca.fit_transform(X_train)
    x_mrmsr_train = mrmsr.feature_selection(X_train, Y_train)
    feature_columns = mrmsr.feature_columns()
    reg1.fit(x_pca_train, Y_train)
    print("PCA result: {0}".format(reg1.score(pca.fit_transform(X_test), Y_test)))

    reg2.fit(x_mrmsr_train, Y_train)
    print("MRMSR result: {0}".format(reg2.score(X_test[:, feature_columns], Y_test)))
