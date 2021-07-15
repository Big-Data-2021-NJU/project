import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix
from scipy import sparse
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from tqdm import tqdm

df = pd.read_table('part-r-00000', header=None)
labels = []
for value in df[0]:
    labels.append(value[:2])
df['y'] = labels

y = df['y']
from sklearn import preprocessing

le = preprocessing.LabelEncoder()
y = le.fit(y).transform(y)
X = []
indices = []
values = []
indices_y = []
i = 0
for x in tqdm(df[1]):
    for pair in x.split(' '):
        indices.append(eval(pair.split(':')[0]))
        values.append(eval(pair.split(':')[1]))
        indices_y.append(i)
    # csr_matrix([indices, values])
    i += 1

X = sparse.coo_matrix((np.array(values), (np.array(indices_y, dtype=int), np.array(indices, dtype=int))),
                      shape=(len(df['y']), 85368))

print('start fitting')
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.svm import LinearSVC

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=5)
from sklearn.linear_model import LogisticRegression

# clf = LogisticRegression(random_state=0, verbose=1).fit(train_X, train_y)
# clf = SVC(kernel='linear', verbose=1).fit(train_X, train_y)
#clf = LinearSVC(verbose=1).fit(train_X, train_y)
# clf = OneVsRestClassifier(SVC(kernel='linear')).fit(train_X, train_y)
clf = LinearSVC(verbose=1).fit(train_X, train_y)
print(clf.score(test_X, test_y))

