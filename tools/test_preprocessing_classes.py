from sklearn.preprocessing import *
from sklearn4x.sklearn4x import save_scikit_learn_model

# targets = ['a', 'a', 'c', 'd', 'a', 'd', 'b', 'b', 'c']
targets = [1,2,3,4,1,2,3]
encoder = LabelBinarizer()

encoder.fit(targets)
print(encoder.transform(targets))

print(encoder.classes_)

save_scikit_learn_model({'encoder': encoder}, 'test.skx')
