import numpy as np
from sklearn import preprocessing

data = np.array([[3,-1.5,2,-5.4],[0,4,-0.3,2.1],[1,3.3,-1.9,-4.3]])
print(data)

# mean removal
data_standardized = preprocessing.scale(data)
print("\nMean = ", data_standardized.mean(axis=0))
print("Std dev =", data_standardized.std(axis=0))

# scaling
data_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
data_scaled = data_scaler.fit_transform(data)
print("\nMin max scaled data =", data_scaled)

# normalization
data_normalized = preprocessing.normalize(data, norm='l1')
print("\nL1 normalized data =", data_normalized)

# binarization
data_binarized = preprocessing.Binarizer(threshold=1.4)\
    .transform(data)
print("\nBinarized data", data_binarized)

# one hot encoding
encoder = preprocessing.OneHotEncoder()
data1 = [[0, 2, 1, 12], [1, 3, 5, 3], [2, 3, 2, 12], [1, 2, 4,
3]]
encoder.fit(data1)
encoded_vector = encoder.transform([[2,3,5,3]]).toarray()
print(data1)
print("\nEncoded vector =", encoded_vector)

# label encoding/decoding
label_encoder = preprocessing.LabelEncoder()
input_classes = ['Audi', 'Ford', 'Audi', 'Toyota', 'Ford', 'BMW']
label_encoder.fit(input_classes)
print("\nClass mapping:")
for i,item in enumerate(label_encoder.classes_):
    print(item, '-->', i)
labels=['Toyota','Ford','Audi']
encoded_labels = label_encoder.transform(labels)
print("\nLabels = ", labels)
print("Encoded labels = ", list(encoded_labels))
encoded_labels = [2,1,0,3,1]
decoded_labels = label_encoder.inverse_transform(encoded_labels)
print("\nEncoded labels = ", encoded_labels)
print("Decoded labels = ", list(decoded_labels))