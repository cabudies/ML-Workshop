import pandas as pd

# specifying the location of iris data flower csv file
file_location = 'iris.csv'

# loading the data set into 'iris_data' folder using panda library
iris_data = pd.read_csv(file_location, sep=',')

# defining columns for our data
columns_name = ["Sepal_Length", "Sepal_Width", "Petal_Length", "Petal_Width", "Category"]
iris_data.columns = columns_name

# pd.set_option('display.max_columns', 5)
#
# print(iris_data.head())

# performing deep copy to create a new instance of the list
new_iris_data = iris_data.__deepcopy__()

# dropping previously 'Category' column
new_iris_data = new_iris_data.drop('Category', axis=1)
# print(new_iris_data.head())

# # Handling category data
# from sklearn.preprocessing import LabelBinarizer
# label_encoder = LabelBinarizer()
#
# # putting values used by Label encoder into a separate data frame.
# output_iris_data = iris_data['Category'].__deepcopy__()
# #
# # # train the encoder for the data.
# label_encoder.fit(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
# # print(label_encoder.classes_)
#
# encoded_data = label_encoder.transform(output_iris_data)
# print(encoded_data)

# mapping new values in place of categorical values
output_iris_data = iris_data['Category'].__deepcopy__()
# print(output_iris_data.head())

# performing map function to map values accordingly.
output_iris_data = output_iris_data.map({
    'Iris-setosa' : 0,
    'Iris-versicolor' : 1,
    'Iris-virginica' : 2
})
# print(output_iris_data.head())

# using KNN classifier
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=3)

# training the model with 'data and labels'
model.fit(new_iris_data, output_iris_data)
# #
# # # Predicting category of new flower
testSet = [[7.2, 3.6, 5.1, 2.5]]
test = pd.DataFrame(testSet)
#
prediction = model.predict(test)

# print out the prediction.
print(prediction)

# # Decoding numerical values back to categorical values.
# inverted_single_value = label_encoder.inverse_transform(prediction)
# print(inverted_single_value)
