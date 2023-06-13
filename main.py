import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from fuzzywuzzy import fuzz
from collections import Counter
pd.set_option('display.max_columns', None)





# Load data
def load_data(products_dir, reviews_dir):

    # Finding the best matching value in Top2_Categories
    def categorize(value):
        scores = [fuzz.ratio(value, category) for category in top_categories]
        max_score = max(scores)
        max_score_index = scores.index(max_score)

        if max_score > 80:
            return top_categories[max_score_index]
        else:
            return 'Other'

    #  Load products data
    all_products_tsv_list = os.listdir(products_dir)
    for single_tsv in all_products_tsv_list:
        single_data = pd.read_csv(os.path.join(products_dir, single_tsv), sep='\t', names=['id', 'category', 'product_title'])
        if single_tsv == all_products_tsv_list[0]:
            products_data = single_data
        else:
            products_data = pd.concat([products_data, single_data], ignore_index=True)

    # Resolving spelling errors in the dataset and check if there are only two types
    top_categories = Counter(products_data['category']).most_common(2)
    top_categories = [category for category, count in top_categories]
    products_data['category'] = products_data['category'].apply(categorize)
    print(products_data['category'].value_counts())

    #  Load reviews data
    all_reviews_tsv_list = os.listdir(reviews_dir)
    for single_tsv in all_reviews_tsv_list:
        single_data = pd.read_csv(os.path.join(reviews_dir, single_tsv), sep='\t', names=['id', 'rating', 'review_text'])

        # Handling data order exceptions in 'reviews-2.tsv'
        if single_tsv == 'reviews-2.tsv':
            single_data[['rating', 'id']] = single_data[['id', 'rating']]

        if single_tsv == all_reviews_tsv_list[0]:
            reviews_data = single_data
        else:
            reviews_data = pd.concat([reviews_data, single_data], ignore_index=True)

    # Considering multiple reviews of a product
    reviews_data['review_text'] = reviews_data.groupby('id')['review_text'].transform(lambda x: ' '.join(x))
    reviews_data['rating'] = reviews_data.groupby('id')['rating'].transform('mean')

    return products_data, reviews_data

if __name__ == "__main__":
    products_dir = 'dataset/products'
    reviews_dir = 'dataset/reviews'
    products_data, reviews_data = load_data(products_dir, reviews_dir)

    # Merge the data on id
    data = pd.merge(products_data, reviews_data, on='id')

    # Split the data into training and test sets
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    # Define preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('title_tfidf', TfidfVectorizer(), 'product_title'),
            ('review_tfidf', TfidfVectorizer(), 'review_text'),
            ('rating_scaler', StandardScaler(), ['rating'])
        ])

    # Define the pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression())
    ])

    # Fit the pipeline
    pipeline.fit(train_data, train_data['category'])

    # Evaluate the pipeline
    train_accuracy = pipeline.score(train_data, train_data['category'])
    test_accuracy = pipeline.score(test_data, test_data['category'])

    print(f'Train Accuracy: {train_accuracy}')
    print(f'Test Accuracy: {test_accuracy}')