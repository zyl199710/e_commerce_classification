# A solution for task 2. Programming: fictitious e-commerce classification case

## Implement

Python (Pandas, Sklearn, Fuzzywuzzy)

## Dataset

The [dataset](https://surfdrive.surf.nl/files/index.php/s/LDwpIdG7HHkQiOs) contains two kinds of input files with tab-separated data.

## Overview

- To address the problem of category data anomalies in the Projects dataset [*(Ktchen)*](./dataset/products/products-data-3.tsv). I designed the program to automate the identification and modification of anomalous data [*(line 40)*](./main.py). Specifically, I first identify the multiple values in the Category column, then match the outliers with their similarity, and finally convert the outliers to the most similar data.

- To solve the problem of incorrect column order in the Reviews dataset [*(review-2.tsv)*](./dataset/reviews/reviews-2.tsv), we choose to directly replace the order of two columns when inputting data [*(line 51)*](./main.py). The reason I did this is that the problem is easier to detect in real scenarios, so it saves computational costs to replace them directly and manually.

- Considering that multiple reviews may exist for a product *(although this dataset does not have this)*, I averaged all the ratings and concatenated all the review text for the same product [*(line 60)*](./main.py).