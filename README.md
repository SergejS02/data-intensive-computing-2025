# data-intensive-computing-2025

### Preprocessing
For preprocessing, make sure to perform the following steps:
- Tokenization to unigrams, using whitespaces, tabs, digits, and the characters ()[]{}.!?,;:+=-_"'`~#@&*%€$§\/ as delimiters
- Case folding
- Stopword filtering: use the stop word list [on TUWEL] (stopwords.txt) . In addition, filter out all tokens consisting of only one character.

### Calculate chi-square
Write MapReduce jobs that efficiently
- Calculate chi-square values for all unigram terms for each category
- Order the terms according to their value per category and preserve the top 75 terms per category
- Merge the lists over all categories
- Produce a file `output.txt` from the development set that contains the following:
    - One line for each product category (categories in alphabetic order), that contains the top 75 most discriminative terms for the category according to the chi-square test in descending order, in the following format: 
    - <category name> term_1st:chi^2_value term_2nd:chi^2_value ... term_75th:chi^2_value
    - One line containing the merged dictionary (all terms space-separated and ordered alphabetically)

### Report
Produce a report.pdf, that contains detailed report including at least four sections:
1. Introduction
2. Problem Overview
3. Methodology and Approach
4. Conclusions
The Methodology and Approach section should have a figure illustrating your strategy and pipeline in one figure (1 page maximum) that shows the data flow clearly and indicate the chosen <key,value> design (all input, intermediary, and output pairs). The overall report should not exceed more than 8 pages (A4 size).


*preprocessing.ipynb:*
preprocesses input data.
Output: chi_input.csv

calculate_chi_square.py
MapReduce algorithm that takes chi_input.csv as input to 
compute the chi_square values for every token occuring in any review.
Output: chi_square_results.txt
To run the file, write the following in the temrinal:
python mapreduce_chi_square.py chi_input.csv --csv-header "reviewText_tokens,category"

## How to run version in src
1. 
`python preprocessing.py --input "../Assignment_1_Assets/reviews_devset.json"`

output file name and stopwords file are defined by default, but can be changed through args

2. 
when saving only the output.txt
`python runner.py --input chi_input.csv`

or saving the full results (calculate_chi_square.txt)
`python runner.py --input chi_input.csv --save_full_result`

3. when running the preprocessing in hadoop using hadoop_preprocessing.py, following needs to be used as command
`python Exercise_1/hadoop_preprocessing.py   -r hadoop   --hadoop-streaming-jar /usr/lib/hadoop/tools/lib/hadoop-streaming-3.3.6.jar   --stopwords hdfs:///user/e12427512/Exercise_1/stopwords.txt   hdfs:///user/dic25_shared/amazon-reviews/full/reviews_devset.json   --output-dir hdfs:///user/e12427512/hadoop_output`
