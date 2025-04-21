# data-intensive-computing-2025
internal deadline: 21st

# to do

### Preprocessing
** Sergej by Sunday 13th**
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