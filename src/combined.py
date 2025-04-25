import sys
import json
import re
import pandas as pd
import ast
from mrjob.job import MRJob
from mrjob.step import MRStep
import logging
"""
python main.py ../Assignment_1_Assets/reviews_devset.json --stopwords ../Assignment_1_Assets/stopwords.txt
"""

class ChiSquareCalculator(MRJob):

    logging.basicConfig(
        filename="debug_mapper.log",
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    def configure_args(self):
        super().configure_args()
        self.add_file_arg('--stopwords')

    def load_stopwords(self):
        self.stopwords = set()
        if self.options.stopwords:
            with open(self.options.stopwords) as f:
                self.stopwords = set(line.strip() for line in f)

    def tokenize_and_filter(self, text, stopwords):
        logging.info(f"tokenizing text: {text}")
        delimiter_chars = r'()\[\]{}.!?,;:+=\-_"\'`~#@&*%€$§\\/'
        split_pattern = rf'[\s\d{re.escape(delimiter_chars)}]+'
        tokens = re.split(split_pattern, str(text).lower())
        filtered_tokens = [t for t in tokens if t and t not in stopwords and len(t) > 1]
        return filtered_tokens

    def mapper_extract_terms(self, _, line):
        try:
            data = json.loads(line)
            logging.info(f"passing in data: {data}")
            tokens = self.tokenize_and_filter(
                data.get('reviewText', '') + ' ' + data.get('summary', ''),
                self.stopwords
            )
            category = data.get('category', '')
            yield ('!DOC_COUNT', category), 1
            seen = set()
            for raw in tokens:
                if isinstance(raw, str):
                    tok = raw.strip().lower()
                    if tok and tok not in seen:
                        seen.add(tok)
                        yield (category, tok), 1
                        yield ('*', tok), 1
        except (Exception, ValueError, SyntaxError) as e:
            self.stderr.write(f"ERROR processing line: {e}\n".encode('utf-8'))



    def combiner(self, key, counts):
        yield key, sum(counts)

    def reducer_sum_counts(self, key, counts):
        yield key, sum(counts)

    def mapper_organize_for_chi(self, key, count):
        kind = key[0]
        if kind == '!DOC_COUNT':
            _, category = key
            yield ('__DOC_COUNT__', category), count
        elif kind == '*':
            _, term = key
            yield ('__TERM_TOTAL__', term), count
        else:
            category, term = key
            yield (term, category), count
    
    def reducer_group_all_data(self, key, values):
        yield '__GLOBAL__', (key, list(values))

    def reducer_final_init(self):
        self.doc_counts = {}
        self.term_totals = {}
        self.to_compute = []

    def reducer_calculate_chi_final(self, _, key_value_pairs):
        for key, values in key_value_pairs:
            kind = key[0]

            if kind == '__DOC_COUNT__':
                _, category = key
                self.doc_counts[category] = sum(values)

            elif kind == '__TERM_TOTAL__':
                _, term = key
                self.term_totals[term] = sum(values)

            else:
                term, category = key
                A = sum(values)
                self.to_compute.append((term, category, A))

        N = sum(self.doc_counts.values())
        for term, category, A in self.to_compute:
            term_total = self.term_totals.get(term, 0)
            C = self.doc_counts.get(category, 0)
            B = term_total - A
            D = N - C - B - A

            try:
                chi = (N * (A * D - B * C) ** 2) / ((A + B) * (C + D) * (A + C) * (B + D))
            except ZeroDivisionError:
                continue

            expected = term_total * (C / N) if N else 0

            yield (term, category), {
                'observed': A,
                'expected': expected,
                'chi_square': chi,
                'total_term': term_total,
                'total_category': C
            }

    def steps(self):
        return [
            MRStep(
                mapper_init=self.load_stopwords
            ),
            MRStep(
                mapper=self.mapper_extract_terms,
                combiner=self.combiner,
                reducer=self.reducer_sum_counts
            ),
            MRStep(
                mapper=self.mapper_organize_for_chi,
                reducer=self.reducer_group_all_data
            ),
            MRStep(
                reducer_init=self.reducer_final_init,
                reducer=self.reducer_calculate_chi_final
            )
        ]

if __name__ == '__main__':
    results = []
    job = ChiSquareCalculator(args=sys.argv[1:])
    with job.make_runner() as runner:
        try:
            runner.run()
            
            for (token, category), stats in job.parse_output(runner.cat_output()):
                results.append({
                    'token': token,
                    'category': category,
                    'observed': stats['observed'],
                    'expected': stats['expected'],
                    'chi_square': stats['chi_square'],
                    'total_term': stats['total_term'],
                    'total_category': stats['total_category']
                })
            
        except Exception as e:
            print(f"Error during job run: {e}")
    
    df = pd.DataFrame(results)
    if not df.empty:
        print("Processing results...")
        top75 = (df.sort_values(['category', 'chi_square'], ascending=[True, False])
                .groupby('category').head(75))

        with open("output.txt", "w", encoding="utf-8") as f:
            for category in sorted(top75['category'].unique()):
                terms = top75[top75['category'] == category]
                line = f"{category} " + ' '.join(
                    f"{row['token']}:{round(row['chi_square'], 3)}"
                    for _, row in terms.iterrows()
                )
                f.write(line.strip() + "\n")

            merged = sorted(set(top75['token']))
            f.write(" ".join(merged) + "\n")

            print("Top 75 terms per category and merged dictionary written to output.txt")
    else:
        print("No results to process or save.")