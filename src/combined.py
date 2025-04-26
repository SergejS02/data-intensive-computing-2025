import sys
import json
import re
import ast
from mrjob.job import MRJob
from mrjob.step import MRStep
import logging

class ChiSquareCalculator(MRJob):
    def configure_args(self):
        super().configure_args()
        self.add_file_arg('--stopwords')

    def load_stopwords(self):
        self.stopwords = set()
        if self.options.stopwords:
            with open(self.options.stopwords, 'r') as f:
                self.stopwords = set(line.strip() for line in f)

    def tokenize_and_filter(self, text, stopwords):
        delimiter_chars = r'()\[\]{}.!?,;:+=\-_"\'`~#@&*%€$§\\/'
        split_pattern = rf'[\s\d{re.escape(delimiter_chars)}]+'
        tokens = re.split(split_pattern, str(text).lower())
        filtered_tokens = [t for t in tokens if t and t not in stopwords and len(t) > 1]
        return filtered_tokens

    def mapper_extract_terms(self, _, line):
        try:
            data = json.loads(line)
            category = data.get('category', '')

            yield ('!DOC_COUNT', category), 1

            try:
                tokens = self.tokenize_and_filter(
                    data.get('reviewText', '') + ' ' + data.get('summary', ''),
                    self.stopwords
                )
            except Exception as e:
                logging.error(f"Error inside tokenize_and_filter: {e}")
                tokens = []

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

        # Organize chi-square scores per category
        category_terms = {}

        for term, category, A in self.to_compute:
            term_total = self.term_totals.get(term, 0)
            C = self.doc_counts.get(category, 0)
            B = term_total - A
            D = N - C - B - A

            try:
                chi_square = (N * (A * D - B * C) ** 2) / ((A + B) * (C + D) * (A + C) * (B + D))
            except ZeroDivisionError:
                continue

            if category not in category_terms:
                category_terms[category] = []
            category_terms[category].append((term, chi_square))

        # Now output per category: Top 75 terms sorted by chi-square
        for category in sorted(category_terms.keys()):
            sorted_terms = sorted(category_terms[category], key=lambda x: -x[1])[:75]
            line = category + " " + ' '.join(
                f"{term}:{round(chi_square, 3)}" for term, chi_square in sorted_terms
            )
            yield None, line

        # Merged dictionary line (all tokens from all categories)
        merged_tokens = set()
        for term_chi_list in category_terms.values():
            for term, _ in term_chi_list:
                merged_tokens.add(term)

        merged_tokens_sorted = sorted(merged_tokens)
        merged_line = ' '.join(merged_tokens_sorted)
        yield None, merged_line

    def steps(self):
        return [
            MRStep(
                mapper_init=self.load_stopwords,
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
    ChiSquareCalculator.run()
