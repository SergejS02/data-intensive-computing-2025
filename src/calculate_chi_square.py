from mrjob.job import MRJob
from mrjob.step import MRStep
from collections import defaultdict
import csv
import io
import ast
import sys

class ChiSquareUnigrams(MRJob):

    def configure_args(self):
        super(ChiSquareUnigrams, self).configure_args()
        self.add_passthru_arg('--category-col', default='category', help='Column name containing categories')
        self.add_passthru_arg('--tokens-col', default='reviewText_tokens', help='Column name containing tokens')
        self.add_passthru_arg('--csv-header', help='CSV header line to use for all mappers')

    def mapper_init(self):
        try:
            reader = csv.reader(io.StringIO(self.options.csv_header))
            self.header = next(reader)
        except Exception as e:
            self.stderr.write(f"ERROR: Failed to parse CSV header: {e}\n".encode('utf-8'))
            self.header = []

    def mapper_extract_terms(self, _, line_input):
        line_str = line_input.decode('utf-8') if isinstance(line_input, bytes) else line_input
        try:
            reader = csv.reader(io.StringIO(line_str))
            values = next(reader)
            if len(values) != len(self.header):
                self.stderr.write(f"WARNING: Column mismatch: {values}\n".encode('utf-8'))
                return
            row = dict(zip(self.header, values))
            category = row[self.options.category_col]
            try:
                tokens = ast.literal_eval(row[self.options.tokens_col])
                if not isinstance(tokens, list):
                    raise ValueError("Parsed tokens are not a list")
            except (ValueError, SyntaxError) as err:
                self.stderr.write(f"WARNING: Token parse error: {err} in row: {row}\n".encode('utf-8'))
                return
            yield ('!DOC_COUNT', category), 1
            seen = set()
            for raw in tokens:
                if isinstance(raw, str):
                    tok = raw.strip().lower()
                    if tok and tok not in seen:
                        seen.add(tok)
                        yield (category, tok), 1
                        yield ('*', tok), 1
        except Exception as e:
            preview = line_str[:100] + ('...' if len(line_str) > 100 else '')
            self.stderr.write(f"ERROR processing line: {e} - Line: {preview}\n".encode('utf-8'))

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
                mapper_init=self.mapper_init,
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
