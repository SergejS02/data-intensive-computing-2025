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

    def reducer_calculate_chi(self, joined_key, values):
        if not hasattr(self, 'doc_counts'):
            self.doc_counts = {}
            self.term_totals = {}
            self.to_compute = []
        if joined_key[0] == '__DOC_COUNT__':
            _, category = joined_key
            self.doc_counts[category] = sum(values)
            return
        if joined_key[0] == '__TERM_TOTAL__':
            _, term = joined_key
            self.term_totals[term] = sum(values)
            return
        term, category = joined_key
        A = sum(values)
        self.to_compute.append((term, category, A))

    def reducer_final(self):
        N = sum(self.doc_counts.values())
        for term, category, A in self.to_compute:
            term_total = self.term_totals.get(term, 0)
            C = self.doc_counts.get(category, 0)
            B = term_total - A
            D = N - C - B
            if (A + B) * (C + D) * (A + C) * (B + D) == 0:
                continue
            chi = (N * (A * D - B * C) ** 2) / ((A + B) * (C + D) * (A + C) * (B + D))
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
                reducer=self.reducer_calculate_chi,
                reducer_final=self.reducer_final
            )
        ]

if __name__ == '__main__':
    import sys
    import pandas as pd

    csv_path = sys.argv[1]
    with open(csv_path, 'r', encoding='utf-8') as f:
        header_line = f.readline().strip()

    args = sys.argv[1:]
    if '--csv-header' in args:
        idx = args.index('--csv-header')
        if idx + 1 < len(args):
            args[idx + 1] = header_line
    else:
        args += ['--csv-header', header_line]

    job = ChiSquareUnigrams(args=args)
    with job.make_runner() as runner:
        runner.run()
        results = []
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

        if results:
            chi_results = pd.DataFrame(results)
            chi_results.to_csv('chi_square_results.txt', sep='\t', index=False, encoding='utf-8')
            print(f"Full results saved to: chi_square_results.txt")

            top75 = (chi_results.sort_values(['category', 'chi_square'], ascending=[True, False])
                     .groupby('category').head(75))

            all_categories = sorted(set(row['category'] for row in results))
            with open('output.txt', 'w', encoding='utf-8') as f:
                for cat in all_categories:
                    group = top75[top75['category'] == cat]
                    line = f"{cat} " + ' '.join(f"{row['token']}:{round(row['chi_square'], 3)}" for _, row in group.iterrows())
                    f.write(line.strip() + '\n')

                merged_terms = sorted(set(top75['token']))
                f.write(' '.join(merged_terms) + '\n')

            print("Top 75 terms per category and merged dictionary written to output.txt")
        else:
            print("No results to process or save.")