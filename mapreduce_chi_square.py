# Run using      python mapreduce_chi_square.py chi_input.csv     in the console
from mrjob.job import MRJob
from mrjob.step import MRStep
from collections import defaultdict
import csv
import io
import ast


class ChiSquareUnigrams(MRJob):

    def configure_args(self):
        super(ChiSquareUnigrams, self).configure_args()
        self.add_passthru_arg('--category-col', default='category', help='Column name containing categories')
        self.add_passthru_arg('--tokens-col', default='reviewText_tokens', help='Column name containing tokens')
        self.add_passthru_arg('--csv-header', help='CSV header line to use for all mappers')

    # first phase term frequency counting
    def mapper_init(self):
        import csv
        import io
        try:
            reader = csv.reader(io.StringIO(self.options.csv_header))
            self.header = next(reader)
        except Exception as e:
            self.stderr.write(f"ERROR: Failed to parse CSV header: {e}\n".encode('utf-8'))
            self.header = []

    def mapper_extract_terms(self, _, line_input):

        if isinstance(line_input, bytes):
            line_str = line_input.decode('utf-8')
        else:
            line_str = line_input

        if not hasattr(self, 'header'):
            reader = csv.reader(io.StringIO(line_str))
            candidate_header = next(reader)

            # treat as header if it looks like one
            if self.options.category_col in candidate_header and self.options.tokens_col in candidate_header:
                self.header = candidate_header
                self.stderr.write(b"Detected header line.\n")
                # skip the current header
                return
            else:
                # otherwise data
                self.header = candidate_header
                self.stderr.write(b"No header detected, treating first line as data.\n")

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

            for token in tokens:
                if isinstance(token, str) and token:
                    token = token.lower()
                    yield (category, token), 1
                    yield ('*', token), 1
                    yield ('DEBUG_CATEGORY', category), 1

        except Exception as e:
            line_preview = line_str[:100] + ('...' if len(line_str) > 100 else '')
            error_msg = f"ERROR processing line: {e} - Line: {line_preview}\n"
            self.stderr.write(error_msg.encode('utf-8'))

    def reducer_sum_counts(self, key, counts):
        """Summing term frequencies per category"""
        yield key, sum(counts)

    # phase 2: Chi-quared calc
    def mapper_organize_for_chi(self, key, count):
        """Reorganizing data for chi-squared calculation"""
        category, token = key
        yield token, (category, count)

    def reducer_calculate_chi(self, token, category_counts):
        counts = list(category_counts)

        term_total = 0
        category_term_counts = {}
        category_totals = defaultdict(int)
        # amount total reviews across all categories
        N = 0
        # amount total reviews per category
        C_total = defaultdict(int)

        # instant pass to extract critical totals
        for category, count in counts:
            if category == '*':
                # amount total reviews containing the token across all categories
                term_total = count
            else:
                category_term_counts[category] = count
                # total tokens in category (bullshit, but we have no review counts)
                category_totals[category] += count

        # extract total reviews (N) and per-category reviews (C_total)
        # TODO: This part is bullshit without document counts but uses term counts as zwischenloesung
        # Total tokens (incorrect proxy for reviews)
        N = sum(category_totals.values())
        for category in category_totals:
            # Tokens in category (incorrect proxy)
            C_total[category] = category_totals[category]

        # calculate chi-square for each category
        for category in category_term_counts:
            A = category_term_counts[category]  # Term count in category (incorrect proxy for reviews)
            C = C_total[category]  # Total tokens in category (incorrect proxy)
            B = term_total - A  # Term count in other categories
            D = N - C - B  # Approximate reviews not in category and without token

            # avoiding division by zero
            if (A + B) * (C + D) * (A + C) * (B + D) == 0:
                continue

            # chi-square formula for reviews from slide 40 (using term counts as proxies)
            chi_square = (N * (A * D - B * C) ** 2) / ((A + B) * (C + D) * (A + C) * (B + D))

            yield (token, category), {
                'observed': A,
                'expected': (term_total * C) / N if N != 0 else 0,
                'chi_square': chi_square,
                'total_term': term_total,
                'total_category': C
            }

    def steps(self):
        return [
            MRStep(mapper_init=self.mapper_init,
                   mapper=self.mapper_extract_terms,
                   reducer=self.reducer_sum_counts),
            MRStep(mapper=self.mapper_organize_for_chi,
                   reducer=self.reducer_calculate_chi)
        ]


if __name__ == '__main__':
    import pandas as pd

    with open('chi_input.csv', 'r',
              encoding='utf-8') as f:
        header_line = f.readline().strip()

# running job on machine
    args = [
        # csv input file with our reeviews
        'chi_input.csv',
        '--category-col', 'category',
        '--tokens-col', 'reviewText_tokens',
        '--csv-header', header_line # need it because otherwise mrjob is not recognizing the header in our HUGE csv file
    ]

    job = ChiSquareUnigrams(args=args)
    with job.make_runner() as runner:
        runner.run()

        # further processing results
        results = []
        # try-except to handle potential StopIteration if output empty
        try:
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
        except StopIteration:
            print("Warning: No output generated by the MapReduce job.")
            results = []

       # analy dataframe
        if results:
            chi_results = pd.DataFrame(results)
            output_txt_path = 'chi_square_results.txt'

            # saving whole output totab sep txt file
            chi_results.to_csv(
                output_txt_path,
                sep='\t',
                index=False,
                encoding='utf-8'
            )
            print(f"Full results saved to: {output_txt_path}")
            # printing some fancy stats
            top_terms = (chi_results.sort_values(['category', 'chi_square'],
                                                 ascending=[True, False])
                         .groupby('category').head(10))

            print("\nTop significant terms per category (display only):")
            print(top_terms)
        else:
            print("No results to process or save.")