from mrjob.job import MRJob
from mrjob.step import MRStep
import csv
import io
import ast
import sys
import pandas as pd
import logging


class ChiSquareUnigrams(MRJob):
    # Configure logging to write to a file
    logging.basicConfig(
        filename="debug_mapper.log",
        filemode="a",  # Append mode
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )


    def configure_args(self):
        super(ChiSquareUnigrams, self).configure_args()
        self.add_passthru_arg('--category-col', default='category', help='Column name containing categories')
        self.add_passthru_arg('--tokens-col', default='reviewText_tokens', help='Column name containing tokens')
        self.add_passthru_arg('--csv-header', help='CSV header line to use for all mappers')

    def mapper_init(self):
        """Initialize the CSV header to map columns properly."""
        try:
            reader = csv.reader(io.StringIO(self.options.csv_header))
            self.header = next(reader)
        except Exception as e:
            self.stderr.write(f"ERROR: Failed to parse CSV header: {e}\n".encode('utf-8'))
            self.header = []

    def mapper_extract_terms(self, _, line_input):
        """
        Phase 1:
        For each document (row):
          - Output (“DOC_COUNT”, category) = 1  => # docs per category
          - For each unique token in the doc, output (category, token) = 1 => # docs with token in that category
          - For each unique token in the doc, output (“TOKEN_COUNT”, token) = 1 => # docs that have this token overall
        """
        if isinstance(line_input, bytes):
            line_str = line_input.decode('utf-8', errors='replace')
        else:
            line_str = line_input

        # Parse the row
        reader = csv.reader(io.StringIO(line_str))
        try:
            row_values = next(reader)
        except StopIteration:
            return  # Empty line
        except Exception as e:
            self.stderr.write(f"ERROR processing line: {e} - {line_str[:100]}...\n".encode('utf-8'))
            return

        # Check if the row has the same columns as the header
        if len(row_values) != len(self.header):
            logging.warning(f"Column mismatch: expected {len(self.header)} cols, got {len(row_values)} -> {row_values}")
            return

        row_dict = dict(zip(self.header, row_values))
        category = row_dict.get(self.options.category_col)
        tokens_str = row_dict.get(self.options.tokens_col)

        if not category or not tokens_str:
            logging.warning(f"Missing fields: {row_dict}")
            return

        try:
            tokens = ast.literal_eval(tokens_str)
            if not isinstance(tokens, list):
                raise ValueError("Parsed tokens are not a list.")
        except Exception as e:
            logging.warning(f"Token parse error: {e} in row: {row_dict}")
            return

        logging.info(f"Parsed row OK: category={category}, tokens={tokens[:5]}...")

        # Emit doc count for the category
        yield ("DOC_COUNT", category), 1

        # We only want to count token presence once per document => use a set
        seen_tokens = set()
        for token in tokens:
            if isinstance(token, str):
                token_clean = token.lower().strip()
                if token_clean and token_clean not in seen_tokens:
                    seen_tokens.add(token_clean)
                    yield (category, token_clean), 1
                    yield ("TOKEN_COUNT", token_clean), 1

    def combiner(self, key, values):
        """Local summation of partial counts."""
        yield key, sum(values)

    def reducer_sum_counts(self, key, values):
        """
        Summation of all:
          - (“DOC_COUNT”, category) => # docs in that category
          - (“TOKEN_COUNT”, token)  => # docs containing that token
          - (category, token) => # docs containing token in that category
        """
        yield key, sum(values)

    # -----------------------------------------------------------
    # Phase 2: Gathering sums, computing chi-square
    # -----------------------------------------------------------
    def mapper_organize_for_chi(self, key, count):
        key_type = key[0]
        key_val = key[1]
        
        # Force everything into a single reducer
        if key_type == '__GLOBAL_TOTAL__':
            yield ('__GLOBAL__', 'N_val'), count
        elif key_type == '__CATEGORY_TOTAL__':
            yield ('__GLOBAL__', 'AC_val'), (key_val, count)
        elif key_type == '__TOKEN_TOTAL__':
            yield ('__GLOBAL__', 'TOKEN_'+key_val), count
        else:
            yield ('__GLOBAL__', 'PAIR_'+key_type+'_'+key_val), count

    def reducer_calculate_chi(self, main_key, values):
        """
        Reducer 2: Receives both broadcast data (N, category totals)
        and token data (A, A+B) under a single key ("ALLDATA").
        """
        # Only one main_key = ('ALLDATA', 'BROADCAST') or ('ALLDATA', 'TOKEN')
        # We accumulate everything in memory, since it's all one partition.
        for val in values:
            what_type = val[0]  # e.g. 'N_val', 'AC_val', 'AB_val', 'A_val', ...
            if what_type == 'N_val':
                # Store global N
                self.N += val[1]
            elif what_type == 'AC_val':
                # Store category totals
                category, cat_count = val[1]
                self.category_totals[category] = (
                    self.category_totals.get(category, 0) + cat_count
                )
            elif what_type == 'AB_val':
                # Store total docs with token
                token, ab_count = val[1]
                # Access the dictionary entry for the token
                self.token_data_buffer[token]['AB'] += ab_count
            elif what_type == 'A_val':
                # Store docs in category with token
                token, category, a_count = val[1]
                self.token_data_buffer[token]['A'][category] = (
                    self.token_data_buffer[token]['A'].get(category, 0) + a_count
                )

        # If we’ve just finished scanning 'BROADCAST' items, we do nothing except store them.
        # If we’re finishing 'TOKEN' items, after that we can do the chi-square math.
        # Typically, you’d run _process_buffered_tokens in a final function. Or check main_key[1].
        if main_key[1] == 'TOKEN':
            # Now that the broadcast data is presumably read by the same reducer,
            # we can do the final calculations for any tokens we have so far.
            for token_val, output in self._process_token_data():
                yield token_val, output
            # Optionally clear data structures here if you want chunk processing.

    def mapper_prepare_for_chi(self, key, aggregated_count):
        """
        Reorganize data for the second reduce phase:
          1) (“DOC_COUNT”, category) => (“DOC_COUNT_BROADCAST”, category), aggregated_count
          2) (“TOKEN_COUNT”, token)  => (“TOKEN_COUNT_BROADCAST”, token), aggregated_count
          3) (category, token) => (token, “PAIR”), (category, aggregated_count)
        """
        label, value = key
        if label == "DOC_COUNT":
            yield ("DOC_COUNT_BROADCAST", label), (value, aggregated_count)  # (category, doc_count)
        elif label == "TOKEN_COUNT":
            yield ("TOKEN_COUNT_BROADCAST", label), (value, aggregated_count)  # (token, global_token_count)
        else:
            # If key is (category, token), rearrange so that we group by token
            category = label
            token = value
            yield (token, "PAIR"), (category, aggregated_count)

    def reducer_init_chi(self):
        """Initialize dicts to store doc counts, token counts, and buffer pairs before we compute chi-square."""
        self.category_doc_counts = {}  # category => # docs in that category
        self.token_doc_counts = {}     # token => # docs containing that token
        self.pairs_buffer = {}         # token => list of (category, A)

    def reducer_collect_and_buffer(self, key, values):
        """
        Collect needed values:
         - (“DOC_COUNT_BROADCAST”, “DOC_COUNT”) -> (category, doc_count)
         - (“TOKEN_COUNT_BROADCAST”, “TOKEN_COUNT”) -> (token, doc_count)
         - (token, “PAIR”) -> list of (category, doc_count) i.e. # docs in that category that contain that token
        """
        label, purpose = key
        if label == "DOC_COUNT_BROADCAST":
            # items => (category, aggregated_count)
            for (cat, count_val) in values:
                self.category_doc_counts[cat] = count_val
        elif label == "TOKEN_COUNT_BROADCAST":
            # items => (token, aggregated_count)
            for (tok, count_val) in values:
                self.token_doc_counts[tok] = count_val
        else:
            # key => (token, “PAIR”) => list of (category, A)
            token = label
            for (cat, a_val) in values:
                if token not in self.pairs_buffer:
                    self.pairs_buffer[token] = []
                self.pairs_buffer[token].append((cat, a_val))

    def reducer_final_compute_chi(self):
        """
        Now compute the chi-square for each (token, category) from pairs_buffer, using:
          N = total number of documents
          A = # docs in category that have the token
          B = # docs in other categories that have the token
          C = total # docs in category (with or without the token)
          D = the remainder of docs (no token, not in category)

        Chi-sq formula:
          chi = [N * (A*D - B*(C-A))^2 ] / [ (A+B)*(C+(D-C))*(A+(C-A))*(B+(D-(C-A))) ]

        or more simply:
          chi = (N * (A*D - B*C)^2 ) / ((A + B) * (C + D) * (A + C) * (B + D))

        so long as A, B, C, D use doc-level presence/absence values.
        """
        # total docs across all categories
        N = sum(self.category_doc_counts.values())

        results = []
        for token, cat_list in self.pairs_buffer.items():
            token_total = self.token_doc_counts.get(token, 0)
            for (category, A) in cat_list:
                C_total = self.category_doc_counts.get(category, 0)
                C_formula = C_total - A  # Docs in category WITHOUT the token
                B = token_total - A
                D = N - C_total - B  # Docs without token in other categories (correct)

                # Compute denominator terms
                denom_product = (A + B) * (A + C_formula) * (B + D) * (C_formula + D)
                chi = (N * (A * D - B * C_formula) ** 2) / denom_product if denom_product != 0 else 0
                logging.debug(f"A={A}, B={B}, C_total={C_total}, C_formula={C_formula}, D={D}")

                expected_val = (token_total * C_formula / float(N)) if N else 0

                # Collect the result
                results.append({
                    "token": token,
                    "category": category,
                    "observed": A,
                    "expected": expected_val,
                    "chi_square": chi,
                    "total_term": token_total,     # how many docs contain this token overall
                    "total_category": C_total            # how many docs in this category overall
                })

        # Once done, yield them all
        for item in results:
            yield (item["token"], item["category"]), {
                "observed": item["observed"],
                "expected": item["expected"],
                "chi_square": item["chi_square"],
                "total_term": item["total_term"],
                "total_category": item["total_category"]
            }

    def steps(self):
        """
        Step 1:
          - mapper_extract_terms => aggregates doc-level presence data
          - combiner => sums partial
          - reducer_sum_counts => final sums

        Step 2:
          - mapper_prepare_for_chi => reorganize data
          - reducer_collect_and_buffer => gather doc counts into arrays
          - reducer_final_compute_chi => compute the chi-square
        """
        return [
            MRStep(
                mapper_init=self.mapper_init,
                mapper=self.mapper_extract_terms,
                combiner=self.combiner,
                reducer=self.reducer_sum_counts
            ),
            MRStep(
                mapper=self.mapper_prepare_for_chi,
                reducer_init=self.reducer_init_chi,
                reducer=self.reducer_collect_and_buffer,
                reducer_final=self.reducer_final_compute_chi
            )
        ]

if __name__ == '__main__':
    # Run the job and write out chi_square_results.txt and output.txt for manual inspection
    args = sys.argv[1:]
    job = ChiSquareUnigrams(args=args)

    with job.make_runner() as runner:
        runner.run()
        results = []
        for (token, category), stats in job.parse_output(runner.cat_output()):
            results.append({
                "token": token,
                "category": category,
                "observed": stats["observed"],
                "expected": stats["expected"],
                "chi_square": stats["chi_square"],
                "total_term": stats["total_term"],
                "total_category": stats["total_category"]
            })

        # Store all results in a dataframe so we can easily write them out
        if results:
            chi_results = pd.DataFrame(results)
            # 1) Write a main results file
            chi_results.to_csv('chi_square_results.txt', sep='\t', index=False, encoding='utf-8')
            print("Full results saved to chi_square_results.txt")

            # 2) Extract top 75 terms by descending chi-square per category
            top75 = (chi_results.sort_values(by=["category", "chi_square"], ascending=[True, False])
                     .groupby("category")
                     .head(75))

            # 3) Output as required: one line per category, top terms in descending chi
            all_categories = sorted(set(top75["category"]))
            with open("output.txt", "w", encoding="utf-8") as f:
                for cat in all_categories:
                    sub = top75[top75["category"] == cat]
                    line_parts = [
                        f"{row['token']}:{round(row['chi_square'], 3)}"
                        for _, row in sub.iterrows()
                    ]
                    line = f"{cat} " + " ".join(line_parts)
                    f.write(line.strip() + "\n")

                # Then one line with merged dictionary (all top terms from all categories), alphabetical
                merged_terms = sorted(set(top75["token"]))
                f.write(" ".join(merged_terms) + "\n")

            print("Top 75 terms per category + merged dictionary written to output.txt")
        else:
            print("No results found.")