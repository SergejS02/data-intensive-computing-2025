import sys
import pandas as pd
import argparse
from calculate_chi_square import ChiSquareUnigrams


def run_job(input_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        header_line = f.readline().strip()

    args = [
        input_path,
        '--csv-header', header_line
    ]

    results = []
    job = ChiSquareUnigrams(args=args)
    with job.make_runner() as runner:
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

    return pd.DataFrame(results)


def save_output(df: pd.DataFrame):

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

def save_output_full(df: pd.DataFrame):
    df.to_csv('chi_square_results.txt', sep='\t', index=False, encoding='utf-8')
    print(f"Full results saved to: chi_square_results.txt")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", help="Path to the input CSV file")
    parser.add_argument("--save_full_results", action="store_true")

    args = parser.parse_args()

    df = run_job(args.input_csv)

    if not df.empty:
        save_output(df)
        if args.save_full_results:
            save_output_full(df)
    else:
        print("No results to process or save.")