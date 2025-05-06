import sys
import json
import re
from mrjob.job import MRJob
from mrjob.step import MRStep
from mrjob.protocol import RawValueProtocol


class HadoopPreprocessor(MRJob):
    OUTPUT_PROTOCOL = RawValueProtocol

    def configure_args(self):
        super().configure_args()
        self.add_file_arg('--stopwords')

    def mapper_init(self):
        self.stopwords = set()
        if self.options.stopwords:
            with open(self.options.stopwords) as f:
                self.stopwords = set(line.strip() for line in f)

    def mapper(self, _, line):
        try:
            data = json.loads(line)
            tokens = self.tokenize_and_filter(
                data.get('reviewText', '') + ' ' + data.get('summary', ''),
                self.stopwords
            )
            yield None, {
                'reviewText_tokens': tokens,
                'category': data.get('category', '')
            }
        except Exception as e:
            sys.stderr.write(f"ERROR: {str(e)}\n")

    def reducer(self, key, values):
        yield None, "reviewText_tokens,category"

        for value in values:
            tokens_str = str(value['reviewText_tokens']).replace('"', "'")
            yield None, f'"{tokens_str}",{value["category"]}'

    def tokenize_and_filter(self, text, stopwords):
        delimiter_chars = r'()\[\]{}.!?,;:+=\-_"\'`~#@&*%€$§\\/'
        split_pattern = rf'[\s\d{re.escape(delimiter_chars)}]+'
        tokens = re.split(split_pattern, str(text).lower())
        return [t for t in tokens if t and t not in stopwords and len(t) > 1]


if __name__ == '__main__':
    HadoopPreprocessor.run()