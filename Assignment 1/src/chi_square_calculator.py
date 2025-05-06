import sys
try:
    import ujson as json  # faster JSON parser if available
except ImportError:
    import json

from mrjob.job import MRJob
from mrjob.step import MRStep
from mrjob.protocol import RawValueProtocol  

class ChiSquareCalculator(MRJob):
    # final output format will be plain strings without JSON or key prefixes
    OUTPUT_PROTOCOL = RawValueProtocol

    # assignment provided delimiters
    _DELIMS = r'''()[]{}.!?,;:+=-_"'`~#@&*%€§\\/0123456789'''
    # translation table to replace delimiters with whitespace
    TRANSLATOR = str.maketrans({c: ' ' for c in _DELIMS})

    def configure_args(self):
        # adding command-line argument for stopword file
        super(ChiSquareCalculator, self).configure_args()
        self.add_file_arg('--stopwords')

    def mapper_init(self):
        # loading stopwords
        sw = set()
        if self.options.stopwords:
            with open(self.options.stopwords) as f:
                for line in f:
                    sw.add(line.strip())
        self.stopwords = sw
        self.translator = ChiSquareCalculator.TRANSLATOR

        # initialize buffer for emitted key-value pairs
        self.buf = {}
        self.ev_count = 0  # event counter manages buffer flush threshold

    def mapper(self, _, line):
        #  reading in each JSON review line, skipping malformed ones
        try:
            doc = json.loads(line)
        except:
            return

        # extracting category and count doc for its category
        cat = doc.get('category', '')
        self.ev_count += 1
        self.buf[('!DOC_COUNT', cat)] = self.buf.get(('!DOC_COUNT', cat), 0) + 1

        # combinimg and preprocessing text
        text = (doc.get('reviewText', '') + ' ' + doc.get('summary', '')).lower()
        toks = text.translate(self.translator).split()

        # filter tokens
        seen = set()
        for t in toks:
            if len(t) > 1 and t not in self.stopwords and t not in seen:
                seen.add(t)
                self.ev_count += 2
                # emitting category-specific token frequency
                self.buf[(cat, t)] = self.buf.get((cat, t), 0) + 1
                # emitting global token frequency
                self.buf[('*', t)] = self.buf.get(('*', t), 0) + 1

        # flush buffer when threshold is reached
        if self.ev_count >= 200000:
            for k, v in self.buf.items():
                yield k, v
            self.buf.clear()
            self.ev_count = 0

    def mapper_final(self):
        # emitting remaining buffered items
        for k, v in self.buf.items():
            yield k, v

    def combiner(self, key, counts):
        # summing values locally
        yield key, sum(counts)

    def reducer_sum(self, key, counts):
        # final aggregation of all counts
        yield key, sum(counts)

    def mapper_stage2(self, key, count):
        # rearranging keys to group data for chi-square input
        kind, val = key
        if kind == '!DOC_COUNT':
            yield ('__DOC_COUNT__', val), count
        elif kind == '*':
            yield ('__TERM_TOTAL__', val), count
        else:
            yield (val, kind), count

    def combiner_stage2(self, key, counts):
        # local sums
        yield key, sum(counts)

    def reducer_stage2(self, key, counts):
        # collecting all values under a common key for final reducer
        yield '__GLOBAL__', (key, sum(counts))

    def reducer_final_init(self):
        # initializing dictionaries
        self.doc_counts = {}
        self.term_totals = {}
        self.observations = []

    def reducer_final(self, _, items):
        # parsing grouped input and build in-memory datasets
        for (kind, ident), cnt in items:
            if kind == '__DOC_COUNT__':
                self.doc_counts[ident] = cnt
            elif kind == '__TERM_TOTAL__':
                self.term_totals[ident] = cnt
            else:
                self.observations.append((kind, ident, cnt))

        # total number of documents
        N = sum(self.doc_counts.values())
        buckets = {} # for categories

        # compute chi-square per (token, category)
        for term, cat, A in self.observations:
            T = self.term_totals.get(term, 0)
            C = self.doc_counts.get(cat, 0)
            B = T - A
            D = N - C - B - A
            denom = (A+B)*(C+D)*(A+C)*(B+D)
            if denom == 0:
                continue
            chi2 = N * (A*D - B*C)**2 / denom
            buckets.setdefault(cat, []).append((term, chi2))

        # collecting top 75 terms per category and merge vocabulary
        merged_terms = set()
        for cat in sorted(buckets):
            top = sorted(buckets[cat], key=lambda x: -x[1])[:75]
            merged_terms.update(t for t, _ in top)
            # emitting top 75 per category
            yield None, cat + ' ' + ' '.join(f"{t}:{v:.3f}" for t, v in top)
        # emitting merged dictionary
        merged = sorted(merged_terms)
        yield None, ' '.join(merged)

    def steps(self):
        tune = {
            'mapreduce.input.fileinputformat.split.maxsize': '134217728',
            'mapreduce.input.fileinputformat.split.minsize': '1048576',
            'mapred.job.inputformat.class': 'org.apache.hadoop.mapred.lib.CombineTextInputFormat',
            'mapreduce.job.reduces': '5',
            'mapreduce.map.output.compress': 'true',
            'mapreduce.map.output.compress.codec': 'org.apache.hadoop.io.compress.SnappyCodec',
            'mapreduce.output.fileoutputformat.compress': 'true',
            'mapreduce.output.fileoutputformat.compress.codec': 'org.apache.hadoop.io.compress.SnappyCodec',
            'mapreduce.job.jvm.numtasks': '-1',
            'mapreduce.map.speculative': 'true',
            'mapreduce.reduce.speculative': 'true',
        }

        return [
            # Stage 1: document + token counting
            MRStep(
                mapper_init=self.mapper_init,
                mapper=self.mapper,
                mapper_final=self.mapper_final,
                combiner=self.combiner,
                reducer=self.reducer_sum,
                jobconf=tune
            ),
            # Stage 2: key reorganization for chi-square
            MRStep(
                mapper=self.mapper_stage2,
                combiner=self.combiner_stage2,
                reducer=self.reducer_stage2,
                jobconf=tune
            ),
            # Stage 3: final chi-square computation and term selection
            MRStep(
                reducer_init=self.reducer_final_init,
                reducer=self.reducer_final
            ),
        ]

if __name__ == '__main__':
    ChiSquareCalculator.run()