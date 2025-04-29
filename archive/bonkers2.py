#!/usr/bin/env python3
import sys
# ultra-fast JSON parsing if available
try:
    import ujson as json
except ImportError:
    import json

from mrjob.job import MRJob
from mrjob.step import MRStep
from mrjob.protocol import RawValueProtocol

class ChiSquareCalculator(MRJob):
    OUTPUT_PROTOCOL = RawValueProtocol

    # characters to translate to spaces (delimiters + digits)
    _DELIMS = '()[]{}.!?,;:+=-_"\'"`~#@&*%€$§\\/0123456789'
    TRANSLATOR = str.maketrans({c: ' ' for c in _DELIMS})

    def configure_args(self):
        super(ChiSquareCalculator, self).configure_args()
        self.add_file_arg('--stopwords')

    def mapper_init(self):
        # load stopwords
        sw = set()
        if self.options.stopwords:
            with open(self.options.stopwords) as f:
                for line in f:
                    sw.add(line.strip())
        self.stopwords = sw
        self.translator = ChiSquareCalculator.TRANSLATOR

        # in-mapper combiner buffers
        self.buf = {}
        self.ev_count = 0

    def mapper(self, _, line):
        try:
            doc = json.loads(line)
        except Exception:
            return

        cat = doc.get('category', '')
        self.ev_count += 1
        self.buf[('!DOC_COUNT', cat)] = self.buf.get(('!DOC_COUNT', cat), 0) + 1

        text = (doc.get('reviewText','') + ' ' + doc.get('summary','')).lower()
        toks = text.translate(self.translator).split()

        seen = set()
        sw = self.stopwords
        for t in toks:
            if len(t) > 1 and t not in sw and t not in seen:
                seen.add(t)
                self.ev_count += 2
                self.buf[(cat, t)] = self.buf.get((cat, t), 0) + 1
                self.buf[('*', t)]   = self.buf.get(('*', t), 0) + 1

        if self.ev_count >= 100000:
            for k, v in self.buf.items():
                yield k, v
            self.buf.clear()
            self.ev_count = 0

    def mapper_final(self):
        for k, v in self.buf.items():
            yield k, v

    def combiner(self, key, counts):
        yield key, sum(counts)

    def reducer_sum(self, key, counts):
        yield key, sum(counts)

    def mapper_stage2(self, key, count):
        kind, val = key
        if kind == '!DOC_COUNT':
            yield ('__DOC_COUNT__', val), count
        elif kind == '*':
            yield ('__TERM_TOTAL__', val), count
        else:
            yield (val, kind), count

    def combiner_stage2(self, key, counts):
        yield key, sum(counts)

    def reducer_stage2(self, key, counts):
        yield '__GLOBAL__', (key, sum(counts))

    def reducer_final_init(self):
        self.doc_counts = {}
        self.term_totals = {}
        self.observations = []

    def reducer_final(self, _, items):
        for (kind, identifier), cnt in items:
            if kind == '__DOC_COUNT__':
                self.doc_counts[identifier] = cnt
            elif kind == '__TERM_TOTAL__':
                self.term_totals[identifier] = cnt
            else:
                term, cat = kind, identifier
                self.observations.append((term, cat, cnt))

        N = sum(self.doc_counts.values())
        buckets = {}

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

        for cat in sorted(buckets):
            top = sorted(buckets[cat], key=lambda x: -x[1])[:75]
            line = cat + ' ' + ' '.join(f"{t}:{v:.3f}" for t, v in top)
            yield None, line

        merged = sorted({t for lst in buckets.values() for t,_ in lst})
        yield None, ' '.join(merged)

    def steps(self):
        tune = {
            'mapreduce.input.fileinputformat.split.maxsize': '134217728',  # 128 MB splits
            'mapreduce.input.fileinputformat.split.minsize': '1048576',    # 1 MB min
            'mapreduce.job.reduces': '5',                                 # more parallel reducers
            'mapreduce.map.output.compress': 'true',                      # compress shuffle
            'mapreduce.map.output.compress.codec': 'org.apache.hadoop.io.compress.SnappyCodec',
            'mapreduce.output.fileoutputformat.compress': 'true',         # optional final compress
            'mapreduce.output.fileoutputformat.compress.codec': 'org.apache.hadoop.io.compress.SnappyCodec',
            'mapred.input.format.class': 'org.apache.hadoop.mapred.lib.CombineTextInputFormat',  # combine small files
        }

        return [
            MRStep(
                mapper_init=self.mapper_init,
                mapper=self.mapper,
                mapper_final=self.mapper_final,
                combiner=self.combiner,
                reducer=self.reducer_sum,
                jobconf=tune,
            ),
            MRStep(
                mapper=self.mapper_stage2,
                combiner=self.combiner_stage2,
                reducer=self.reducer_stage2,
                jobconf=tune,
            ),
            MRStep(
                reducer_init=self.reducer_final_init,
                reducer=self.reducer_final,
            ),
        ]

if __name__ == '__main__':
    ChiSquareCalculator.run()
