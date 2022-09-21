from typing import Dict
import argparse, math

argparser = argparse.ArgumentParser()
argparser.add_argument("-n", type=int)
argparser.add_argument("-r", nargs='*', type=str)
argparser.add_argument("-c", nargs="*", type=str)
argparser.add_argument("-l", nargs='*', type=float)

def count_dict(gram_len: int, input_list: list):
    cur_count = dict()
    for idx in range(len(input_list)-gram_len+1):
        gram = input_list[idx:idx+gram_len]
        cur_count[str(gram)] = cur_count.get(str(gram), 0)+1
    return cur_count

def bleu(args: argparse.Namespace):
    max_len = args.n
    refs = [ref.split() for ref in args.r]
    cands = [cand.split() for cand in args.c]
    lambdas = args.l
    ref_counts = dict()
    for gram_len in range(1, max_len+1):
        for ref in refs:
            cur_ref_count = count_dict(gram_len, ref)
            for gram,count in cur_ref_count.items():
                ref_counts[gram] = max(ref_counts.get(gram, 0), count)
    scores = []
    for cand in cands:
        cur_scores = []
        for gram_len in range(1, max_len+1):
            cand_counts = count_dict(gram_len, cand)
            numerator = 0
            for gram,count in cand_counts.items():
                numerator += min(ref_counts.get(str(gram), 0), cand_counts[str(gram)])
            denominator = len(cand)-gram_len+1
            p = numerator / denominator
            print(p)
            cur_scores.append(p)
        print(len(cand))
        ref_max_len = max([len(ref) for ref in refs])
        print(ref_max_len)
        bp = math.exp(1-ref_max_len/len(cand)) if len(cand) < ref_max_len else 1
        print(bp)
        final_score = 0 if math.prod(cur_scores) == 0 else bp*math.exp(sum([lambdas[i]*math.log(cur_scores[i]) for i in range(len(cur_scores))]))
        scores.append(final_score)
    return scores

if __name__ == '__main__':
    args = argparser.parse_args()
    print(bleu(args))