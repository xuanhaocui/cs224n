# Calculate the accuracy of a baseline that simply predicts "London" for every
#   example in the dev set.
# Hint: Make use of existing code.
# Your solution here should only be a few lines.
if __name__ == '__main__':
    import argparse
    import utils
    argp = argparse.ArgumentParser()
    argp.add_argument('--eval_corpus_path',
        help="Path of the corpus to evaluate on", default=None)
    args = argp.parse_args()
    num_ans = sum([1 for x in open(args.eval_corpus_path, encoding='utf8')])
    ret = utils.evaluate_places(args.eval_corpus_path, ['London'] * num_ans)
    print("Accuracy: {}/{}, {}%".format(ret[1], ret[0], ret[1]/ret[0]*100))