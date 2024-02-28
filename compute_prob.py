from lm import LMProb
from lm.data import Dictionary
import tqdm
import sys

sys.path.append("./lm")

lms = [
    r'result_models\lm_models\lm_java.pt',
    r'result_models\lm_models\lm_nl.pt'
]
read_file_paths = [
    r'data\dual_data\java\tmp3\code.original',
    r'data\dual_data\java\tmp3\javadoc.original'
]
dicts = [r'data\lm_data\java\dict_code.pkl',
         r'data\lm_data\nl\dict_code.pkl']
write_file_paths = [
    r'data\dual_data\java\tmp3\code.score',
    r'data\dual_data\java\tmp3\javadoc.score'
]


def get_score(line, num):
    sent = line.strip().split(' ')
    lm_score = lm_model.get_prob(sent)
    return (num, lm_score)


for i in range(2):
    lm_model = LMProb(lms[i], dicts[i])
    fw = open(write_file_paths[i], 'w')
    f = open(read_file_paths[i])
    lines = f.readlines()
    f.close()
    # with ProcessPoolExecutor(max_workers=8) as executor:
    #     results = executor.map(get_score, lines, list(range(len(lines))))
    # scores = {}
    scores = []
    for line in tqdm.tqdm(lines):
        line = line.strip().split(' ')
        score = lm_model.get_prob(line)
        scores.append(score)
    for i in range(len(lines)):
        fw.write(str(scores[i]))
        fw.write('\n')
    fw.close()
