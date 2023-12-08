from argparse import ArgumentParser
import json
import random
from typing import Tuple, Mapping
from tqdm import tqdm

PHRASES_CONFIG = {
    'Nationality': ('who', 'is', 'is not'),
    'Food': ('that', 'likes', "doesn't eat"),
    'Pet': ('that', 'has a', 'does not own a'),
    'Job': ('who', 'is a', 'is not a'),
    'Beverage': ('who', 'drinks', 'dislikes'),
    'Transport': ('that', 'travels by', 'avoids getting on a'),
    'Music-Genre': ('who', 'listens to', 'cannot stand'),
    'Movie-Genre': ('that', 'watches', 'hates'),
    'Sport': ('that', 'plays', 'cannot play'),
    'Hobby': ('who', 'likes', 'will not even try'),
}

PTH = [None,
       'first', 'second', 'third', 'fourth', 'fifth',
       'sixth', 'seventh', 'eighth', 'nineth', 'tenth']

POSITION_Q: Mapping[str, Tuple[str, str]] = {
    'Nationality': ('What is the nationality of {qualifier}?',
                    'The person is {val}.'),
    'Food': ('What food does {qualifier} like?',
             'They like to eat {val}.'),
    'Pet': ('What kind of pet does {qualifier} own?',
            'The person owns a {val}.'),
    'Job': ('What job does {qualifier} have?',
            'The person is a {val}.'),
    'Beverage': ('What does {qualifier} prefer to drink?',
                 'They like to drink {val}.'),
    'Transport': ('How does {qualifier} travel?',
                  'They travel by {val}.'),
    'Music-Genre': ('What kind of music does {qualifier} enjoy?',
                    'The person listens to {val}.'),
    'Movie-Genre': ('What movie genre does {qualifier} prefer?',
                    'The person watches {val}.'),
    'Sport': ('What sport does {qualifier} play?',
              'They play {val}'),
    'Hobby': ('What is the hobby of {qualifier}?',
              'Their hobby is {val}'),
}


PROMPT_TEMPLATE = """\
There are {n_objects} people standing in a line numbered 1 through {n_objects} in a left to right order.
Each person has a set of attributes: {attributes}.
The attributes have the following possible values:
{attribute_values}
and exactly one person in the line has a given value for an attribute.

Given the following premises about the line of people:
{premises_list}

Answer the following question:
"""

RESPONSE_TEMPLATE = """\
The premises are satisfied by the following assignments:
{assignments_table}

Using this table the answer is:
{answer}
"""


def process(problem: dict):
    attrs = list(problem['attributes'])
    random.shuffle(attrs)
    premises = list(problem['premises'])
    av_list = []
    for attr in attrs:
        values = list(problem['answer'][attr])
        random.shuffle(values)
        av_list.append(f'- {attr}: {", ".join(values)}')
    attribute_values = '\n'.join(av_list)
    random.shuffle(premises)
    premises_list = '\n'.join('- ' + p for p in premises)
    prompt = PROMPT_TEMPLATE.format(
            n_objects=problem['n_objects'],
            attributes=', '.join(attrs),
            attribute_values=attribute_values,
            premises_list=premises_list)

    random.shuffle(attrs)
    attrh = ['-' * len(a) for a in attrs]
    soln = problem['answer']
    assignments_table = '\n'.join([
        f'| Person | {" | ".join(attrs)} |',
        f'|--------|-{"-|-".join(attrh)}-|'
    ] + [
        f'| {idx + 1} |{" | ".join([soln[a][idx] for a in attrs])} |'
        for idx in range(problem['n_objects'])
    ])

    question_type = random.choice(['position', 'at-position', 'by-attr'])

    position = random.randint(1, problem['n_objects'])
    attr = random.choice(attrs)
    val = problem['answer'][attr][position - 1]
    question, answer = None, None
    if question_type == 'position':
        qt, at = POSITION_Q[attr]
        pth = PTH[position]
        qualifier = random.choice([
            f'the {pth} person',
            f'the person in the {pth} position',
            f'the person at the {pth} position',
        ])
        question = qt.format(qualifier=qualifier)
        answer = at.format(val=val)
    elif question_type == 'at-position':
        obj, verb, _ = PHRASES_CONFIG[attr]
        question = f'At what position is the person {obj} {verb} {val}?'
        answer = f'At position {position}'
    elif question_type == 'by-attr':
        other_attr = random.choice([a for a in attrs if a != attr])
        other_val = problem['answer'][other_attr][position - 1]
        obj, verb, _ = PHRASES_CONFIG[other_attr]
        qualifier = f'the person {obj} {verb} {other_val}'
        qt, at = POSITION_Q[attr]
        question = qt.format(qualifier=qualifier)
        answer = at.format(val=val)

    return [
        {
            'role': 'user',
            'content': prompt + question
        },
        {
            'role': 'assistant',
            'content': RESPONSE_TEMPLATE.format(
                assignments_table=assignments_table,
                answer=answer)
        }
    ]


def main():
    parser = ArgumentParser(description='Create training samples from problems.')
    parser.add_argument('--problems', required=True, help='File with problems')
    parser.add_argument('--output', required=True, help='Training sample output file.')
    parser.add_argument('--npp', default=10, type=int, help='Number of examples per problem.')
    parser.add_argument('--seed', default=0x5eed, type=int, help='Seed for RNG')
    args = parser.parse_args()
    random.seed(args.seed)

    with open(args.problems) as problems_file, open(args.output, 'w') as output_file:
        for problem_json in tqdm(iter(problems_file)):
            problem = json.loads(problem_json)
            for _ in range(args.npp):
                training_sample = process(problem)
                json.dump(training_sample, output_file)
                output_file.write('\n')


if __name__ == '__main__':
    main()
