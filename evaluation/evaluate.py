import json
import re
import nltk
import wandb
import argparse

from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support

from collections import Counter

# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')
# nltk.download('punkt')

benchmark_output_type = {
    'pubmedqa': 'boolean',
    'newpubmedqa': 'boolean',
    'medmcqa': 'mcq',
    'mmlu_medical': 'mcq',
    'mmlu_general': 'mcq',
    'medqa': 'mcq',
    'medqa4': 'mcq',
    'blurb': 'ner',
    'gsm8k': 'numeric',
    'truthfulqa': 'boolean',
}

def load_json(filename):
    """Load json file"""
    with open(filename, 'r') as read_file:
        data = json.load(read_file)
    return data

def load_jsonl(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Error decoding JSON for line: {line}")
    return data

def save_dictlist_to_json(mydictlist, filename):
    """Save a list of dictionaries to json file"""
    f = open(filename, 'w', encoding='utf-8')
    json.dump(mydictlist, f, ensure_ascii=False, indent=4)
    f.close()

def clean_mcq_answer(output):
    output = clean_answer(output)
    try:
        output = output[0]
    except Exception:
        return output
    return output

def clean_double_answer(output):
    if "yesyes" in output:
        output = output.replace('yesyes', 'yes')
    elif "nono" in output:
        output = output.replace('nono', 'no')
    elif "yesno" in output:
        output = output.replace('yesno', 'yes')
    elif "noyes" in output:
        output = output.replace('noyes', 'no')
    output = clean_answer(output)
    return output

def clean_answer(output):
    output_clean = output.encode('ascii', 'ignore').decode('ascii')
    return output_clean

def lemmatize(phrase: str):
    """
    Lemmatize a phrase using the WordNet lemmatizer.
    """
    lemmatizer = WordNetLemmatizer()
    def pos_tagger(nltk_tag):
        if nltk_tag.startswith('J'):
            return wordnet.ADJ
        elif nltk_tag.startswith('V'):
            return wordnet.VERB
        elif nltk_tag.startswith('N'):
            return wordnet.NOUN
        elif nltk_tag.startswith('R'):
            return wordnet.ADV
        else:
            return None

    pos_tagged = nltk.pos_tag(nltk.word_tokenize(phrase))
    wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))

    lemmatized_phrase = []
    for word, tag in wordnet_tagged:
        if tag is None:
            lemmatized_phrase.append(word)
        else:
            lemmatized_phrase.append(lemmatizer.lemmatize(word, tag))
    lemmatized_phrase = " ".join(lemmatized_phrase)
    return lemmatized_phrase

def evaluate_entities(predictions, ground_truth):
    """
    Evaluate the performance of the model on the entity extraction task.
    """
    if not len(ground_truth) and not len(predictions):
        return 1, 1, 1
    elif not len(ground_truth):
        return 0, 0, 0
    else:
        ps = PorterStemmer()

        predictions = list(map(lambda x: ps.stem(lemmatize(x.lower())), predictions))
        ground_truth = list(map(lambda x: ps.stem(lemmatize(x.lower())), ground_truth))

        tp, fp, fn = [], [], []
        nb_ground_truth = len(ground_truth)
        for pred in predictions:
            if pred in ground_truth:
                tp.append(pred)
                ground_truth.remove(pred)
            elif re.search(r" \(.*?\)", pred):
                pred1 = re.search(r"\(.*?\)", pred).group(0).replace("(", "").replace(")", "").strip()
                pred2 = re.sub(r"\(.*?\)", "", pred)
                if pred1 in ground_truth:
                    tp.append(pred1)
                    ground_truth.remove(pred1)
                elif pred2 in ground_truth:
                    tp.append(pred2)
                    ground_truth.remove(pred2)
            else:
                fp.append(pred)

        fn = [e for e in ground_truth if e not in tp]
        prc = len(tp) / len(predictions) if len(predictions) else 0
        rec = len(tp) / nb_ground_truth
        f1 = 2*rec*prc / (prc+rec) if prc+rec else 0
        return prc, rec, f1

def verbose_metric_report(metric_dict):
    print(f'# Accuracy: {metric_dict["accuracy"]}')
    print(f'# Accuracy (calibrated): {metric_dict["accuracy_calibrate"]}')
    print(f'# Precision: {metric_dict["precision"]}')
    print(f'# Recall: {metric_dict["recall"]}')
    print(f'# F1: {metric_dict["f1"]}')

    print(f'# Correct: {metric_dict["correct"]}')
    print(f'# Counted: {metric_dict["counted"]}')
    print(f'# Total: {metric_dict["total"]}')
    print(f'# Unable to find answer: {metric_dict["unable_to_find_answer"]}')
    print(f'# Ignored prompts: {len(metric_dict["ignored"])}')

def eval(output_full, answer, shot=False, cot=False, answer_type="mcq"):
    output = output_full
    default = (2, output_full, answer)

    if "\n##" in output:
        try:
            output = output.split("\n##")[1].split("\n")[0].strip().lower()
        except Exception:
            return default
    if "###" in answer:
        try:
            answer = answer.split("answer is:")[1].split("###")[0].strip()
        except Exception:
            return default
    if shot:
        output = output.split("\n\n")[0].strip()

    output = re.sub(r"[^a-zA-Z0-9]", " ", output).strip()
    output = re.sub(" +", " ", output)

    if cot:
        output = output.split("answer is")
        try:
            output = output[-1].split()[0]
        except Exception:
            return default

    if answer_type == 'boolean':
        output = clean_double_answer(output)
    elif answer_type == 'mcq':
        output = clean_mcq_answer(output)

    if output in ['a', 'b', 'c', 'd', 'e', 'yes', 'no']:
        return output == answer, output, answer
    else:
        return default

def ner_metric(data, **kwargs):
    preds = [row['output'] for row in data]
    golds = [row['gold'] for row in data]

    precision, recall, f1 = evaluate_entities(preds, golds)

    print(f'# Precision: {precision}')
    print(f'# Recall: {recall}')
    print(f'# F1: {f1}')

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "total": len(data),
        "ignore": []
    }

def accuracy_metric(data, **kwargs):
    acc, counter, error = 0, 0, 0
    preds, golds = [], []
    ignored_prompts = []
    shot = True if kwargs["shots"] > 0 else False
    for row in data:
        answer = row['gold'].lower()
        output = row['output'].lower()
        correct, pred, gold = eval(
            output, answer, shot=shot,
            cot=kwargs["cot"], answer_type=kwargs["answer_type"])

        preds.append(pred)
        golds.append(gold)

        if correct == 2:
            error += 1
            correct = 0
            ignored_prompts.append(row)
        else:
            acc += correct
            counter += 1

    accuracy =  accuracy_score(preds, golds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        preds, golds, average='weighted', zero_division=0)
    assert accuracy == acc / len(data)

    return {
        "accuracy": accuracy_score(preds, golds),
        "accuracy_calibrate": acc / counter if counter > 0 else 0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "correct": acc,
        "counted": counter,
        "ignored": ignored_prompts,
        "unable_to_find_answer": error,
        "total": len(data)
    }

def sc_cot_accuracy_metric(data, **kwargs):
    matched = {}
    for row in data:
        promtp = row['prompt'].lower()
        answer = row['gold'].lower()
        output = row['output'].lower()

        _, pred, gold = eval(
            output, answer, shot=0,
            cot=True, answer_type=kwargs["answer_type"])

        if promtp in matched:
            matched[promtp]['pred'].append(pred)
        else:
            matched[promtp] = {
                'row': row,
                'gold': gold,
                'pred': [pred]
            }

    acc, counter, error = 0, 0, 0
    preds, golds = [], []
    ignored_prompts = []
    for prompt in matched:
        gold = matched[prompt]['gold']
        pred_pool = matched[prompt]['pred']
        pred = Counter(pred_pool).most_common(1)[0][0]
        preds.append(pred)
        golds.append(gold)
        if pred not in ['a', 'b', 'c', 'd', 'e', 'yes', 'no']:
            error += 1
            counter -= 1
            ignored_prompts.append(matched[prompt]['row'])
        elif pred == gold:
            acc += 1
        counter += 1

    accuracy =  accuracy_score(preds, golds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        preds, golds, average='weighted', zero_division=0)
    assert accuracy == acc / len(matched)

    return {
        "accuracy": accuracy_score(preds, golds),
        "accuracy_calibrate": acc / counter,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "correct": acc,
        "counted": counter,
        "ignored": ignored_prompts,
        "unable_to_find_answer": error,
        "total": len(matched)
    }

def sort_predictions(data, multi_seed, run_name):
    if "mmlu_medical" in run_name:
        subsets = [
            'anatomy',
            'college_biology',
            'college_medicine',
            'professional_medicine',
            'medical_genetics',
            'virology',
            'clinical_knowledge',
            'high_school_biology',
            'nutrition',
        ]
    if "truthfulqa" in run_name:
        subsets = ['Health', 'Nutrition', 'Psychology', 'Science']
    if multi_seed:
        subsets = [1234, 432, 32]
    subset_acc_dict = {subset:{'data': [], 'acc': 0} for subset in subsets}

    for item in data:
        if multi_seed:
            subset_acc_dict[item['seed']]['data'].append(item)
        elif item['subset'] in subset_acc_dict:
            subset_acc_dict[item['subset']]['data'].append(item)
    return subset_acc_dict

def display(metric_dict, run_name, benchmark, subset=None, verbose=False):
    print("====================================")
    if subset is not None:
        print(f'Report accuracy for {run_name} on {benchmark}-{subset}:')
    else:
        print(f'Report accuracy for {run_name} on {benchmark}:')
    print(f'# Accuracy: {metric_dict["accuracy"]}')

    if verbose:
        print(f'# Accuracy (calibrated): {metric_dict["accuracy_calibrate"]}')
        print(f'# Precision: {metric_dict["precision"]}')
        print(f'# Recall: {metric_dict["recall"]}')
        print(f'# F1: {metric_dict["f1"]}')
        print("------------------------------------")
        print(f'# Correct: {metric_dict["correct"]}')
        print(f'# Counted: {metric_dict["counted"]}')
        print(f'# Total: {metric_dict["total"]}')
        print(f'# Unable to find answer: {metric_dict["unable_to_find_answer"]}')
        print(f'# Ignored prompts: {len(metric_dict["ignored"])}')
    print("====================================")

def match_truthfulqa(generations):
    dataset = load_jsonl('../benchmarks/ft_preprocessed/truthfulqa_truthfulqa_test.jsonl')
    matched_data = {}
    for data in dataset:
        matched_data[data['question']] = data

    for generation in generations:
        prompt = generation["prompt"]
        question = prompt.split("Question:")[1].split("\n\n")[0].strip()
        if question in matched_data:
            generation["subset"] = matched_data[question].get("category", "Unknown")
        else:
            print("Not found")
            generation["subset"] = "Unknown"

def main(args):
    args.out_dir = f'{args.out_dir}/{args.benchmark}'

    if args.shots > 0:
        path = f'{args.out_dir}/{args.benchmark}-{args.checkpoint}-{args.shots}-shot.jsonl'
    elif args.sc_cot:
        path = f'{args.out_dir}/{args.benchmark}-{args.checkpoint}.jsonl'
        path = path.replace('cot', 'sc-cot')
    else:
        path = f'{args.out_dir}/{args.benchmark}-{args.checkpoint}.jsonl'

    run_name = path.split('/')[-1].split('.')[0]
    dataset = run_name.split('-')[0]
    model = '-'.join(run_name.split('.')[0].split('-')[1:])
    answer_type = benchmark_output_type[dataset]

    cot = args.cot
    if "medical" in run_name or "cot" in run_name:
        cot = True

    data = load_jsonl(path)
    # prompt_pth = f'{args.out_dir}/{args.benchmark}-{args.checkpoint}-ignored.json'
    # prompts = load_json(prompt_pth)
    # data.extend(prompts)

    reduced = []
    if "mmlu_medical" in run_name:
        for sample in data:
            if "chemistry" not in sample['subset']:
                reduced.append(sample)
        data = reduced

    if "truthfulqa" in run_name:
        match_truthfulqa(data)

    accuracy_kwargs = {
        'shots': args.shots,
        'cot': cot,
        'answer_type': answer_type
    }

    eval_method = accuracy_metric
    if args.sc_cot:
        eval_method = sc_cot_accuracy_metric
    elif answer_type == 'ner':
        eval_method = ner_metric

    metrics = eval_method(data, **accuracy_kwargs)
    display(
        metrics, run_name, args.benchmark,
        subset=None, verbose=args.verbose
    )

    if cot and len(metrics["ignored"]) > 0:
        save_dictlist_to_json(
            metrics["ignored"],
            f'{args.out_dir}/{args.benchmark}-{args.checkpoint}-ignored.json')

    if args.multi_seed or "mmlu_medical" in run_name or "truthfulqa" in run_name:
        subset_acc_dict =  sort_predictions(data, args.multi_seed, run_name)
        for subset in subset_acc_dict:
            subset_data = subset_acc_dict[subset]['data']
            metrics = eval_method(subset_data, **accuracy_kwargs)
            display(
                metrics, run_name, args.benchmark,
                subset=subset, verbose=args.verbose
            )

    if args.wandb:
        metrics["dataset"] = dataset,
        metrics["model"] = model,
        del metrics["ignored"]

        wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=run_name)
        artifact = wandb.Artifact(run_name, type="dataset", metadata=metrics)
        artifact.add_file(path)
        wandb.log_artifact(artifact)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, default='../benchmarks/generations',
                        help="The directory to save the generations")
    parser.add_argument('--benchmark', type=str, default='medqa',
                        help="The benchmark to evaluate on: [pubmedqa, medqa, medqa4, medmcqa, mmlu_medical, mmlu_general]")
    parser.add_argument('--checkpoint', type=str, default='replay-pubmedqa',
                        help="The checkpoint to evaluate on")
    parser.add_argument('--cot', action='store_true',
                        help="Whether chain-or-thought is used for inference")
    parser.add_argument('--shots', type=int, default=0,
                        help="Number of shots used for in-context learning")
    parser.add_argument('--multi_seed', action='store_true',
                        help="Whether multiple seeds are used for in-context learning")
    parser.add_argument('--sc_cot', action='store_true',
                        help="Whether self-consistency chain-or-thought is used for inference")
    parser.add_argument('--wandb', action='store_true',
                        help="Whether to log the results to wandb")
    # Wandb arguments
    parser.add_argument('--wandb_project', type=str, default='generations',
                        help="The project name for wandb")
    parser.add_argument('--wandb_entity', type=str, default='meditron',
                        help="The entity name for wandb")
    parser.add_argument('--verbose', action='store_true',
                        help="Whether to print detailed results")
    args = parser.parse_args()

    main(args)