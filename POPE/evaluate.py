import json
import argparse

def load_data(ans_file, label_file):
    answers = [json.loads(q) for q in open(ans_file, 'r')]
    label_list = [json.loads(q)['label'] for q in open(label_file, 'r')]
    return answers, label_list

def process_answers(answers):
    for answer in answers:
        text = answer['answer']
        if text.find('.') != -1:
            text = text.split('.')[0]
        text = text.replace(',', '')
        words = text.split(' ')
        if 'No' in words or 'not' in words or 'no' in words:
            answer['answer'] = 'no'
        else:
            answer['answer'] = 'yes'
    return answers

def calculate_metrics(answers, label_list):
    pred_list = [0 if answer['answer'] == 'no' else 1 for answer in answers]

    pos = 1
    neg = 0
    yes_ratio = pred_list.count(1) / len(pred_list)

    TP, TN, FP, FN = 0, 0, 0, 0
    for pred, label in zip(pred_list, label_list):
        if pred == pos and label == pos:
            TP += 1
        elif pred == pos and label == neg:
            FP += 1
        elif pred == neg and label == neg:
            TN += 1
        elif pred == neg and label == pos:
            FN += 1

    metrics = {
        'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN,
        'Accuracy': (TP + TN) / (TP + TN + FP + FN),
        'Precision': float(TP) / float(TP + FP) if TP + FP > 0 else 0,
        'Recall': float(TP) / float(TP + FN) if TP + FN > 0 else 0,
        'F1': 0,
        'Yes_ratio': yes_ratio
    }

    if metrics['Precision'] + metrics['Recall'] > 0:
        metrics['F1'] = 2 * metrics['Precision'] * metrics['Recall'] / (metrics['Precision'] + metrics['Recall'])
    
    return metrics

def print_metrics(metrics):
    print('TP\tFP\tTN\tFN\t')
    print('{}\t{}\t{}\t{}'.format(metrics['TP'], metrics['FP'], metrics['TN'], metrics['FN']))
    print('Accuracy: {}'.format(metrics['Accuracy']))
    print('Precision: {}'.format(metrics['Precision']))
    print('Recall: {}'.format(metrics['Recall']))
    print('F1 score: {}'.format(metrics['F1']))
    print('Yes ratio: {}'.format(metrics['Yes_ratio']))

def main():
    parser = argparse.ArgumentParser(description='Evaluate the answers based on provided labels.')
    parser.add_argument('--ans_file', type=str, required=True, help='Path to the answer JSON file')
    parser.add_argument('--label_file', type=str, required=True, help='Path to the label JSON file')
    args = parser.parse_args()

    answers, label_list = load_data(args.ans_file, args.label_file)
    answers = process_answers(answers)
    metrics = calculate_metrics(answers, label_list)
    print_metrics(metrics)

if __name__ == '__main__':
    main()
