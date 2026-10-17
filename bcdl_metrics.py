"""
Metrics for bcdl-identity: accuracy, precision, recall, F1, confusion matrix, loss.

Everything is computed from one table, the confusion matrix:

                    predicted
                    Pacino  De Niro
    true  Pacino       5       1        row    = all images that really are this person
          De Niro      2       4        column = all images the model called this person
                                        diagonal = correct answers

Used by the training script (validation after each epoch) and for a final
evaluation of a saved checkpoint.
"""

import torch


# --- Step 1: run the model over a loader and fill the confusion matrix --------

@torch.no_grad()                                   # no gradients needed, saves memory and time
def evaluate(model, loader, num_classes, device='cpu', criterion=None):
    model.eval()                                   # Dropout off, BatchNorm fixed
    cm = [[0] * num_classes for _ in range(num_classes)]
    total_loss, n = 0.0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)                             # [batch, num_classes] scores
        if criterion is not None:
            total_loss += criterion(out, y).item() * y.size(0)   # loss is a batch mean -> back to a sum
        pred = out.argmax(dim=1)                   # highest score -> predicted class index
        for t, p in zip(y.tolist(), pred.tolist()):
            cm[t][p] += 1
        n += y.size(0)

    m = metrics_from_confusion(cm)
    m['loss'] = total_loss / n if criterion is not None else None
    return m


# --- Step 2: confusion matrix -> numbers -------------------------------------

def _div(a, b):
    return a / b if b else 0.0                     # 0 instead of a crash when a class is never predicted


def metrics_from_confusion(cm):
    k = len(cm)
    total = sum(sum(row) for row in cm)
    correct = sum(cm[i][i] for i in range(k))

    per_class = []
    for i in range(k):
        tp = cm[i][i]                              # correctly recognised as person i
        actual = sum(cm[i])                        # row sum: images that really are person i
        predicted = sum(cm[r][i] for r in range(k))  # column sum: images the model called person i
        precision = _div(tp, predicted)            # "when it says i, how often is it right?"
        recall = _div(tp, actual)                  # "of all images of i, how many did it find?"
        f1 = _div(2 * precision * recall, precision + recall)
        per_class.append({'precision': precision, 'recall': recall, 'f1': f1, 'support': actual})

    return {
        'accuracy': _div(correct, total),
        'correct': correct,
        'total': total,
        'per_class': per_class,
        'macro_precision': sum(c['precision'] for c in per_class) / k,
        'macro_recall': sum(c['recall'] for c in per_class) / k,
        'macro_f1': sum(c['f1'] for c in per_class) / k,
        'confusion': cm,
    }


# --- Step 3: readable report --------------------------------------------------

def print_report(m, classes):
    w = max(len(c) for c in classes) + 2

    print('Confusion matrix (rows = true, columns = predicted)')
    print(' ' * w + ''.join(c.rjust(w) for c in classes))
    for name, row in zip(classes, m['confusion']):
        print(name.ljust(w) + ''.join(str(v).rjust(w) for v in row))

    print()
    print('class'.ljust(w) + 'precision  recall      f1  support')
    for name, c in zip(classes, m['per_class']):
        print(f"{name.ljust(w)}{c['precision']:9.3f}{c['recall']:8.3f}{c['f1']:8.3f}{c['support']:9d}")
    print(f"{'macro avg'.ljust(w)}{m['macro_precision']:9.3f}{m['macro_recall']:8.3f}{m['macro_f1']:8.3f}{m['total']:9d}")

    print()
    print(f"accuracy: {m['accuracy']:.3f}  ({m['correct']}/{m['total']})")
    if m.get('loss') is not None:
        print(f"loss:     {m['loss']:.4f}")


if __name__ == '__main__':
    # Self-test on a hand-made confusion matrix (no model, no images needed).
    classes = ('Al Pacino', 'Robert De Niro')
    m = metrics_from_confusion([[5, 1],
                                [2, 4]])
    print_report(m, classes)
    # Expected: accuracy 0.750 (9/12)
    #           Al Pacino       precision 5/7 = 0.714  recall 5/6 = 0.833
    #           Robert De Niro  precision 4/5 = 0.800  recall 4/6 = 0.667
