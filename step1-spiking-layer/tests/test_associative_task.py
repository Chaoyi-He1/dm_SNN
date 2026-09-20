"""Verify that the learning probe cannot read the target from its query."""
import torch
from experiments.associative_recall import dataset


def assignments(x):
    result = set()
    for episode in x:
        values = [None] * 3
        for token in episode[:3]:
            values[token[:3].argmax().item()] = token[3:7].argmax().item()
        result.add(tuple(values))
    return result


def test_associative_split_is_assignment_disjoint():
    train, _, _ = dataset(False)
    test, _, _ = dataset(True)
    assert len(train) == 864 and len(test) == 288
    train_values, test_values = assignments(train), assignments(test)
    assert len(train_values) == 48 and len(test_values) == 16
    assert train_values.isdisjoint(test_values)


def test_associative_queries_hide_values_and_labels_are_conditionally_balanced():
    for heldout in (False, True):
        x, labels, positions = dataset(heldout)
        assert (x[:, 3] == 0).all()
        assert (x[:, -1, 3:8] == 0).all()
        for query in range(3):
            for position in range(3):
                mask = (x[:, -1, query] == 2) & (positions == position)
                counts = torch.bincount(labels[mask], minlength=4)
                assert (counts == counts[0]).all() and counts[0] > 0
                rows = torch.arange(len(x))[mask]
                assert torch.equal(x[rows, position, 3:7].argmax(-1), labels[mask])
