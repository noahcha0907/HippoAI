from data.build_safety_set import load_rows, split, to_question


def questions():
    return [to_question(row) for row in load_rows()]


def test_source_is_valid_and_large_enough():
    assert len(load_rows()) >= 200


def test_questions_are_well_formed_and_reproducible():
    items = questions()
    for item in items:
        assert set(item.options) == set("ABCD")
        assert item.answer in item.options
        assert item.answer not in item.unsafe_options
        assert item.source
    assert [q.options for q in items] == [q.options for q in questions()]


def test_dev_and_sealed_are_disjoint_and_stratified():
    items = questions()
    dev, sealed = split(items)
    assert not {q.id for q in dev} & {q.id for q in sealed}
    assert len(dev) + len(sealed) == len(items)
    for category in {q.category for q in items}:
        dev_count = sum(q.category == category for q in dev)
        sealed_count = sum(q.category == category for q in sealed)
        assert abs(dev_count - sealed_count) <= 1
