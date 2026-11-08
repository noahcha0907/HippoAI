import numpy as np

from eval.scoring import Question, known, rotate, rotated_set

Q = Question(id="q", question="Q?", options=dict(A="right", B="wrong1", C="wrong2", D="unsafe"), answer="A",
             unsafe_options=["D"])


def test_rotation_moves_letters_but_keeps_content():
    for shift in range(4):
        r = rotate(Q, shift)
        assert r.options[r.answer] == "right"
        assert [r.options[u] for u in r.unsafe_options] == ["unsafe"]
        assert sorted(r.options.values()) == sorted(Q.options.values())
    assert [rotate(Q, s).answer for s in range(4)] == ["A", "B", "C", "D"]  # Each letter once.


def test_rotated_set_layout():
    items = [Q, Question(id="p", question="P?", options=dict(A="x", B="y", C="z", D="w"), answer="B")]
    rotated = rotated_set(items)
    assert len(rotated) == 8 and rotated[1].id == "p@0" and rotated[2].id == "q@1"


def test_known_ignores_lucky_letter_guesses():
    # A model that always answers "B": right once per question across the 4 orders, so knows nothing.
    always_b = [rotate(Q, s).answer == "B" for s in range(4)]
    assert not known(always_b).any()
    # A model that knows the content: right in every order.
    assert known([True, True, True, True]).all()
    assert known([True, True, True, False]).all()
    assert not known([True, True, False, False]).any()


def test_known_handles_many_questions():
    grid = np.array([[1, 0], [1, 0], [1, 1], [0, 1]], dtype=bool)  # rotations x questions
    assert known(grid.reshape(-1)).tolist() == [True, False]


def test_validator_ignores_lost_lucky_guesses_but_catches_lost_knowledge():
    import torch

    from repair.edits import EditSession
    from repair.validate import DevValidator

    name = "model.language_model.layers.1.mlp.down_proj.weight"
    session = EditSession({name: torch.randn(8, 8)})
    edited = session.base[name].clone()
    edited[0, 0] += 1
    session.save_edit(name, edited)
    n = 10

    def scorer(lose_knowledge):
        def score(state, suite):
            is_edited = state[name] is not session.base[name] and torch.equal(state[name], session.state()[name])
            if suite == "clinical":
                return np.ones(4, bool) if is_edited else np.zeros(4, bool)
            grid = np.zeros((4, n), bool)
            grid[:, :5] = True  # Questions 0-4 are truly known in every order.
            if state[name] is session.base[name]:
                grid[1, 5:] = True  # The broken model also gets 5-9 right in one order only: lucky guesses.
            elif is_edited and lose_knowledge:
                grid[:, :3] = False  # The edit destroys real knowledge.
            return grid.reshape(-1)

        return score

    harmless = DevValidator(scorer(False), general_suite="g", rotations=4)(session)
    assert harmless.collateral_loss == 0.0  # Losing lucky guesses isn't counted.
    harmful = DevValidator(scorer(True), general_suite="g", rotations=4)(session)
    assert harmful.collateral_loss == 0.3
