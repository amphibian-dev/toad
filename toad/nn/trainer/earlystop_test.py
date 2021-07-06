import torch
from .earlystop import earlystopping



def test_earlystopping():
    model = torch.nn.Linear(10, 10)

    @earlystopping(delta = -1, patience = 3)
    def scoring(history):
        return history['loss']

    rounds = []
    for i in range(10):
        if scoring(model = model, history = {"loss": 1}):
            break

        rounds.append(i)
    
    assert len(rounds) == 3


def test_best_state():
    model = torch.nn.Linear(10, 1)

    @earlystopping(delta = -1, patience = 1)
    def scoring(history):
        return history['loss']
     
    with torch.no_grad():
        model.weight.fill_(1.)

    # save init weight
    scoring(model = model, history = {"loss": 10})
    assert scoring.best_state["weight"].sum().item() == 10

    # change weight
    with torch.no_grad():
        model.weight.fill_(0.)
    
    # save best weight
    scoring(model = model, history = {"loss": 5})
    assert scoring.best_state["weight"].sum().item() == 0

