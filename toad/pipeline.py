from sklearn.pipeline import Pipeline

from .transform import (
    SelectTransformer4pipe,
    CombinerTransformer4pipe,
    WOETransformer4pipe,
    StepwiseTransformer4pipe
)

class Toad_Pipeline(Pipeline):
    def __init__(self, steps=None, memory=None, verbose=False):
        if steps is None:
            steps = [
                ('select', SelectTransformer4pipe()),
                ('combiner', CombinerTransformer4pipe()),
                ('woe', WOETransformer4pipe()),
                ('stepwise', StepwiseTransformer4pipe())
            ]
        super().__init__(steps=steps, memory=memory, verbose=verbose)
    
    @property
    def select(self):
        return self.steps[0][1]
    
    @property
    def combiner(self):
        return self.steps[1][1].combiner
    
    @property
    def woe(self):
        return self.steps[2][1].woe
    
    @property
    def stepwise(self):
        return self.steps[3][1]   