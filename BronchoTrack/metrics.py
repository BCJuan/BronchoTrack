import torch
from torchmetrics import Metric

# TODO: review metrics with Carles


class EuclideanDistance(Metric):
    """Euclidean distance metric works on 1D vectors
    to obtain position error.
    MEtric reproduced according to
    Merritt, Scott A., Rahul Khare, Rebecca Bascom, and William E. Higgins.
    “Interactive CT-Video Registration for the Continuous Guidance
    of Bronchoscopy.”
    IEEE Transactions on Medical Imaging 32, no. 8 (August 2013): 1376–96.
    https://doi.org/10.1109/TMI.2013.2252361. Section III A
    """

    def __init__(self):
        super().__init__()
        self.add_state("squared_sum", torch.tensor(0, dtype=torch.float32),
                       dist_reduce_fx="sum")
        self.add_state("count", torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds, targets):
        self.squared_sum += torch.sqrt(torch.sum((preds - targets)**2))
        self.count += 1

    def compute(self):
        return self.squared_sum/self.count


class DirectionError(Metric):
    """Direction error metric, works on 1D vectors. MEtric reproduced according to
    https://doi.org/10.1109/TMI.2013.2252361. Section III A
    """
    # NOTE: is in degrees the result or what? What are our original units?

    def __init__(self):
        super().__init__()
        self.add_state("squared_sum", torch.tensor(0, dtype=torch.float32),
                       dist_reduce_fx="sum")
        self.add_state("count", torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds, targets):
        self.squared_sum += torch.nan_to_num(torch.acos(
            torch.dot(preds, targets)))
        self.count += 1

    def compute(self):
        return self.squared_sum/self.count


class NeedleError(Metric):
    """Needle error metric, works on 1D vectors. MEtric reproduced according to
    https://doi.org/10.1109/TMI.2013.2252361. Section III A
    Works with both components, angular and positional. First 3 values are
    positional and second 3 angular
    """

    def __init__(self, distance=3):
        super().__init__()
        self.add_state("squared_sum", torch.tensor(0, dtype=torch.float32),
                       dist_reduce_fx="sum")
        self.add_state("count", torch.tensor(0), dist_reduce_fx="sum")
        self.distance = distance

    def update(self, preds, targets):
        self.squared_sum += torch.sqrt(torch.sum((
            preds[:3] + self.distance*preds[3:] -
            (targets[:3] + self.distance*targets[3:]))**2))
        self.count += 1

    def compute(self):
        return self.squared_sum/self.count
