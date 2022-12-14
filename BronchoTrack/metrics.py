import torch
from torchmetrics import Metric
from .losses import QuaternionDistanceLoss


# TODO: review metrics with Carles
def direction_vec(preds, axis=0):
    # batch, step, pose
    # preds[0]==Rx==theta1==roll, preds[1]==Ry==theta2==pitch, preds[2]==Rz==theta3
    # transform from vector (1, 0, 0)
    if axis==0:
        result = torch.tensor([
            torch.cos(preds[1])*torch.cos(preds[2]),
            torch.sin(preds[0])*torch.sin(preds[1])*torch.cos(preds[2]) +
            torch.cos(preds[0])*torch.sin(preds[2]),
            -torch.cos(preds[0])*torch.sin(preds[1])*torch.cos(preds[2]) +
            torch.sin(preds[0])*torch.sin(preds[2])
        ]).type_as(preds)
    elif axis==1:
        result = torch.tensor([
            -torch.cos(preds[1])*torch.sin(preds[2]),
            -torch.sin(preds[0])*torch.sin(preds[1])*torch.sin(preds[2]) +
            torch.cos(preds[0])*torch.cos(preds[2]),
            torch.cos(preds[0])*torch.sin(preds[1])*torch.sin(preds[2]) +
            torch.sin(preds[0])*torch.cos(preds[2])
        ]).type_as(preds)
    elif axis==2:
        result = torch.tensor([
            torch.sin(preds[1]),
            -torch.sin(preds[0])*torch.cos(preds[1]),
            torch.cos(preds[0])*torch.cos(preds[1])
        ]).type_as(preds)  
    return result


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
        self.squared_sum += self.euclidean(preds, targets)
        self.count += 1

    def compute(self):
        return self.squared_sum/self.count

    @staticmethod
    def euclidean(preds, targets):
        return torch.sqrt(torch.sum((preds - targets)**2))


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
        self.squared_sum += self.inverse_cos(preds, targets)
        self.count += 1

    def compute(self):
        return self.squared_sum/self.count

    @staticmethod
    def inverse_cos(preds, targets, axis=0):
        return torch.acos(
            torch.dot(direction_vec(preds, axis=axis), direction_vec(targets, axis=axis)))


class NeedleError(Metric):
    """Needle error metric, works on 1D vectors. MEtric reproduced according to
    https://doi.org/10.1109/TMI.2013.2252361. Section III A
    Works with both components, angular and positional. First 3 values are
    positional and second 3 angular
    """

    def __init__(self, distance=1):
        super().__init__()
        self.add_state("squared_sum", torch.tensor(0, dtype=torch.float32),
                       dist_reduce_fx="sum")
        self.add_state("count", torch.tensor(0), dist_reduce_fx="sum")
        self.distance = distance

    def update(self, preds, targets):
        self.squared_sum += self.needle(preds, targets, self.distance)
        self.count += 1

    def compute(self):
        return self.squared_sum/self.count

    @staticmethod
    def needle(preds, targets, distance=3):
        return torch.sqrt(torch.sum((
            preds[:3] + distance*direction_vec(preds[3:]) -
            (targets[:3] + distance*direction_vec(targets[3:])))**2))


class CosMetric(Metric):

    def __init__(self, indiv=None):
        super().__init__()
        self.add_state("squared_sum", torch.tensor(0, dtype=torch.float32),
                       dist_reduce_fx="sum")
        self.add_state("count", torch.tensor(0), dist_reduce_fx="sum")
        self.indiv = indiv

    def update(self, preds, targets):
        if self.indiv:
            self.squared_sum += torch.mean(1 - torch.cos(preds[self.indiv] - targets[self.indiv]))
        else:
            self.squared_sum += torch.mean(1 - torch.cos(preds - targets))
        self.count += 1

    def compute(self):
        return self.squared_sum/self.count


class QuatMetric(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("squared_sum", torch.tensor(0, dtype=torch.float32),
                       dist_reduce_fx="sum")
        self.add_state("count", torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds, targets):
        self.squared_sum += torch.mean(
            QuaternionDistanceLoss.quaternion_distance(
                QuaternionDistanceLoss.euler2q(preds),
                QuaternionDistanceLoss.euler2q(targets)
            )
        )
        self.count += 1

    def compute(self):
        return self.squared_sum/self.count


class MSE(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("squared_sum", torch.tensor(0, dtype=torch.float32),
                       dist_reduce_fx="sum")
        self.add_state("count", torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds, targets):
        self.squared_sum += torch.mean((preds-targets)**2)
        self.count += 1

    def compute(self):
        return self.squared_sum/self.count


