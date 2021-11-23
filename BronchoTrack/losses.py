import torch
from torch import nn
import pyquaternion as pyq
from .metrics import DirectionError


class CosLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, labels):
        return 1 - torch.mean(torch.cos(x-labels))


class DirectionLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, labels):
        return torch.mean(DirectionLoss.inverse_cos(x, labels))

    @staticmethod
    def inverse_cos(preds, targets):
        return torch.acos(
            torch.sum(
                DirectionLoss.direction_vec(preds) *
                DirectionLoss.direction_vec(targets), dim=-1))

    @staticmethod
    def direction_vec(preds):
        result = torch.zeros_like(preds).type_as(preds)
        result[..., 0] = torch.cos(preds[:, :, 1])*torch.cos(preds[:, :, 2])
        result[..., 1] = torch.sin(preds[:, :, 0])*torch.sin(preds[:, :, 1])*torch.cos(preds[:, :, 2]) + torch.cos(preds[:, :, 0])*torch.sin(preds[:, :, 2])
        result[..., 2] = -torch.cos(preds[:, :, 0])*torch.sin(preds[:, :, 1])*torch.cos(preds[:, :, 2]) + torch.sin(preds[:, :, 0])*torch.sin(preds[:, :, 2])
        return result


def _sum_of_squares(q):
    return torch.sum(q*q, axis=-1)


def _q_conjugate(q):
    """
    Returns the complex conjugate of the input quaternion tensor of shape [*, 4].
    """
    assert q.shape[-1] == 4

    conj = torch.tensor([1, -1, -1, -1]).type_as(q)  # multiplication coefficients per element
    return q * conj.expand_as(q)


def _q_inverse(q):
    """Inverse of the quaternion object, encapsulated in a new instance.

    For a unit quaternion, this is the inverse rotation, i.e. when combined with the original rotation, will result in the null rotation.

    Returns:
        A new Quaternion object representing the inverse of this object
    """
    ss = _sum_of_squares(q)
    return _q_conjugate(q) / torch.stack([ss, ss, ss, ss], dim=-1)


def _q_norm(q):
    """L2 norm of the quaternion 4-vector.

    This should be 1.0 for a unit quaternion (versor)
    Slow but accurate. If speed is a concern, consider using _fast_normalise() instead

    Returns:
        A scalar real number representing the square root of the sum of the squares of the elements of the quaternion.
    """
    mag_squared = _sum_of_squares(q)
    return torch.sqrt(mag_squared)


def _q_normalize(q):
    q_n = _q_norm(q)
    return q / torch.stack([q_n, q_n, q_n, q_n], dim=-1)


def _q_log(q):
    """Quaternion Logarithm.

    Find the logarithm of a quaternion amount.

    Params:
            q: the input quaternion/argument as a Quaternion object.

    Returns:
            A quaternion amount representing log(q) := (log(|q|), v/|v|acos(w/|q|)).

    Note:
        The method computes the logarithm of general quaternions. See [Source](https://math.stackexchange.com/questions/2552/the-logarithm-of-quaternion/2554#2554) for more details.
    """
    v_norm = torch.linalg.norm(q[..., 1:])
    q_norm = _q_norm(q)
    vec = q[..., 1:] / v_norm
    new_q = torch.zeros_like(q).type_as(q)
    new_q[..., 0] = torch.log(q_norm)
    angles = torch.acos(q[..., 0]/q_norm)
    new_q[..., 1:] = torch.stack([angles, angles, angles], dim=-1)*vec
    return new_q


class QuaternionDistanceLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, labels):
        return torch.mean(
            QuaternionDistanceLoss.quaternion_distance(
                QuaternionDistanceLoss.euler2q(x),
                QuaternionDistanceLoss.euler2q(labels)
            )
        )

    @staticmethod
    def euler2q(vec):
        result = torch.zeros((*vec.shape[:2], 4)).type_as(vec)
        result[..., 0] = -torch.sin(vec[..., 0]/2)*torch.sin(vec[..., 1]/2)*torch.sin(vec[..., 2]/2) + \
            torch.cos(vec[..., 0]/2)*torch.cos(vec[..., 1]/2)*torch.cos(vec[..., 2]/2)
        result[..., 1] = torch.sin(vec[..., 0]/2)*torch.cos(vec[..., 1]/2)*torch.cos(vec[..., 2]/2) + \
            torch.cos(vec[..., 0]/2)*torch.sin(vec[..., 1]/2)*torch.sin(vec[..., 2]/2)
        result[..., 2] = -torch.sin(vec[..., 0]/2)*torch.cos(vec[..., 1]/2)*torch.sin(vec[..., 2]/2) + \
            torch.cos(vec[..., 0]/2)*torch.sin(vec[..., 1]/2)*torch.cos(vec[..., 2]/2)
        result[..., 3] = torch.sin(vec[..., 0]/2)*torch.sin(vec[..., 1]/2)*torch.cos(vec[..., 2]/2) + \
            torch.cos(vec[..., 0]/2)*torch.cos(vec[..., 1]/2)*torch.sin(vec[..., 2]/2)
        return result

    @staticmethod
    def quaternion_distance(q, p):
        q_n, p_n = _q_normalize(q), _q_normalize(p)
        q_pred = _q_inverse(q_n)*p_n
        q_pred = _q_log(q_pred)
        return _q_norm(q_pred)
