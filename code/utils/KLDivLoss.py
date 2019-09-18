import torch
import torch.nn as nn
import torch.nn.functional as F


class KLDivLoss(nn.Module):
    r"""The `Kullback-Leibler divergence`_ Loss

    KL divergence is a useful distance measure for continuous distributions
    and is often useful when performing direct regression over the space of
    (discretely sampled) continuous output distributions.

    As with `NLLLoss`, the `input` given is expected to contain
    *log-probabilities*, however unlike `ClassNLLLoss`, `input` is not
    restricted to a 2D Tensor, because the criterion is applied element-wise.

    This criterion expects a `target` `Tensor` of the same size as the
    `input` `Tensor`.

    The loss can be described as:

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = y_n \odot \left( \log y_n - x_n \right),

    where :math:`N` is the batch size. If reduce is ``True``, then:

    .. math::
        \ell(x, y) = \begin{cases}
            \operatorname{mean}(L), & \text{if}\; \text{size_average} = \text{True},\\
            \operatorname{sum}(L),  & \text{if}\; \text{size_average} = \text{False}.
        \end{cases}

    By default, the losses are averaged for each minibatch over observations
    **as well as** over dimensions. However, if the field
    `size_average` is set to ``False``, the losses are instead summed.

    .. _Kullback-Leibler divergence:
        https://en.wikipedia.org/wiki/Kullback-Leibler_divergence

    Args:
        size_average (bool, optional: By default, the losses are averaged
            for each minibatch over observations **as well as** over
            dimensions. However, if ``False`` the losses are instead summed.
        reduce (bool, optional): By default, the losses are averaged
            over observations for each minibatch, or summed, depending on
            size_average. When reduce is ``False``, returns a loss per input/target
            element instead and ignores size_average. Default: ``True``

    Shape:
        - input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - target: :math:`(N, *)`, same shape as the input
        - output: scalar. If `reduce` is ``True``, then :math:`(N, *)`,
            same shape as the input

    """
    def __init__(self):
        super(KLDivLoss, self).__init__()
        self.size_average = True
        self.reduce = False

    def forward(self, input, target, lad):
        kl_divs = F.kl_div(torch.log(input), target, size_average=self.size_average, reduce=self.reduce)
        batch_size, class_num = kl_divs.shape
        max, index = torch.max(target, 1)
        weights = 1 + (1 - max) ** lad  # ** torch.abs(input[index] - target)
        kl_divs = torch.mean(torch.mean(kl_divs, 1) * weights)

        return kl_divs
