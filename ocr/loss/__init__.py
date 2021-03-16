from .label_smoothed_ce import LabelSmoothedCE


def pytorch_loss_forward_wrapper(real_forward_func):
    def wrap(preds, targets, lengths):
        logit_shape = preds.size(2)
        preds = preds.reshape(-1, logit_shape)
        targets = targets.reshape(-1)
        return real_forward_func(preds, targets)

    return wrap


def pytorch_inject_loss(loss_module):
    loss_module.forward = pytorch_loss_forward_wrapper(loss_module.forward)
    return loss_module
