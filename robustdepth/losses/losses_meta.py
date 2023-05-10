from robustdepth.models.models_meta import StringEnum


class DepthLossType(StringEnum):
    MEAN_ABS_ERROR = "MeanAbsoluteError"
    MEAN_SQR_ERROR = "MeanSquaredError"
    SCALED_SI_ERROR = "ScaledSIError"
    BERHU = "BerHu"
    HUBER = "Huber"
    RUBER = "Ruber"
    FOSLL1_TRAPEZOIDAL = "FOSLL1Trapezoidal"
    EPS_SENS_L1 = "EpsSensL1"
    EPS_SENS_L2 = "EpsSensL2"
    WEIGHTED_L2 = "WeightedL2"
    BARRON = "Barron"
    TRIM = "Trimmed"


def get_loss_type_by_name(loss_name, case_sensitive=False):
    if not case_sensitive:
        loss_name = loss_name.lower()

    if loss_name == "l1":
        return DepthLossType.MEAN_ABS_ERROR
    elif loss_name == "l2":
        return DepthLossType.MEAN_SQR_ERROR
    elif loss_name == "scaled_si_error":
        return DepthLossType.SCALED_SI_ERROR
    elif loss_name == "berhu":
        return DepthLossType.BERHU
    elif loss_name == "huber":
        return DepthLossType.HUBER
    elif loss_name == "fosll1_trapezoidal":
        return DepthLossType.FOSLL1_TRAPEZOIDAL
    elif loss_name == "weighted_l2":
        return DepthLossType.WEIGHTED_L2
    elif loss_name == "barron":
        return DepthLossType.BARRON
    elif loss_name == "eps_sens_l1":
        return DepthLossType.EPS_SENS_L1
    elif loss_name == "eps_sens_l2":
        return DepthLossType.EPS_SENS_L2
    elif loss_name == "trim":
        return DepthLossType.TRIM
    elif loss_name == "ruber":
        return DepthLossType.RUBER
    else:
        raise ValueError("Unknown loss type: {}".format(loss_name))


def is_trapezoidal_fosl(loss_type):
    return loss_type in [DepthLossType.FOSLL1_TRAPEZOIDAL]


def is_si_error(loss_type):
    return loss_type in [DepthLossType.SCALED_SI_ERROR]


def is_eps_sens_error(loss_type):
    return loss_type in [DepthLossType.EPS_SENS_L1, DepthLossType.EPS_SENS_L2]
