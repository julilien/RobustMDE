from robustdepth.models.models_meta import StringEnum


class Dataset(StringEnum):
    NYU = "NYU"
    IBIMS = "IBIMS"
    DIODE = "DIODE"
    SUNRGBD = "SunRGBD"


def get_dataset_type_by_name(dataset_name):
    dataset_name = dataset_name.lower()
    if dataset_name == Dataset.NYU.value.lower():
        return Dataset.NYU
    elif dataset_name == Dataset.IBIMS.value.lower():
        return Dataset.IBIMS
    elif dataset_name == Dataset.DIODE.value.lower():
        return Dataset.DIODE
    elif dataset_name == Dataset.SUNRGBD.value.lower():
        return Dataset.SUNRGBD
    else:
        raise ValueError("Unknown dataset name: {}".format(dataset_name))
