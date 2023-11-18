from math import pi

import torch


def acasxu_property_bounds(property_str: str) -> tuple:
    if property_str == "2":
        init_lb = [55947.691, -pi, -pi, 1145, 0]
        init_ub = [60760, pi, pi, 1200, 60]
        classifier_fn, expected = torch.argmin, [
            torch.tensor(i, dtype=torch.long) for i in [1, 2, 3, 4]
        ]
    elif property_str == "3":
        init_lb = [1500, -0.06, 3.1, 980, 960]
        init_ub = [1800, 0.06, pi, 1200, 1200]
        classifier_fn, expected = torch.argmin, [
            torch.tensor(i, dtype=torch.long) for i in [1, 2, 3, 4]
        ]
    elif property_str == "4":
        init_lb = [1500, -0.06, 0, 1000, 700]
        init_ub = [1800, 0.06, 0, 1200, 800]
        classifier_fn, expected = torch.argmin, [
            torch.tensor(i, dtype=torch.long) for i in [1, 2, 3, 4]
        ]
    elif property_str == "5":
        init_lb = [250, 0.2, -3.141592, 100, 0]
        init_ub = [400, 0.4, -3.141592 + 0.005, 400, 400]
        classifier_fn, expected = torch.argmin, [
            torch.tensor(i, dtype=torch.long) for i in [4]
        ]
    elif property_str == "7":
        init_lb = [0, -3.141592, -3.141592, 100, 0]
        init_ub = [60760, 3.141592, 3.141592, 1200, 1200]
        classifier_fn, expected = torch.argmin, [
            torch.tensor(i, dtype=torch.long) for i in [0, 1, 2]
        ]
    elif property_str == "8":
        init_lb = [0, -3.141592, -0.1, 600, 600]
        init_ub = [60760, -0.75 * 3.141592, 0.1, 1200, 1200]
        classifier_fn, expected = torch.argmin, [
            torch.tensor(i, dtype=torch.long) for i in [0, 1]
        ]
    elif property_str == "9":
        init_lb = [2000, -0.4, -3.141592, 100, 0]
        init_ub = [7000, -0.14, -3.141592 + 0.01, 150, 150]
        classifier_fn, expected = torch.argmin, [
            torch.tensor(i, dtype=torch.long) for i in [3]
        ]
    elif property_str == "10":
        init_lb = [36000, 0.7, -3.141592, 900, 600]
        init_ub = [60760, 3.141592, -3.141592 + 0.01, 1200, 1200]
        classifier_fn, expected = torch.argmin, [
            torch.tensor(i, dtype=torch.long) for i in [0]
        ]
    else:
        raise RuntimeError(f"unsupported property string: {property_str}")
    init_lb = torch.tensor(init_lb, dtype=torch.float32)
    init_ub = torch.tensor(init_ub, dtype=torch.float32)
    means_for_scaling = torch.tensor(
        [19791.091, 0.0, 0.0, 650.0, 600.0], dtype=torch.float32
    )
    range_for_scaling = torch.tensor(
        [60261.0, 6.28318530718, 6.28318530718, 1100.0, 1200.0], dtype=torch.float32
    )

    init_lb = (init_lb - means_for_scaling) / range_for_scaling
    init_ub = (init_ub - means_for_scaling) / range_for_scaling

    return init_lb, init_ub


def acasxu_generate():
    torch.manual_seed(7777)
    data = []

    def generate(prop):
        for i in range(10):
            r1, r2 = acasxu_property_bounds(property_str=str(prop))
            x_t = torch.rand((1, 1, 1, 5)).float()
            x_t = (r1 - r2) * x_t + r2
            yield x_t

    for prop in [2, 3, 4, 5, 7, 8, 9, 10]:
        data.extend(list(generate(prop)))
    return data


def acasxu_dataloaders():
    train_data = acasxu_generate()
    loaders = {
        "train": train_data,
        "test": train_data,
    }
    return loaders
