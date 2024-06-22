from _decimal import Context, getcontext
from decimal import Decimal
from typing import Iterator, List, Tuple, Dict, Any, Union, Optional

import numpy as np

from ..utils import (
    error_if_mismatched_list_args,
    zip_with_fill,
)

custom_context = Context(prec=8)


class IntRangeNode:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(s) -> Dict[str, Dict[str, Any]]:
        return {
            "required": {
                "start": ("INT", {"default": 0, "min": -4096, "max": 4096, "step": 1}),
                "stop": ("INT", {"default": 0, "min": -4096, "max": 4096, "step": 1}),
                "step": ("INT", {"default": 0, "min": -4096, "max": 4096, "step": 1}),
                "end_mode": (["Inclusive", "Exclusive"], {"default": "Inclusive"}),
            },
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("range", "range_sizes")
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True, True)
    FUNCTION = "build_range"

    CATEGORY = "List Stuff"

    def build_range(
        self, start: List[int], stop: List[int], step: List[int], end_mode: List[str]
    ) -> Tuple[List[int], List[int]]:
        error_if_mismatched_list_args(locals())

        ranges = []
        range_sizes = []
        for e_start, e_stop, e_step, e_end_mode in zip_with_fill(
            start, stop, step, end_mode
        ):
            if e_end_mode == "Inclusive":
                e_stop += 1
            vals = list(range(e_start, e_stop, e_step))
            ranges.extend(vals)
            range_sizes.append(len(vals))

        return ranges, range_sizes


class IntNumStepsRangeNode:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(s) -> Dict[str, Dict[str, Any]]:
        return {
            "required": {
                "start": ("INT", {"default": 0, "min": -4096, "max": 4096, "step": 1}),
                "stop": ("INT", {"default": 0, "min": -4096, "max": 4096, "step": 1}),
                "num_steps": (
                    "INT",
                    {"default": 0, "min": -4096, "max": 4096, "step": 1},
                ),
                "end_mode": (["Inclusive", "Exclusive"], {"default": "Inclusive"}),
                "allow_uneven_steps": (["True", "False"], {"default": "False"}),
            }
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("range", "range_sizes")
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True, True)
    FUNCTION = "build_range"

    CATEGORY = "List Stuff"

    def build_range(
        self,
        start: List[int],
        stop: List[int],
        num_steps: List[int],
        end_mode: List[str],
        allow_uneven_steps: List[str],
    ) -> Tuple[List[int], List[int]]:
        if len(allow_uneven_steps) > 1:
            raise Exception("List input for allow_uneven_steps is not supported.")

        error_if_mismatched_list_args(locals())

        ranges = []
        range_sizes = []
        for e_start, e_stop, e_num_steps, e_end_mode in zip_with_fill(
            start, stop, num_steps, end_mode
        ):
            direction = 1 if e_stop > e_start else -1
            if e_end_mode == "Exclusive":
                e_stop -= direction

            # Check for uneven steps
            step_size = (e_stop - e_start) / (e_num_steps - 1)
            if not allow_uneven_steps[0] == "True" and step_size != int(step_size):
                raise ValueError(
                    f"Uneven steps detected for start={e_start}, stop={e_stop}, num_steps={e_num_steps}."
                )

            vals = (
                np.rint(np.linspace(e_start, e_stop, e_num_steps)).astype(int).tolist()
            )
            ranges.extend(vals)
            range_sizes.append(len(vals))

        return ranges, range_sizes


class FloatRangeNode:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(s) -> Dict[str, Dict[str, Any]]:
        return {
            "required": {
                "start": (
                    "FLOAT",
                    {"default": 0, "min": -4096, "max": 4096, "step": 1},
                ),
                "stop": ("FLOAT", {"default": 0, "min": -4096, "max": 4096, "step": 1}),
                "step": ("FLOAT", {"default": 0, "min": -4096, "max": 4096, "step": 1}),
                "end_mode": (["Inclusive", "Exclusive"], {"default": "Inclusive"}),
            },
        }

    RETURN_TYPES = ("FLOAT", "INT")
    RETURN_NAMES = ("range", "range_sizes")
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True, True)
    FUNCTION = "build_range"

    CATEGORY = "List Stuff"

    @staticmethod
    def _decimal_range(
        start: Decimal, stop: Decimal, step: Decimal, inclusive: bool
    ) -> Iterator[float]:
        ret_val = start
        if inclusive:
            stop = stop + step

        direction = 1 if step > 0 else -1
        # while ret_val < stop:
        #     yield float(ret_val)
        #     ret_val += step
        while (ret_val - stop) * direction < 0:
            yield float(ret_val)
            ret_val += step

    def build_range(
        self,
        start: List[Union[float, Decimal]],
        stop: List[Union[float, Decimal]],
        step: List[Union[float, Decimal]],
        end_mode: List[str],
    ) -> Tuple[List[float], List[int]]:
        error_if_mismatched_list_args(locals())
        getcontext().prec = 12

        start = [Decimal(s) for s in start]
        stop = [Decimal(s) for s in stop]
        step = [Decimal(s) for s in step]

        ranges = []
        range_sizes = []
        for e_start, e_stop, e_step, e_end_mode in zip_with_fill(
            start, stop, step, end_mode
        ):
            vals = list(
                self._decimal_range(e_start, e_stop, e_step, e_end_mode == "Inclusive")
            )
            ranges.extend(vals)
            range_sizes.append(len(vals))

        return ranges, range_sizes


class FloatNumStepsRangeNode:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(s) -> Dict[str, Dict[str, Any]]:
        return {
            "required": {
                "start": (
                    "FLOAT",
                    {"default": 0, "min": -4096, "max": 4096, "step": 1},
                ),
                "stop": ("FLOAT", {"default": 0, "min": -4096, "max": 4096, "step": 1}),
                "num_steps": ("INT", {"default": 1, "min": 1, "max": 4096, "step": 1}),
            },
        }

    RETURN_TYPES = ("FLOAT", "INT")
    RETURN_NAMES = ("range", "range_sizes")
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True, True)
    FUNCTION = "build_range"

    CATEGORY = "List Stuff"

    @staticmethod
    def _decimal_range(
        start: Decimal, stop: Decimal, num_steps: int
    ) -> Iterator[float]:
        step = (stop - start) / (num_steps - 1)
        direction = 1 if step > 0 else -1

        ret_val = start
        for _ in range(num_steps):
            if (
                ret_val - stop
            ) * direction > 0:  # Ensure we don't exceed the 'stop' value
                break
            yield float(ret_val)
            ret_val += step

    def build_range(
        self,
        start: List[Union[float, Decimal]],
        stop: List[Union[float, Decimal]],
        num_steps: List[int],
    ) -> Tuple[List[float], List[int]]:
        error_if_mismatched_list_args(locals())
        getcontext().prec = 12

        start = [Decimal(s) for s in start]
        stop = [Decimal(s) for s in stop]

        ranges = []
        range_sizes = []
        for e_start, e_stop, e_num_steps in zip_with_fill(start, stop, num_steps):
            vals = list(self._decimal_range(e_start, e_stop, e_num_steps))
            ranges.extend(vals)
            range_sizes.append(len(vals))

        return ranges, range_sizes
