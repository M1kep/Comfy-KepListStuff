from typing import Any, Dict, List, Tuple

from torch import Tensor

from custom_nodes.Comfy_KepListStuff.utils import AnyType

any_type = AnyType("*")


class ListLengthNode:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(self) -> Dict[str, Dict[str, Any]]:
        return {
            "required": {"In": (any_type, {})},
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("Length",)
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (False,)
    FUNCTION = "get_len"

    CATEGORY = "List Stuff"

    def get_len(self, In: List[Any]) -> Tuple[int]:
        return (len(In),)

class RepeatList:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(self) -> Dict[str, Dict[str, Any]]:
        return {
            "required": {
                "In": (any_type, {}),
                "Count": ("INT", {"default": 0, "min": 0, "max": 99999, "step": 1}),
            },
        }

    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("Extended",)
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "repeat_list"

    CATEGORY = "List Stuff"

    def repeat_list(self, In: List[Any], Count: List[int]) -> Tuple[List[Any]]:
        if len(Count) != 1:
            raise ValueError("Count does not support multiple values")
        return (In * Count[0],)

class JoinListAny:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(self) -> Dict[str, Dict[str, Any]]:
        return {
            "required": {
                "In1": (any_type, {}),
                "In2": (any_type, {}),
            },
            "optional": {
                "In3": (any_type, {}),
                "In4": (any_type, {}),
                "In5": (any_type, {}),
            },
        }

    RETURN_TYPES = (any_type, "INT")
    RETURN_NAMES = ("Joined", "Sizes")
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True, True)
    FUNCTION = "join_lists"

    CATEGORY = "List Stuff"


    def join_lists(
            self,
            *args: List[Tensor],
            **kwargs: List[Tensor],
    ) -> Tuple[List[Tensor], List[int]]:
        sizes = []
        joined = []
        for arg in args:
            sizes.append(len(arg))
            joined.extend(arg)
        for arg in kwargs.values():
            if arg is not None:
                sizes.append(len(arg))
                joined.extend(arg)

        return joined, sizes

class ReverseList:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(self) -> Dict[str, Dict[str, Any]]:
        return {
            "required": {"In": (any_type, {})},
        }

    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("Reversed",)
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "reverse_list"

    CATEGORY = "List Stuff"

    def reverse_list(self, In: List[Any]) -> Tuple[List[Any]]:
        return (In[::-1],)

class JoinFloatLists:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(self) -> Dict[str, Dict[str, Any]]:
        return {
            "required": {
                "In1": ("FLOAT", {"forceInput": True}),
                "In2": ("FLOAT", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("Joined",)
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "join_lists"

    CATEGORY = "List Stuff"

    def join_lists(self, In1: List[float], In2: List[float]) -> Tuple[List[float]]:
        return (In1 + In2,)

class JoinImageLists:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(self) -> Dict[str, Dict[str, Any]]:
        return {
            "required": {
                "In1": ("IMAGE",),
                "In2": ("IMAGE",),
            },
            "optional": {
                "In3": ("IMAGE",),
                "In4": ("IMAGE",),
                "In5": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("Joined", "Sizes")
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True, True)
    FUNCTION = "join_lists"

    CATEGORY = "List Stuff"

    def join_lists(
        self,
        *args: List[Tensor],
        **kwargs: List[Tensor],
    ) -> Tuple[List[Tensor], List[int]]:
        sizes = []
        joined = []
        for arg in args:
            sizes.append(len(arg))
            joined.extend(arg)
        for arg in kwargs.values():
            if arg is not None:
                sizes.append(len(arg))
                joined.extend(arg)

        return joined, sizes


class StringList:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(self) -> Dict[str, Dict[str, Any]]:
        return {
            "required": {
                "Text1": ("STRING", {}),
                "Text2": ("STRING", {}),
            },
            "optional": {
                "Text3": ("STRING", {}),
                "Text4": ("STRING", {}),
                "Text5": ("STRING", {}),
                "Text6": ("STRING", {}),
                "Text7": ("STRING", {}),
            },
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("Strings", "Num Strings")
    INPUT_IS_LIST = False
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "to_string_list"

    CATEGORY = "List Stuff"

    def to_string_list(
            self,
            *args: str,
            **kwargs: str,
    ) -> Tuple[List[str], List[int]]:
        ret = []
        for arg in args:
            ret.append(arg)
        for arg in kwargs.values():
            if arg != "":
                ret.append(arg)

        return ret, [len(ret)]

class StringListFromNewline:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(self) -> Dict[str, Dict[str, Any]]:
        return {
            "required": {
                "Text": ("STRING", {"multiline": True}),
            },
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("Strings", "Num Strings")
    INPUT_IS_LIST = False
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "to_string_list"

    CATEGORY = "List Stuff"

    def to_string_list(self, Text: str) -> Tuple[List[str], List[int]]:
        return Text.split("\n"), [len(Text.split("\n"))]
