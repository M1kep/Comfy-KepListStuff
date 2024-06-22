from .nodes.deprecated import StackImages
from .nodes.images import (
    ImageLabelOverlay,
    EmptyImages,
    XYImage,
    ImageListLoader,
    VariableImageBuilder,
)
from .nodes.list_utils import (
    ListLengthNode,
    JoinFloatLists,
    JoinImageLists,
    StringList,
    ReverseList,
    RepeatList,
    JoinListAny,
    StringListFromNewline,
)
from .nodes.range_nodes import (
    IntRangeNode,
    FloatRangeNode,
    IntNumStepsRangeNode,
    FloatNumStepsRangeNode,
)
from .nodes.xy import UnzippedProductAny

NODE_CLASS_MAPPINGS = {
    "Range(Step) - Int": IntRangeNode,
    "Range(Num Steps) - Int": IntNumStepsRangeNode,
    "Range(Step) - Float": FloatRangeNode,
    "Range(Num Steps) - Float": FloatNumStepsRangeNode,
    "List Length": ListLengthNode,
    "Image Overlay": ImageLabelOverlay,
    "Stack Images": StackImages,
    "Empty Images": EmptyImages,
    "Join Image Lists": JoinImageLists,
    "Join Float Lists": JoinFloatLists,
    "XYAny": UnzippedProductAny,
    "XYImage": XYImage,
    "ImageListLoader": ImageListLoader,
    "KepStringList": StringList,
    "KepStringListFromNewline": StringListFromNewline,
    "Kep_VariableImageBuilder": VariableImageBuilder,
    "Kep_ReverseList": ReverseList,
    "Kep_RepeatList": RepeatList,
    "Kep_JoinListAny": JoinListAny
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Stack Images": "Stack Images(Deprecated)",
    "ImageListLoader": "Image List Loader",
    "KepStringList": "String List",
    "Kep_VariableImageBuilder": "Variable Image Builder",
    "Kep_ReverseList": "Reverse List",
    "Kep_RepeatList": "Repeat List",
    "Kep_JoinListAny": "Join List Any",
    "KepStringListFromNewline": "String List From Newline",
}
