from .Donut_Detailer import NODE_CLASS_MAPPINGS as m1
from .Donut_Detailer_2 import NODE_CLASS_MAPPINGS as m2
from .Donut_Detailer_4 import NODE_CLASS_MAPPINGS as m3
from .Donut_Detailer_6 import NODE_CLASS_MAPPINGS as m4
from .DonutDetailerXLBlocks import NODE_CLASS_MAPPINGS as m5
>>>>>>> dada855 (Add Donut Detailer XL Blocks node and register in __init__.py)

NODE_CLASS_MAPPINGS = {**m1, **m2, **m3, **m4, **m5}
__all__ = ["NODE_CLASS_MAPPINGS"]
