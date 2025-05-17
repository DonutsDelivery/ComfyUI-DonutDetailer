# existing nodes’ class mappings
from .DonutDetailer         import NODE_CLASS_MAPPINGS as m1
from .DonutDetailer2        import NODE_CLASS_MAPPINGS as m2
from .DonutDetailer4        import NODE_CLASS_MAPPINGS as m3
from .DonutDetailer5        import NODE_CLASS_MAPPINGS as m4
from .DonutDetailerXLBlocks import NODE_CLASS_MAPPINGS as m5
from .DonutClipEncode       import NODE_CLASS_MAPPINGS as m6
from .DonutWidenMerge       import NODE_CLASS_MAPPINGS as m7
from .DonutLoadUNetModels   import NODE_CLASS_MAPPINGS as m8
from .DonutLoadCLIPModels   import NODE_CLASS_MAPPINGS as m9

# new LoRA nodes (include display names)
from .donut_lora_nodes      import NODE_CLASS_MAPPINGS        as m_lora
from .donut_lora_nodes      import NODE_DISPLAY_NAME_MAPPINGS as d_lora

# build globals
NODE_CLASS_MAPPINGS = {
    **m1, **m2, **m3, **m4, **m5,
    **m6, **m7, **m8, **m9,
    **m_lora,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **d_lora,
}
