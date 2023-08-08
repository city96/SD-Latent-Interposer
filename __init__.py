# only import if running as a custom node
try:
	import comfy.utils
except ImportError:
	pass
else:
	from .comfy_latent_interposer import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
	__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
