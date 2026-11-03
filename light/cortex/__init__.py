"""CLN — Columnar Laminar Network: корковая сеть из колонок со слоями,
нейромодуляторами, ожиданиями, сном и иерархией областей."""
from .connectome import Connectome
from .encoder import WordEncoder
from .neuromodulators import Neuromodulators
from .node import CorticalNode
from .region import CorticalRegion

__all__ = ["CorticalRegion", "CorticalNode", "Connectome", "WordEncoder", "Neuromodulators"]
