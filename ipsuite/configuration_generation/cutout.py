import ipsuite as ips
from ase import Atoms
import ase.units
import numpy as np
import zntrack


class CutoutsFromStructures(ips.base.ProcessSingleAtom):

    structure = zntrack.params()
    central_atom_index: int = zntrack.params(None)
    r_cutoff: float = zntrack.params()
    seed: int = zntrack.params(1)

    def __post_init__(self):
        self.structure = self.get_data()
        if self.central_atom_index is None:
            self.central_atom_index = np.random.randint(len(self.structure))

    def _center_wrap(self):
        v = np.array([0.5, 0.5, 0.5]) - self.structure.get_scaled_positions()[self.central_atom_index]
        self.structure.set_scaled_positions(self.structure.get_scaled_positions() + v)
        self.structure.wrap()


    def run(self):
        np.random.seed(self.seed)

        




