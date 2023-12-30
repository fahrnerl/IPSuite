import ipsuite as ips
from ase import Atoms
import ase.units
import ase
import numpy as np
import zntrack


class CutoutsFromStructures(ips.base.ProcessSingleAtom):

    structure = zntrack.params(None)
    central_atom_index: int = zntrack.params(None)
    r_cutoff: float = zntrack.params()
    seed: int = zntrack.params(1)

    def __post_init__(self):
        self.structure = self.get_data()
        np.random.seed(self.seed)
        if self.central_atom_index is None:
            self.central_atom_index = np.random.randint(len(self.structure))

    def _center_wrap(self, structure: Atoms, atom_index: int):
        v = np.array([0.5, 0.5, 0.5]) - structure.get_scaled_positions()[atom_index]
        structure.set_scaled_positions(structure.get_scaled_positions() + v)
        structure.wrap()
        return structure
    
    def _molecule_coord_correction(self, molecule: Atoms, structure: Atoms, atom_index: int):

        distances = np.linalg.norm(
            molecule.get_scaled_positions() - structure.get_scaled_positions()[atom_index],
            axis=1
        )
        min_index = np.argmin(distances)
        v = np.array([0.5, 0.5, 0.5]) - molecule.get_scaled_positions()[min_index]
        molecule = self._center_wrap(molecule, min_index)
        molecule.set_scaled_positions(molecule.get_scaled_positions() - v)

        return molecule


    def _merge(self, mol_list, structure: Atoms, atom_index: int):

        new_structure = []

        for molecule in mol_list:
            molecule = self._molecule_coord_correction(molecule, structure, atom_index)
            for atom in molecule:
                new_structure.append(atom)

        return Atoms(new_structure)

    def _cut(self):
        self.structure = self._center_wrap(self.structure, self.central_atom_index)
        mol_list = ase.build.separate(self.structure)
        cutout_molecules = []

        for molecule in mol_list:
            for atom in molecule:
                if np.linalg.norm(atom.position - self.structure[self.central_atom_index].position) <= self.r_cutoff:
                    cutout_molecules.append(molecule)
                    break

        soft = self._merge(cutout_molecules, self.structure, self.central_atom_index)

        return soft
    
    
    def _cell_opt(self):
        pass


    def run(self):
        pass

        




