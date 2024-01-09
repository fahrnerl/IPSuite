import ipsuite as ips
from ase import Atoms
import ase.units
import ase
import numpy as np
import zntrack
from scipy.optimize import minimize


class CutoutsFromStructures(ips.base.ProcessSingleAtom):

    """
    Node for performing spheric cutout of a system while
    keeping molecules intact followed by a basic cell optimization.

    Attributes
    ----------
    structure: Atoms
        System from which cutout is performed.
    central_atom_index: int
        Index of atom that is the center of the spheric cutout.
    r_cutoff: float
        Radius of the sphere.
    seed: int
        Seed value.
    threshhold: float
        Minimal distance(mic) for cell optimization.
    cell_opt_type: str
        Method for cell optimization. Either cubic or tetragonal.
    """

    structure: Atoms = zntrack.params(None)
    central_atom_index: int = zntrack.params(None)
    r_cutoff: float = zntrack.params()
    seed: int = zntrack.params(1)
    threshold: float = zntrack.params(1.8)
    cell_opt_type: str = zntrack.params('cubic')

    cutout: Atoms = zntrack.outs()

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
    
    def _function_to_optimize(self, cell_param):

        if len(cell_param) == 1:
            cell_param = np.array([cell_param[0], cell_param[0], cell_param[0]])

        self.structure.set_cell(cell_param)
        self.structure.set_pbc([True, True, True])
        self.structure.center()
        
        distances = []
        for i, atom in enumerate(self.structure):
            distances.append(self.structure.get_all_distances(mic=True))
        
        distances = np.array(distances)

        result = np.sum(np.maximum([0], np.subtract(self.threshold, distances)))

        return result
    
    def _initial_cell(self):
        
        min = np.min(self.structure.get_positions(), axis=0)
        max = np.max(self.structure.get_positions(), axis=0)
        tetragonal = max - min
        cubic = np.max(tetragonal)

        return tetragonal, cubic

    def _optimize_cubic_cell(self):

        opt = minimize(self._function_to_optimize, self._initial_cell()[1], tol=1e-2)
        return np.array([opt.get("x")[0], opt.get("x")[0], opt.get("x")[0]])
    
    def _optimize_tetragonal_cell(self):
        
        opt = minimize(self._function_to_optimize, self._initial_cell()[0], tol=1e-2)
        return opt.get("x")


    def run(self):
        self.cutout = self._cut()

        if self.cell_opt_type == 'cubic':
            self.cutout.set_cell(self._optimize_cubic_cell())
            self.cutout.set_pbc([True, True, True])

        elif self.cell_opt_type == 'tetragonal':
            self.cutout.set_cell(self._optimize_tetragonal_cell())
            self.cutout.set_pbc([True, True, True])

        else:
            raise NotImplementedError("string has to be 'cubic' or 'tetragonal'")
        
        self.atoms = [self.cutout]

        




