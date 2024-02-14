from ipsuite import base
from ase import Atoms
from ase.build import separate
import ase.units
import ase
import numpy as np
import zntrack
from scipy.optimize import minimize
from tqdm import tqdm


class CutoutFromStructure(base.ProcessSingleAtom):

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
    r_cutoff: float = zntrack.params(8.)
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

        self.structure = self._merge(cutout_molecules, self.structure, self.central_atom_index)

        return self.structure
    
    def _function_to_optimize(self, cell_param):

        if len(cell_param) == 1:
            cell_param = np.array([cell_param[0], cell_param[0], cell_param[0]])

        self.structure.set_cell(cell_param)
        self.structure.set_pbc([True, True, True])
        self.structure.center()
        
        distances = self.structure.get_all_distances(mic=True)

        result = np.sum(np.maximum([0], self.threshold - distances))

        return result
    
    def _initial_cell(self):
        
        min = np.min(self.structure.get_positions(), axis=0)
        max = np.max(self.structure.get_positions(), axis=0)
        tetragonal = (max - min)/2
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
        
        self.cutout.center()
        self.atoms = [self.cutout]

def center_wrap(structure: ase.Atoms, atom_index: int) -> ase.Atoms:

    """
    Centers Structure around atom_index
    and wraps atoms in cell
    """

    v = np.array([0.5, 0.5, 0.5]) - structure.get_scaled_positions()[atom_index]
    structure.set_scaled_positions(structure.get_scaled_positions() + v)
    structure.wrap()
    return structure

def molecule_coord_correction(molecule: ase.Atoms, structure: ase.Atoms, atom_index: int) -> ase.Atoms:

    """
    Unwraps molecules by taking atom
    nearest to central atom and centering,
    wraping around it then moving molecule
    to original position of the atom.
    """

    distances = np.linalg.norm(
        molecule.get_scaled_positions() - structure.get_scaled_positions()[atom_index],
        axis=1
    )
    min_index = np.argmin(distances)
    v = np.array([0.5, 0.5, 0.5]) - molecule.get_scaled_positions()[min_index]
    molecule = center_wrap(molecule, min_index)
    molecule.set_scaled_positions(molecule.get_scaled_positions() - v)

    return molecule

def merge(mol_list: list[ase.Atoms], structure: ase.Atoms, atom_index: int) -> ase.Atoms:

    """
    Creating atoms object with corrected
    coordinates from molecule list.
    """

    new_structure = []

    for molecule in mol_list:
        molecule = molecule_coord_correction(molecule, structure, atom_index)
        for atom in molecule:
            new_structure.append(atom)

    return Atoms(new_structure)

def cut(structure: ase.Atoms, central_atom_index: int, r_cutoff: float) -> ase.Atoms:

    """
    Performs cutout aroung central atom index.
    """

    structure = center_wrap(structure, central_atom_index)
    mol_list = separate(structure)
    cutout_molecules = []

    for molecule in mol_list:
        for atom in molecule:
            if np.linalg.norm(atom.position - structure[central_atom_index].position) <= r_cutoff:
                cutout_molecules.append(molecule)
                break

    structure = merge(cutout_molecules, structure, central_atom_index)

    return structure

class ObjectiveFunction:

    """
    Class for cell size correction that returns
    objective function where threshold
    defines minimum distance atoms should
    have to atoms of the neighbouring cells.
    """

    def __init__(self, structure, threshold) -> None:
        self.structure: ase.Atoms = structure.copy()
        self.threshold: float = threshold

    def __call__(self, cell_param):

        if len(cell_param) == 1:
            cell_param = np.array([cell_param[0], cell_param[0], cell_param[0]])

        self.structure.set_cell(cell_param)
        self.structure.set_pbc([True, True, True])
        self.structure.center()
        
        distances = self.structure.get_all_distances(mic=True)

        result = np.sum(np.maximum([0], self.threshold - distances))

        return result

def initial_cell(structure: ase.Atoms) -> np.ndarray[float]:
    
    """
    Returns tetragonal and cubic cell
    for initial guess.
    """

    min = np.min(structure.get_positions(), axis=0)
    max = np.max(structure.get_positions(), axis=0)
    tetragonal = (max - min)
    cubic = np.array([np.max(tetragonal)])

    return tetragonal, cubic

def cell_size_correction(structure: ase.Atoms, obj_func=None, threshold=1.8, type=1, step_size=0.05, sensitivity=0.1) -> np.ndarray[float]:

    """
    Decreases cell size in one or three
    dimensions until gradient of objective
    function increases.
    """

    if obj_func == None:
        obj_func = ObjectiveFunction(structure, threshold)

    cell = initial_cell(structure)[type]+threshold

    for axis in range(len(cell)):

        cell_x = cell.copy()
        cell_x[axis] -= step_size
        gradient = (obj_func(cell_x) - obj_func(cell))/step_size

        while gradient <= sensitivity:

            cell = cell_x.copy()
            cell_x[axis] -= step_size
            gradient = (obj_func(cell_x) - obj_func(cell))/step_size

    return cell

class CutoutsFromStructures(base.ProcessAtoms):

    """
    Node for performing spheric cutout around central atoms
    from structures while keeping molecules intact
    followed by a basic cell optimization.
    
    Attributes
    ----------
    central_atom_index: list[int]
        Indices of atoms that are the center of the spheric cutout.
        If None index is chosen randomly.
    r_cutoff: float
        Radius of the sphere.
    seed: int
        Seed value.
    threshhold: float
        Minimal distance(mic) for cell size correction.
    obj_func: function
        Objective Function for cell size correction.
        Has to take cell parameters as input.
    step_size: float
        Increments while decreasing cell size.
    sensitivity: float
        Termination criterion for cell size correction.
    cell_size_correction_type: str
        Method for cell size correction. Either cubic or tetragonal.
    atoms: list[ase.Atoms]
        The processed atoms data. This is an output of the Node.
    """

    central_atom_indices: list[int] = zntrack.deps(None)
    r_cutoff: float = zntrack.params(8.)
    seed: int = zntrack.params(1)
    threshold: float = zntrack.params(1.8)
    obj_func = zntrack.params(None)
    step_size: float = zntrack.params(0.05)
    sensitivity: float = zntrack.params(0.1)
    cell_size_correction_type: str = zntrack.params("cubic")

    def run(self):
        np.random.seed(self.seed)
        if self.central_atom_indices is None:
            self.central_atom_indices = [np.random.randint(len(atoms)) for atoms in self.get_data()]
        else:
            self.central_atom_indices = self.central_atom_indices.selected_indices
        if len(self.get_data()) != len(self.central_atom_indices):
            raise ValueError("central_atom_indices and data have to be of the same length")

        cutouts = []

        for i, structure in enumerate(tqdm(self.get_data())):
            cutout = cut(structure, self.central_atom_indices[i], self.r_cutoff)

            if self.cell_size_correction_type == "cubic":
                x = cell_size_correction(structure, obj_func=self.obj_func, threshold=self.threshold, type=1, step_size=self.step_size, sensitivity=self.sensitivity)
                cutout.set_cell(np.full(3, x[0]))
                cutout.set_pbc([True, True, True])

            elif self.cell_size_correction_type == "tetragonal":
                x = cell_size_correction(structure, obj_func=self.obj_func, threshold=self.threshold, type=0, step_size=self.step_size, sensitivity=self.sensitivity)
                cutout.set_cell(x)
                cutout.set_pbc([True, True, True])

            else:
                raise NotImplementedError("string has to be 'cubic' or 'tetragonal'")
            
            cutout.center()
            cutouts.append(cutout)
        
        self.atoms = cutouts

        




