"""Selecting atoms with a given step between them."""
import typing
from ipsuite import base
from ipsuite import fields

import ase
import matplotlib.pyplot as plt
import numpy as np
import zntrack
from ipsuite.utils import combine

from ipsuite.analysis.ensemble import plot_with_uncertainty
from ipsuite.configuration_selection import ConfigurationSelection


class ThresholdSelection(ConfigurationSelection):
    """Select atoms based on a given threshold.

    Select atoms above a given threshold or the n_configurations with the
    highest / lowest value. Typically useful for uncertainty based selection.

    Attributes
    ----------
    key: str
        the key in 'calc.results' to select from
    threshold: float, optional
        All values above (or below if negative) this threshold will be selected.
        If n_configurations is given, 'self.threshold' will be prioritized,
        but a maximum of n_configurations will be selected.
    reference: str, optional
        For visualizing the selection a reference value can be given.
        For 'energy_uncertainty' this would typically be 'energy'.
    n_configurations: int, optional
        number of configurations to select.
    min_distance: int, optional
        minimum distance between selected configurations.
    """

    key = zntrack.params("energy_uncertainty")
    reference = zntrack.params("energy")
    threshold = zntrack.params(None)
    n_configurations = zntrack.params(None)
    min_distance: int = zntrack.params(1)
    img_selection = zntrack.outs_path(zntrack.nwd / "selection.png")

    def _post_init_(self):
        if self.threshold is None and self.n_configurations is None:
            raise ValueError("Either 'threshold' or 'n_configurations' must not be None.")

        return super()._post_init_()

    def select_atoms(self, atoms_lst: typing.List[ase.Atoms]) -> typing.List[int]:
        """Take every nth (step) object of a given atoms list.

        Parameters
        ----------
        atoms_lst: typing.List[ase.Atoms]
            list of atoms objects to arange

        Returns
        -------
        typing.List[int]:
            list containing the taken indices
        """
        if self.key == "forces_uncertainty":
            values = np.array([np.max(np.linalg.norm(atoms.calc.results[self.key], axis=1)) for atoms in atoms_lst])
        else:
            values = np.array([atoms.calc.results[self.key] for atoms in atoms_lst])
        if self.threshold is not None:
            if self.threshold < 0:
                indices = np.where(values < self.threshold)[0]
                if self.n_configurations is not None:
                    indices = np.argsort(values)
            else:
                indices = np.where(values > self.threshold)[0]
                if self.n_configurations is not None:
                    indices = np.argsort(values)[::-1]
        else:
            if np.mean(values) > 0:
                indices = np.argsort(values)[::-1]
            else:
                indices = np.argsort(values)

        selected = []
        for val in indices:
            # If the value is close to any of the already selected values, skip it.
            if not any(np.abs(val - np.array(selected)) < self.min_distance):
                selected.append(val)
            if len(selected) == self.n_configurations:
                break

        self._get_plot(atoms_lst, np.array(selected))

        return selected

    def _get_plot(self, atoms_lst: typing.List[ase.Atoms], indices: typing.List[int]):
        if self.key == "forces_uncertainty":
            values = np.array([np.max(np.linalg.norm(atoms.calc.results[self.key], axis=1)) for atoms in atoms_lst])
        else:
            values = np.array([atoms.calc.results[self.key] for atoms in atoms_lst])
        if self.reference is not None:
            reference = np.array(
                [atoms.calc.results[self.reference] for atoms in atoms_lst]
            )
            fig, ax, _ = plot_with_uncertainty(
                {"std": values, "mean": reference},
                ylabel=self.key,
                xlabel="configuration",
            )
            ax.plot(indices, reference[indices], "x", color="red")
        else:
            fig, ax = plt.subplots()
            ax.plot(values, label=self.key)
            ax.plot(indices, values[indices], "x", color="red")
            ax.set_ylabel(self.key)
            ax.set_xlabel("configuration")

        fig.savefig(self.img_selection, bbox_inches="tight")

class SingleAtomThresholdSelection(base.ProcessAtoms):
    """Select single atom index per atoms object based on a given threshold.

    Attributes
    ----------
    key: str
        The key in 'calc.results' to select from.
    """

    key: str = zntrack.params("forces_uncertainty")
    selected_indices: typing.Dict[str, typing.List[int]] = zntrack.outs()

    def select_atoms(self, data) -> list[int]:
        """Take a single atom index per atoms object of a given atoms list.

        Returns
        -------
        typing.List[int]:
            List containing the taken indices.
        """

        indices = []

        for atoms in data:
            values = np.array(atoms.calc.results[self.key])
            if "forces" in self.key:
                values = np.linalg.norm(values, axis=1)
            index = np.argmax(values)
            indices.append(index)

        return indices
    
    def run(self):

        data = self.get_data()
        self.selected_indices = [int(i) for i in self.select_atoms(data)]
        self.atoms = data
