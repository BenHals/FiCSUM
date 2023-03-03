
from numpy import zeros, max, sqrt, isnan, isinf, dot, diag, count_nonzero
from numpy.linalg import svd, linalg
from scipy.linalg import svd as scipy_svd
from scipy.sparse.linalg import svds as scipy_svds
import numpy as np


class MatrixSketcherBase:

    def __init__(self, d, ell):
        self.d = d
        self.ell = ell
        self._sketch = zeros((self.ell, self.d))

    # Appending a row vector to sketch
    def append(self, vector):
        pass

    # Convenient looping numpy matrices row by row
    def extend(self, vectors):
        for vector in vectors:
            self.append(vector)

    # returns the sketch matrix
    def get(self):
        return self._sketch

    # Convenience support for the += operator  append
    def __iadd__(self, vector):
        self.append(vector)
        return self


class FrequentDirections(MatrixSketcherBase):

    def __init__(self, d, ell):
        self.class_name = 'FrequentDirections'
        self.d = d
        self.ell = ell
        self.m = 2*self.ell
        self._sketch = zeros((self.m, self.d))
        self.row_multipliers = np.full(self.ell, 1.0)
        self.nextZeroRow = 0

    def append(self, vector):
        if count_nonzero(vector) == 0:
            return

        if self.nextZeroRow >= self.m:
            self.__rotate__()

        self._sketch[self.nextZeroRow, :] = vector
        self.nextZeroRow += 1

    def __rotate__(self):
        try:
            [_, s, Vt] = svd(self._sketch, full_matrices=False)
        except linalg.LinAlgError as err:
            [_, s, Vt] = scipy_svd(self._sketch, full_matrices=False)

        if len(s) >= self.ell:
            try:
                shrunk = s[:self.ell]**2 - s[self.ell]**2
                try:
                    sShrunk = sqrt(shrunk)
                except:
                    sShrunk = sqrt([v if v > 0.0 else 0.0 for v in shrunk])

            except Exception as e:
                print("Error!")
                print(s)
                print(s[:self.ell]**2 - s[self.ell-1]**2)
                raise e
            self.row_multipliers = sShrunk
            self._sketch[:self.ell:, :] = dot(diag(sShrunk), Vt[:self.ell, :])
            self._sketch[self.ell:, :] = 0
            self.nextZeroRow = self.ell
        else:
            self.row_multipliers[:len(s)] = s
            self.row_multipliers[self.row_multipliers == 0.0] = 1.0
            self._sketch[:len(s), :] = dot(diag(s), Vt[:len(s), :])
            self._sketch[len(s):, :] = 0
            self.nextZeroRow = len(s)

    def get(self):
        return self._sketch[:self.ell, :]

    def get_observation_matrix(self):
        bounded_obs = min(self.ell, self.nextZeroRow)
        combined_observations = self._sketch[:bounded_obs, :]
        multiplier = np.reciprocal(
            self.row_multipliers[:bounded_obs]).reshape(bounded_obs, 1)
        normed_observations = np.multiply(combined_observations, multiplier)
        return normed_observations

    def get_column(self, column_index):
        """ Get the values of the column.
        The rows maintained in the matrix are linear combinations of observations,
        wheras we are interested in the raw observations.
        self.row_multipliers represents the eiganvalues used to create the linear combinations,
        so we divide by these to extract values closer to the raw observations.
        """
        raw_column = self._sketch[:self.ell, column_index]
        # row multipliers represents eiganvalues > 0
        # Sometimes it gets 0 values, so we have to reset
        # and recalculate
        recalced = False
        try:
            scaled = np.multiply(np.reciprocal(
                self.row_multipliers[:self.ell]), raw_column[:self.ell])
        except Exception as e:
            # Sometimes a 0 is in row_multiplers, so we have to recreate.
            recalced = True
            try:
                [_, s, Vt] = svd(self._sketch, full_matrices=False)
            except linalg.LinAlgError as err:
                [_, s, Vt] = scipy_svd(self._sketch, full_matrices=False)
            shrunk = s[:self.ell]**2 - s[self.ell-1]**2
            shrunk[-1] = 0.0
            try:
                sShrunk = sqrt(shrunk)
            except:
                sShrunk = sqrt([v if v > 0.0 else 0.0 for v in shrunk])
            self.row_multipliers = np.full(self.ell, 1.0)
            last_v = 1
            for i, v in enumerate(sShrunk):
                if v <= 0:
                    self.row_multipliers[i] = last_v
                else:
                    self.row_multipliers[i] = v
                    last_v = v
            scaled = np.multiply(np.reciprocal(
                self.row_multipliers), raw_column)
        if np.isnan(scaled).any():
            print(
                f"Got Nan in get_column: multiplers: {self.row_multipliers}, raw_column: {raw_column}, recalced: {recalced}")
        return scaled[:self.ell]

    def get_column_min(self, column_index):
        """ Get the values of the column, restricted to those seen.
        (get_column returns nan if it has't seen up to self.ell)
        The rows maintained in the matrix are linear combinations of observations,
        wheras we are interested in the raw observations.
        self.row_multipliers represents the eiganvalues used to create the linear combinations,
        so we divide by these to extract values closer to the raw observations.
        """
        num_seen = min(self.ell, self.nextZeroRow)
        raw_column = self._sketch[:num_seen, column_index]
        # row multipliers represents eiganvalues > 0
        # Sometimes it gets 0 values, so we have to reset
        # and recalculate
        recalced = False
        try:
            scaled = np.multiply(np.reciprocal(
                self.row_multipliers[:num_seen]), raw_column[:num_seen])
        except Exception as e:
            # Sometimes a 0 is in row_multiplers, so we have to recreate.
            recalced = True
            try:
                [_, s, Vt] = svd(self._sketch, full_matrices=False)
            except linalg.LinAlgError as err:
                [_, s, Vt] = scipy_svd(self._sketch, full_matrices=False)
            shrunk = s[:num_seen]**2 - s[num_seen-1]**2
            shrunk[-1] = 0.0
            try:
                sShrunk = sqrt(shrunk)
            except:
                sShrunk = sqrt([v if v > 0.0 else 0.0 for v in shrunk])
            self.row_multipliers = np.full(num_seen, 1.0)
            last_v = 1
            for i, v in enumerate(sShrunk):
                if v <= 0:
                    self.row_multipliers[i] = last_v
                else:
                    self.row_multipliers[i] = v
                    last_v = v
            scaled = np.multiply(np.reciprocal(
                self.row_multipliers), raw_column)
        return scaled[:num_seen]

    def make_reset_column(self, value):
        """ When we reset, we want to set a column to one value, to represent forgetting
        past observations. However, we have to maintain the structure of the matrix rows.
        We multiply the value by the eiganvalues of the matrix. 
        We use the eigan values as of the last rotation, so the replacements match
        the first self.ell rows.
        """
        raw_column = np.full(self.ell, value)
        # row multipliers represents eiganvalues > 0
        # Sometimes it gets 0 values, so we have to reset
        # and recalculate
        try:
            scaled = np.multiply(self.row_multipliers, raw_column)
        except Exception as e:
            try:
                [_, s, Vt] = svd(self._sketch, full_matrices=False)
            except linalg.LinAlgError as err:
                [_, s, Vt] = scipy_svd(self._sketch, full_matrices=False)
            shrunk = s[:self.ell]
            scaled = np.multiply(shrunk, raw_column)
        return scaled[:self.ell]
