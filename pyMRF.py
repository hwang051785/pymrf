"""
pyMRF is a vectorized Python repository for Markov random field simulation,

************************************************************************************************
"""

import numpy as np
from copy import copy
from scipy.stats import multivariate_normal, norm  # normal distributions
from math import ceil
from tqdm import trange  # smart-ish progress bar
from scipy.linalg import sqrtm


class Element:
    # Element object is the most basic object for random field simulation
    
    def __init__(self, coord=None, phys_shp=None, init_field=np.array([]), mask=None, n_labels=None, stencil=None):
        """
            phys_shp (:obj:'np.array'): vector containing the number of elements of each physical dimension,
                                     for 1-d problem: len(phys_shp) == 1, [number of nodes]
                                     for 2-d problem: len(phys_shp) == 2, [num_rows, num_columns]
                                     for 3-d problem: len(phys_shp) == 3, [num_layers, num_rows, num_columns]

            coord (:obj:'np.array'): two-dimensional matrix containing the first and last coordinate of each physical
                                     dimension
                following format:

                    1D scenario: coord = [y_0, y_n]
                    2D scenario: coord = [[y_0, y_n],
                                          [x_0, x_n]]
                    3D scenario: coord = [[y_0, y_n],
                                          [x_0, x_n],
                                          [z_0, z_n]]
                    Note: x,y,z are coordinates
            init_field (:obj:'np.array'): initial MRF with 'np.nan' filled at unknown elements.
                NOTE: if init_field with np.nan at unknown pixels, the entire np.array will have type 'float64' instead
                of 'int32', hence the data type could be either 'float64' or 'int32'.

            mask (:obj: boolean 'np.array'): the mask indicating elements that will be considered for likelihood
                estimation. The default value is "None", which means no mask. The shape of mask must be the same as
                init_field. The boolean "True"s inside this matrix indicate elements that will be used for likelihood
                calculation; "False"s indicate they are masked elements that will not be considered for likelihood
                estimation.

            n_labels (:int): number of labels.

            stencil (:str): specifying the stencil of the neighborhood system used in the Gibbs energy
                calculation. The default value is 'None'. Currently only the eight-point '8p' neighborhood system
                is supported.
        """
        # store physical dimension
        if init_field.size == 0:
            if phys_shp is None:
                raise Exception("phys_shp need to be provided!")
            else:
                self.phys_shp = phys_shp
                self.labels = []
                self.init_field = np.empty(phys_shp).flatten()
                self.init_field[:] = np.nan
                self.fixed_flag = ~np.isnan(self.init_field)  # flag vector indicating known elements
                self.num_fixed = np.sum(self.fixed_flag) 
                if n_labels is None:
                    raise Exception("n_labels need to be provided!")
                else:
                    self.n_labels = n_labels
                self.num_not_fixed = np.sum(~self.fixed_flag)
                self.init_field = self.init_field.astype(int)  # need to be converted to 'int32'
        else:
            self.phys_shp = init_field.shape
            self.labels = []
            self.init_field = init_field.flatten()
            self.fixed_flag = ~np.isnan(self.init_field)
            self.num_fixed = np.sum(self.fixed_flag) 
            self.init_field = self.init_field.astype(int)  # need to be converted to 'int32'
            self.n_labels = len(np.unique(self.init_field[self.fixed_flag]))
            self.num_not_fixed = np.sum(~self.fixed_flag)

        # store the number of physical dimension
        self.phyDim = len(self.phys_shp)
        # calculate the total number of pixels
        self.num_pixels = np.prod(self.phys_shp)
        # implement the pseudo-color method
        self.stencil = stencil
        self.colors = pseudocolor(self.phys_shp, self.stencil)
        # store the flattened mask matrix
        if mask is None:
            self.mask = None
        else:
            self.mask = mask.flatten()

        # calculate coordinate for each pixel
        # 1D scenario
        if self.phyDim == 1:
            if coord is not None:
                # calculate the coordinate increment
                # y_0 = coord[0]; y_n = coord[1]
                delta_coord = (coord[1] - coord[0]) / (self.phys_shp[0] - 1)
                # create coordinate vector
                # an n-by-one matrix
                self.coords = np.array([np.arange(coord[0], coord[1] + delta_coord, delta_coord)]).T
            else:
                self.coords = np.array([np.arange(0, self.phys_shp[0], 1)]).T

        # 2D scenario
        elif self.phyDim == 2:
            if coord is not None:
                # calculate the coordinate increment
                # x_0 = coord[1, 0];    x_n = coord[1, 1];
                # y_0 = coord[0, 0];    y_n = coord[0, 1];
                delta_coord_y = (coord[0, 1] - coord[0, 0]) / (self.phys_shp[0] - 1)
                delta_coord_x = (coord[1, 1] - coord[1, 0]) / (self.phys_shp[1] - 1)
                # create coordinate grid vector
                y, x = np.indices(dimensions=self.phys_shp, dtype=float)
                self.coords = np.array([y.flatten(), x.flatten()]).T
                # y-coords
                self.coords[:, 0] = (self.coords[:, 0].max() - self.coords[:, 0]) * delta_coord_y + coord[0, 0]
                # x-coords
                self.coords[:, 1] = self.coords[:, 1] * delta_coord_x + coord[1, 0]
            else:
                # create coordinate grid vector
                y, x = np.indices(dimensions=self.phys_shp, dtype=float)
                self.coords = np.array([y.flatten(), x.flatten()]).T
                # y-coords
                self.coords[:, 0] = (self.coords[:, 0].max() - self.coords[:, 0])

        # 3D scenario
        elif self.phyDim == 3:
            if coord is not None:
                # calculate the coordinate increment
                # x_0 = coord[1, 0];    x_n = coord[1, 1];
                # y_0 = coord[0, 0];    y_n = coord[0, 1];
                # z_0 = coord[2, 0];    y_n = coord[2, 1];
                delta_coord_y = (coord[0, 1] - coord[0, 0]) / (self.phys_shp[0] - 1)
                delta_coord_x = (coord[1, 1] - coord[1, 0]) / (self.phys_shp[2] - 1)
                delta_coord_z = (coord[2, 1] - coord[2, 0]) / (self.phys_shp[1] - 1)
                # create coordinate grid vector
                y, z, x = np.indices(dimensions=self.phys_shp, dtype=float)
                self.coords = np.array([y.flatten(), z.flatten(), x.flatten()]).T
                # y-coords
                self.coords[:, 0] = (self.coords[:, 0].max() - self.coords[:, 0]) * delta_coord_y + coord[0, 0]
                # z-coords
                self.coords[:, 1] = (self.coords[:, 1].max() - self.coords[:, 1]) * delta_coord_z + coord[1, 0]
                # x-coords
                self.coords[:, 2] = self.coords[:, 2] * delta_coord_x + coord[2, 0]       
            else:
                # create coordinate grid vector
                y, z, x = np.indices(dimensions=self.phys_shp, dtype=float)
                self.coords = np.array([y.flatten(), z.flatten(), x.flatten()]).T
                # y-coords
                self.coords[:, 0] = (self.coords[:, 0].max() - self.coords[:, 0])
                # z-coords
                self.coords[:, 1] = (self.coords[:, 1].max() - self.coords[:, 1])
        # physical dimension mismatch
        else:
            raise Exception("Data format appears to be wrong (neither 1-, 2- or 3-D).")

        # INITIALIZATION ===============================================================================================
        self.betas = []
        self.storage_te = []  # "te" is the total energy
        self.log_target = []
        self.beta_acc_ratio = np.array([])
        self.beta_est = np.nan
        self.beta_std = np.nan
        self.beta_cov = np.nan
        self.label_prob = np.nan
        self.label_map_est = np.nan
        self.info_entr = np.nan  # "info_entr" means "information entropy"
        self.beta_dim = np.nan
        self.prior_beta = np.nan
               
        # For faster calculating gibbs energy, prepare comparison matrix
        if self.phyDim == 2:
            self.comp_deck = np.einsum('ijk,i->ijk',
                                       np.array([np.ones(self.phys_shp), ]*self.n_labels),
                                       np.arange(self.n_labels))
        elif self.phyDim == 3:
            ### xz plane represents the plane of axis 1 and axis 2; yz plane represents the plane of axis 0 and axis 1
            # comparison matrix for xz plane
            self.comp_deck_xz = np.einsum('ijk,i->ijk',
                                       np.array([np.ones((self.phys_shp[1], self.phys_shp[2])), ]*self.n_labels),
                                       np.arange(self.n_labels))
            self.comp_deck_xz_extend = np.kron(self.comp_deck_xz, np.ones((self.phys_shp[0], 1, 1))).astype(float) 
            # comparison matrix for yz plane
            self.comp_deck_yz = np.einsum('ijk,k->ijk',
                                       np.array(np.ones((self.phys_shp[0], self.phys_shp[1], self.n_labels))),
                                       np.arange(self.n_labels))
            self.comp_deck_yz_extend = np.kron(self.comp_deck_yz, np.ones((1, 1, self.phys_shp[2]))).astype(float)
            
    
    
    def calc_gibbs_energy(self, labels, beta):
        """Calculates the Gibbs energy for each element using the granular coefficient(s) beta.

        Args:
            labels (:obj:'np.ndarray'): the list of labels assigned to each element
            beta (:obj:'float' or 'list' of float): if  len(beta) == 1, use isotropic Potts model or 1D scenario, else,
            use anisotropic Potts model.

        Returns:
            ge_list (:obj:`np.ndarray`) : Gibbs energy at every element for each label.
        """
        # 1D
        if self.phyDim == 1:
            # tile
            lt = np.tile(labels, (self.n_labels, 1)).T
            ge = np.arange(self.n_labels)  # elements x labels
            ge = np.tile(ge, (len(labels), 1)).astype(float)

            # first row
            top = np.expand_dims(np.not_equal(np.arange(self.n_labels), lt[1, :]) * beta, axis=0)
            # mid
            mid = (np.not_equal(ge[1:-1, :], lt[:-2, :]).astype(float) + np.not_equal(ge[1:-1, :],
                                                                                      lt[2:, :]).astype(float)) * beta
            # last row
            bot = np.expand_dims(np.not_equal(np.arange(self.n_labels), lt[-2, :]) * beta, axis=0)
            # put back together and return gibbs energy
            ge_list = np.concatenate((top, mid, bot))
            return ge_list

        # 2D
        elif self.phyDim == 2:
            # reshape the labels to 2D for "stencil-application"
            label_image = labels.reshape(self.phys_shp[0], self.phys_shp[1])

            # prepare gibbs energy array
            ref_matrix = np.empty((self.phys_shp[0] + 2, self.phys_shp[1] + 2))
            ref_matrix[:] = np.nan
            ref_matrix[1:self.phys_shp[0] + 1, 1:self.phys_shp[1] + 1] = label_image

            # extract neighbors
            left = np.array([ref_matrix[1:self.phys_shp[0] + 1, 0:self.phys_shp[1]], ]*self.n_labels)
            right = np.array([ref_matrix[1:self.phys_shp[0] + 1, 2:self.phys_shp[1] + 2], ]*self.n_labels)
            top = np.array([ref_matrix[0:self.phys_shp[0], 1:self.phys_shp[1] + 1], ]*self.n_labels)
            bottom = np.array([ref_matrix[2:self.phys_shp[0] + 2, 1:self.phys_shp[1] + 1], ]*self.n_labels)
            upper_left = np.array([ref_matrix[0:self.phys_shp[0], 0:self.phys_shp[1]], ]*self.n_labels)
            upper_right = np.array([ref_matrix[0:self.phys_shp[0], 2:self.phys_shp[1] + 2], ]*self.n_labels)
            lower_left = np.array([ref_matrix[2:self.phys_shp[0] + 2, 0:self.phys_shp[1]], ]*self.n_labels)
            lower_right = np.array([ref_matrix[2:self.phys_shp[0] + 2, 2:self.phys_shp[1] + 2], ]*self.n_labels)

            # calculate left neighbor energy
            diff = self.comp_deck - left
            temp_1 = np.zeros_like(diff, dtype=float)
            temp_1[(diff != 0) & (~np.isnan(diff))] = 1

            # calculate right neighbor energy
            diff = self.comp_deck - right
            temp_2 = np.zeros_like(diff, dtype=float)
            temp_2[(diff != 0) & (~np.isnan(diff))] = 1

            # calculate top neighbor energy
            diff = self.comp_deck - top
            temp_3 = np.zeros_like(diff, dtype=float)
            temp_3[(diff != 0) & (~np.isnan(diff))] = 1

            # calculate bottom neighbor energy
            diff = self.comp_deck - bottom
            temp_4 = np.zeros_like(diff, dtype=float)
            temp_4[(diff != 0) & (~np.isnan(diff))] = 1

            # calculate upper-left neighbor energy
            diff = self.comp_deck - upper_left
            temp_5 = np.zeros_like(diff, dtype=float)
            temp_5[(diff != 0) & (~np.isnan(diff))] = 1

            # calculate upper-right neighbor energy
            diff = self.comp_deck - upper_right
            temp_6 = np.zeros_like(diff, dtype=float)
            temp_6[(diff != 0) & (~np.isnan(diff))] = 1

            # calculate lower-left neighbor energy
            diff = self.comp_deck - lower_left
            temp_7 = np.zeros_like(diff, dtype=float)
            temp_7[(diff != 0) & (~np.isnan(diff))] = 1

            # calculate lower-right neighbor energy
            diff = self.comp_deck - lower_right
            temp_8 = np.zeros_like(diff, dtype=float)
            temp_8[(diff != 0) & (~np.isnan(diff))] = 1

            # multiply beta
            if self.beta_dim == 1:
                ge = (temp_1 + temp_2 + temp_3 + temp_4 + temp_5 + temp_6 + temp_7 + temp_8) * beta
            elif self.beta_dim == 4:
                #  3  1  2
                #   \ | /
                #   --+-- 0
                #   / | \

                self.ge = (temp_1 + temp_2) * beta[0] + \
                     (temp_3 + temp_4) * beta[1] + \
                     (temp_5 + temp_8) * beta[3] + \
                     (temp_6 + temp_7) * beta[2]
            else:
                raise Exception("Other beta configurations are not supported")
            # reshape and transpose gibbs energy, return
            return self.ge.reshape((self.n_labels, -1)).T           
            
        # 3D
        elif self.phyDim == 3:                                 
            # reshape the labels to 3D for "stencil-application"
            label_image = labels.reshape(self.phys_shp[0], self.phys_shp[1], self.phys_shp[2])

            # prepare gibbs energy array for xz plane
            ref_matrix_xz = np.empty((self.phys_shp[0], self.phys_shp[1] + 2, self.phys_shp[2] + 2,))
            ref_matrix_xz[:] = np.nan
            ref_matrix_xz[:, 1:self.phys_shp[1] + 1, 1:self.phys_shp[2] + 1] = label_image            
            ref_matrix_xz_deck = np.tile(ref_matrix_xz, (self.n_labels, 1, 1)).astype(float)
            # prepare gibbs energy array for yz plane
            ref_matrix_yz = np.empty((self.phys_shp[0] + 2, self.phys_shp[1] + 2, self.phys_shp[2],))
            ref_matrix_yz[:] = np.nan
            ref_matrix_yz[1:self.phys_shp[0] + 1, 1:self.phys_shp[1] + 1, :] = label_image            
            ref_matrix_yz_deck = np.tile(ref_matrix_yz, ( 1, 1, self.n_labels)).astype(float)
            
            # extract neighbors
            left_xz = ref_matrix_xz_deck[:, 1:self.phys_shp[1] + 1, 0:self.phys_shp[2]]
            right_xz = ref_matrix_xz_deck[:, 1:self.phys_shp[1] + 1, 2:self.phys_shp[2] + 2]
            top_xz = ref_matrix_xz_deck[:, 0:self.phys_shp[1], 1:self.phys_shp[2] + 1]
            bottom_xz = ref_matrix_xz_deck[:, 2:self.phys_shp[1] + 2, 1:self.phys_shp[2] + 1]
            upper_left_xz = ref_matrix_xz_deck[:, 0:self.phys_shp[1], 0:self.phys_shp[2]]
            upper_right_xz = ref_matrix_xz_deck[:, 0:self.phys_shp[1], 2:self.phys_shp[2] + 2]
            lower_left_xz = ref_matrix_xz_deck[:, 2:self.phys_shp[1] + 2, 0:self.phys_shp[2]]
            lower_right_xz = ref_matrix_xz_deck[:, 2:self.phys_shp[1] + 2, 2:self.phys_shp[2] + 2]
            
            left_yz = ref_matrix_yz_deck[ 0:self.phys_shp[0], 1:self.phys_shp[1]+1 , :]
            right_yz = ref_matrix_yz_deck[ 2:self.phys_shp[0] + 2, 1:self.phys_shp[1] + 1, :]            
            upper_left_yz = ref_matrix_yz_deck[0:self.phys_shp[0], 0:self.phys_shp[1], :]
            upper_right_yz = ref_matrix_yz_deck[2:self.phys_shp[0] + 2, 0:self.phys_shp[1], :]
            lower_left_yz = ref_matrix_yz_deck[0:self.phys_shp[0], 2:self.phys_shp[1] + 2, :]
            lower_right_yz = ref_matrix_yz_deck[2:self.phys_shp[0] + 2, 2:self.phys_shp[1] + 2, :]

            # calculate left_xz neighbor energy
            diff = self.comp_deck_xz_extend - left_xz
            temp_1 = np.zeros_like(diff, dtype=float)
            temp_1[(diff != 0) & (~np.isnan(diff))] = 1

            # calculate right_xz neighbor energy
            diff = self.comp_deck_xz_extend - right_xz
            temp_2 = np.zeros_like(diff, dtype=float)
            temp_2[(diff != 0) & (~np.isnan(diff))] = 1

            # calculate top_xz neighbor energy
            diff = self.comp_deck_xz_extend - top_xz
            temp_3 = np.zeros_like(diff, dtype=float)
            temp_3[(diff != 0) & (~np.isnan(diff))] = 1

            # calculate bottom_xz neighbor energy
            diff = self.comp_deck_xz_extend - bottom_xz
            temp_4 = np.zeros_like(diff, dtype=float)
            temp_4[(diff != 0) & (~np.isnan(diff))] = 1

            # calculate upper-left_xz neighbor energy
            diff = self.comp_deck_xz_extend - upper_left_xz
            temp_5 = np.zeros_like(diff, dtype=float)
            temp_5[(diff != 0) & (~np.isnan(diff))] = 1

            # calculate upper-right_xz neighbor energy
            diff = self.comp_deck_xz_extend - upper_right_xz
            temp_6 = np.zeros_like(diff, dtype=float)
            temp_6[(diff != 0) & (~np.isnan(diff))] = 1

            # calculate lower-left_xz neighbor energy
            diff = self.comp_deck_xz_extend - lower_left_xz
            temp_7 = np.zeros_like(diff, dtype=float)
            temp_7[(diff != 0) & (~np.isnan(diff))] = 1

            # calculate lower-right_xz neighbor energy
            diff = self.comp_deck_xz_extend - lower_right_xz
            temp_8 = np.zeros_like(diff, dtype=float)
            temp_8[(diff != 0) & (~np.isnan(diff))] = 1
            
                       
            # calculate left_yz neighbor energy
            diff = self.comp_deck_yz_extend - left_yz
            temp_9 = np.zeros_like(diff, dtype=float)
            temp_9[(diff != 0) & (~np.isnan(diff))] = 1

            # calculate right_yz neighbor energy
            diff = self.comp_deck_yz_extend - right_yz
            temp_10 = np.zeros_like(diff, dtype=float)
            temp_10[(diff != 0) & (~np.isnan(diff))] = 1
            
            # calculate upper-left_yz neighbor energy
            diff = self.comp_deck_yz_extend - upper_left_yz
            temp_11 = np.zeros_like(diff, dtype=float)
            temp_11[(diff != 0) & (~np.isnan(diff))] = 1

            # calculate upper-right_yz neighbor energy
            diff = self.comp_deck_yz_extend - upper_right_yz
            temp_12 = np.zeros_like(diff, dtype=float)
            temp_12[(diff != 0) & (~np.isnan(diff))] = 1

            # calculate lower-left_yz neighbor energy
            diff = self.comp_deck_yz_extend - lower_left_yz
            temp_13 = np.zeros_like(diff, dtype=float)
            temp_13[(diff != 0) & (~np.isnan(diff))] = 1

            # calculate lower-right_yz neighbor energy
            diff = self.comp_deck_yz_extend - lower_right_yz
            temp_14 = np.zeros_like(diff, dtype=float)
            temp_14[(diff != 0) & (~np.isnan(diff))] = 1
                                
            # multiply beta
            if self.beta_dim == 1:
                ge = (temp_1 + temp_2 + temp_3 + temp_4 + temp_5 + temp_6 + temp_7 + temp_8
                      + temp_9 + temp_10+ temp_11 + temp_12+ temp_13 + temp_14) * beta
            elif self.beta_dim == 7:
                # xz:
                #  3  1  2
                #   \ | /
                #   --+-- 0
                #   / | \
                
                # yz:
                #  6  1  5
                #   \ | /
                #   --+-- 4
                #   / | \
                
                ge_xz = (temp_1 + temp_2) * beta[0] + \
                         (temp_3 + temp_4) * beta[1] + \
                        (temp_5 + temp_8) * beta[3] + \
                        (temp_6 + temp_7) * beta[2]
                ge_yz = (temp_9 + temp_10) * beta[4] + \
                        (temp_12 + temp_13) * beta[5] + \
                        (temp_11 + temp_14) * beta[6]
                # make ge_yz have the same shape with ge_xz
                ge_yz_split = np.split(ge_yz,self.n_labels,2)                
                ge_yz_reshape = []
                for t in range(self.n_labels):
                    ge_yz_reshape.extend(ge_yz_split[t])   
                ge_yz_reshape = np.array(ge_yz_reshape)
                # sum ge
                self.ge = ge_xz.reshape((self.n_labels, -1)).T + ge_yz_reshape.reshape((self.n_labels, -1)).T     
            else:
                raise Exception("Other beta configurations are not supported")
            # reshape and transpose gibbs energy, return
            return self.ge
        else:
            raise Exception("Data format appears to be wrong (neither 1-, 2- or 3-D).")
            
    
    def log_prior_density_beta(self, beta):
        """Calculates the log prior density for a given beta array."""
        return np.log(self.prior_beta.pdf(beta))
    
    def get_log_likelihood(self, label_list, beta, t):
        """Get the log_likelihood of a given label_list
        :arg
            label_list:'np.array': a list of field labels
            beta:'list': beta parameter
        :return
            :'float': log_likelihood of the entire label_list
        """
        ge_list = self.calc_gibbs_energy(label_list, beta)
        if self.mask is None:
            label_prob = self._calc_labels_prob(ge_list[self.fixed_flag], t)
            return sum(np.log(label_prob[np.arange(len(label_prob)), label_list[self.fixed_flag]]))
        else:
            label_prob = self._calc_labels_prob(ge_list[self.fixed_flag & self.mask], t)
            return sum(np.log(label_prob[np.arange(len(label_prob)), label_list[self.fixed_flag & self.mask]]))

    def gibbs_sample(self, t, beta_jump_length, verbose, fix_beta, beta_compl_accept, 
                     dist_for_KHMD):
        """Takes care of the Gibbs sampling. This is the main function of the algorithm.

        Args:
            t: Hyperparameter for MCMC
            beta_jump_length: Hyperparameter
            verbose (bool or :obj:`str`): Toggles verbosity.
            fix_beta (bool): Fixed beta to the initial value if True, else adaptive.
            beta_compl_accept (bool): len(beta_compl_accept) is same as len(beta_prior_mean), beta_i proposed 
            from prior beta distribution and accepted completly if beta_compl_accept[i] is True

        Returns:
            The function updates directly on the object variables and appends new draws of labels and
            betas to their respective storages.
        """
        # TODO: [GENERAL] In-depth description of the gibbs sampling function
        # Calculate gibbs/mrf energy
        gibbs_energy = self.calc_gibbs_energy(self.labels[-1], self.betas[-1])
        if verbose == "energy":
            print("gibbs energy:", gibbs_energy)
        total_energy = self.get_total_energy(dist_for_KHMD, gibbs_energy)               
        # Calculate probability of labels
        labels_prob = self._calc_labels_prob(total_energy, t)
        if verbose == "energy":
            print("Labels probability:", labels_prob)

        # append total energy of the latest label_list
        self.storage_te.append(sum(total_energy[np.arange(len(total_energy)), self.labels[-1]]))

        if self.num_not_fixed > 0:
            # make copy of previous labels
            new_labels = copy(self.labels[-1])
            for i, color_f in enumerate(self.colors):
                new_labels[color_f] = draw_labels_vect(labels_prob[color_f])
                new_labels[self.fixed_flag] = self.labels[0][self.fixed_flag]
                # now recalculate gibbs energy from the mixture of old and new labels
                if i < (len(self.colors) - 1):
                    gibbs_energy = self.calc_gibbs_energy(new_labels, self.betas[-1])
                    total_energy = self.get_total_energy(dist_for_KHMD, gibbs_energy)  
                    labels_prob = self._calc_labels_prob(total_energy, t)

            # append labels generated from the current iteration
            self.labels.append(new_labels)
  
            
        if not fix_beta:
            beta_next = copy(self.betas[-1])
            # PROPOSAL STEP
            # make proposals for beta, beta depends on physical dimensions, for 1d its size 1
            beta_prop = self.propose_beta(self.betas[-1], beta_jump_length, beta_compl_accept)
            # ************************************************************************************************
            # UPDATE BETA
            lp_beta_prev = self.log_prior_density_beta(self.betas[-1])
            lg_likelihood_beta_prev = self.get_log_likelihood(self.labels[-1], self.betas[-1], t)
            log_target_prev = lp_beta_prev + lg_likelihood_beta_prev
            log_target = log_target_prev

            if self.beta_dim == 1:
                lp_beta_prop = self.log_prior_density_beta(beta_prop)
                lg_likelihood_beta_prop = self.get_log_likelihood(self.labels[-1], beta_prop, t)
                log_target_prop = lp_beta_prop + lg_likelihood_beta_prop
                beta_eval = evaluate(log_target_prop, log_target_prev)
                if beta_eval[0]:
                    beta_next = beta_prop
                    log_target = log_target_prop
                else:
                    pass
                self.beta_acc_ratio = np.append(self.beta_acc_ratio, beta_eval[1])  # store
                self.betas.append(beta_next)
            else:
                beta_acc_ratio = np.array([])
                for i_beta in range(self.beta_dim):
                    if beta_compl_accept[i_beta]:
                        beta_next[i_beta] = beta_prop[i_beta]
                    else:
                        beta_temp = copy(beta_next)
                        beta_temp[i_beta] = beta_prop[i_beta]
                        # calculate gibbs energy with new labels and proposed beta
                        lp_beta_prop = self.log_prior_density_beta(beta_temp)
                        lg_likelihood_beta_prop = self.get_log_likelihood(self.labels[-1], beta_temp, t)
                        log_target_prop = lp_beta_prop + lg_likelihood_beta_prop
                        beta_eval = evaluate(log_target_prop, log_target_prev)
                        if beta_eval[0]:
                            beta_next[i_beta] = beta_prop[i_beta]
                            log_target = log_target_prop
                        else:
                            pass
                        # store for acc_ratio of beta_i
                        beta_acc_ratio = np.append(beta_acc_ratio, beta_eval[1])
                # store for acc_ratio for a beta vector
                self.beta_acc_ratio = np.append(self.beta_acc_ratio, beta_acc_ratio)
                self.betas.append(beta_next)
            self.log_target.append(log_target)
        else:
            pass
   
    def propose_beta(self, beta_prev, beta_jump_length, beta_compl_accept):
        """Proposes a perturbed beta based on a jump length hyperparameter.

        Args:
            beta_prev:
            beta_jump_length:
            beta_compl_accept (bool): 

        Returns:
            Generated normal distributed random number/vector
        """
        # Possible beta_dim values are in [1, 4]
        if len(np.shape(beta_jump_length)) == 0:
            sigma_prop = np.eye(self.beta_dim) * beta_jump_length
            propose_beta = multivariate_normal(mean=beta_prev, cov=sigma_prop).rvs()
        elif (len(np.shape(beta_jump_length)) == 1) and (len(beta_jump_length) == 4):
            if self.beta_dim == 4:
                sigma_prop = np.diag(beta_jump_length)
            else:
                raise Exception("Beta dimension and beta_jump_length are not compatible")
            propose_beta = multivariate_normal(mean=beta_prev, cov=sigma_prop).rvs()
        elif (len(np.shape(beta_jump_length)) == 1) and (len(beta_jump_length) == 7):
            if self.beta_dim == 7:
                sigma_prop = np.diag(beta_jump_length)
            else:
                raise Exception("Beta dimension and beta_jump_length are not compatible")
            propose_beta = multivariate_normal(mean=beta_prev, cov=sigma_prop).rvs()
        else:
            raise Exception("Possible length of beta_jump_length should be 1 or 4")        
        propose_beta[beta_compl_accept] = self.prior_beta.rvs()[beta_compl_accept]
        return propose_beta

    def fit(self, num_of_iter=100, beta_prior_mean=1, beta_compl_accept=False, fix_beta=False, beta_prior_cov=100,
            beta_init=None, image_init=None, beta_jump_length=0.01, t=1, verbose=False,
            dist_KHMD=None, labels_prob_init=None):
        """Fit the segmentation parameters to the given data.

        Args:
            num_of_iter (int): Number of iterations.
            beta_prior_mean (float or list): Initial granularity parameters for Gibbs energy calculation.
                1) if it is a one-dimensional problem or isotropic problem, beta is a float number
                2) if it is a two-dimensional anisotropic problem, beta is a list of 4 float numbers
            beta_init (list): initial beta
            image_init (ndarray): initial image (flattened)
            fix_beta (bool): If beta is fixed during the computation. If beta is fixed, stochastic simulation will be
            performed instead of estimating beta.
            beta_compl_accept (bool): len(beta_compl_accept) is same as len(beta_prior_mean), beta_i proposed from prior 
            beta distribution and accepted completly if beta_compl_accept[i] is True.
            labels_prob_init (ndarray): initial label probability matrix, n_elements x n_labels.
            beta_jump_length (float): Hyper-parameter specifying the beta proposal.
            t (float): temperature for simulating annealing.
            verbose (bool or :obj:`str`):
            beta_prior_cov (np.ndarray): the covariance matrix (or standard deviation if the length is 1)
            of the beta prior distribution, default is 100.
        """

        # Initialize PRIOR distributions for beta
        if len(self.betas) == 0:
            # No sample in the self.betas list
            if self.phyDim == 1:
                # assign beta_dim, for 1-D problem, beta is a single number.
                self.beta_dim = 1
                # initialize the first beta
                if beta_init is None:
                    self.betas = [beta_prior_mean]
                else:
                    self.betas = [beta_init]
                if not fix_beta:
                    # if beta is not fixed
                    # initialize the beta prior distribution. Note: "beta_prior_cov" is standard deviation.
                    self.prior_beta = norm(beta_prior_mean, beta_prior_cov)
            elif self.phyDim == 2:
                # For 2-D problems
                if len(np.shape(beta_prior_mean)) == 0:
                    # if beta_prior_mean is a single number
                    # using isotropic Potts model
                    self.beta_dim = 1
                    # initialize the first beta
                    if beta_init is None:
                        self.betas = [beta_prior_mean]
                    else:
                        self.betas = [beta_init]
                    if not fix_beta:
                        self.prior_beta = norm(beta_prior_mean, beta_prior_cov)
                elif (len(np.shape(beta_prior_mean)) == 1) and (len(beta_prior_mean) == 4):
                    # if beta_prior_mean is a vector
                    # using anisotropic Potts model
                    self.beta_dim = 4
                    # initialize the first beta
                    if beta_init is None:
                        self.betas = [beta_prior_mean]
                    else:
                        self.betas = [beta_init]
                    if not fix_beta:
                        if len(np.shape(beta_prior_cov)) == 0:
                            # beta_prior_cov is a single number, using eye matrix will same diagonal numbers
                            self.prior_beta = multivariate_normal(beta_prior_mean,
                                                                  np.eye(self.beta_dim) * beta_prior_cov)
                        elif len(np.shape(beta_prior_cov)) == 1:
                            # beta_prior_cov is a vector, using it to generate the diagonal matrix
                            self.prior_beta = multivariate_normal(beta_prior_mean,
                                                                  np.diag(beta_prior_cov))
                        elif len(np.shape(beta_prior_cov)) == 2:
                            # beta_prior_cov is a matrix, directly use it
                            self.prior_beta = multivariate_normal(beta_prior_mean,
                                                                  beta_prior_cov)
                        else:
                            raise Exception(
                                "'beta_prior_cov' can be a single float number, a list of 4 float numbers, or a 4x4 "
                                "matrix")
                else:
                    raise Exception("For 2D problem, beta should be a float number or a list of 4 float numbers")
            elif self.phyDim == 3:
                # For 3-D problems               
                if len(np.shape(beta_prior_mean)) == 0:
                    # if beta_prior_mean is a single number
                    # using isotropic Potts model
                    self.beta_dim = 1
                    # initialize the first beta
                    if beta_init is None:
                        self.betas = [beta_prior_mean]
                    else:
                        self.betas = [beta_init]
                    if not fix_beta:
                        self.prior_beta = norm(beta_prior_mean, beta_prior_cov)
                elif (len(np.shape(beta_prior_mean)) == 1) and (len(beta_prior_mean) == 7):
                    # if beta_prior_mean is a vector
                    # using anisotropic Potts model
                    self.beta_dim = 7
                    # initialize the first beta
                    if beta_init is None:
                        self.betas = [beta_prior_mean]
                    else:
                        self.betas = [beta_init]                    
                    if not fix_beta:
                        if len(np.shape(beta_prior_cov)) == 0:
                            # beta_prior_cov is a single number, using eye matrix will same diagonal numbers
                            self.prior_beta = multivariate_normal(beta_prior_mean,
                                                                  np.eye(self.beta_dim) * beta_prior_cov)
                        elif len(np.shape(beta_prior_cov)) == 1:
                            # beta_prior_cov is a vector, using it to generate the diagonal matrix
                            self.prior_beta = multivariate_normal(beta_prior_mean,
                                                                  np.diag(beta_prior_cov))
                        elif len(np.shape(beta_prior_cov)) == 2:
                            # beta_prior_cov is a matrix, directly use it
                            self.prior_beta = multivariate_normal(beta_prior_mean,
                                                                  beta_prior_cov)
                        else:
                            raise Exception(
                                "'beta_prior_cov' can be a single float number, a list of 7 float numbers, or a 7x7 "
                                "matrix")
                else:
                    raise Exception("For 3D problem, beta should be a float number or a list of 7 float numbers")
                
            else:
                raise Exception("higher dimensional physical space (more than 3-D) not yet supported.")

        # Generate the complete initial image
        if len(self.labels) == 0:
            self.labels = [self.init_field]
            if self.num_not_fixed > 0:
                if image_init is None:
                    if labels_prob_init is None:
                        if self.phyDim == 3:
                            labels_prob = np.ones(shape=(self.num_not_fixed, self.n_labels))*(1/self.n_labels)
                        else:
                            labels_prob_all = np.ones(shape=(self.num_not_fixed, self.n_labels))*(1/self.n_labels) 
                            labels_prob = labels_prob_all[~self.fixed_flag]
                    else:
                        if self.phyDim == 3:
                            labels_prob = labels_prob_init[~self.fixed_flag]
                        else:
                            labels_prob = labels_prob_init                     
                    self.labels[0][~self.fixed_flag] = draw_labels_vect(labels_prob)
                else:
                    self.labels[0][~self.fixed_flag] = image_init.astype(int).flatten()[~self.fixed_flag]                              
        # ************************************************************************************************
        # start gibbs sampler
        #for g in tqdm.trange(num_of_iter):
        for g in trange(num_of_iter):
            self.gibbs_sample(t=t, beta_jump_length=beta_jump_length, verbose=verbose, 
                              fix_beta=fix_beta, beta_compl_accept=beta_compl_accept, 
                              dist_for_KHMD=dist_KHMD)
    
    def get_estimator(self, start_iter):
        est = estimator(self.betas, start_iter)
        self.beta_est = est[0]
        self.beta_std = est[1]
        self.beta_cov = est[2]
    
    def get_label_prob(self, start_iter):
        self.label_prob = np.full((self.n_labels, self.num_pixels), np.nan)
        label_bin = np.array(self.labels)[start_iter:, :]
        for i in range(self.n_labels):
            count_i = np.sum(label_bin == i, axis=0)
            self.label_prob[i, :] = count_i / label_bin.shape[0]
   
    def get_map(self):
        # calculate MAP of labels. NOTE: get_label_prob must be executed first
        self.label_map_est = np.argmax(self.label_prob, axis=0)
    
    def get_ie(self):
        # calculate information entropy. NOTE: get_label_prob must be executed first
        temp = np.copy(self.label_prob)
        temp[np.where(temp == 0)] = 1
        self.info_entr = np.sum(-temp * np.log(temp), axis=0)
    
    def get_total_energy(self, dist_nearest, gibbs_energy):
        
        if dist_nearest is None:
            total_energy = gibbs_energy
        else:
            init_field_lebels = self.init_field[self.fixed_flag]
            init_field_lebels_desk = np.array([init_field_lebels, ]* self.n_labels).T
            comp_deck = np.tile(np.arange(self.n_labels),self.num_fixed).reshape((self.num_fixed,self.n_labels))
            diff = init_field_lebels_desk - comp_deck
            temp = np.zeros_like(diff, dtype=float)
            temp[diff == 0] = 1
            label_flag = np.array(temp).astype(bool) 
            
            dist_for_fixed = np.zeros((self.num_fixed, self.n_labels))
            dist_for_fixed[~label_flag] = 10**10
                        
            dist_for_all = np.ones((self.num_pixels , self.n_labels))
            dist_for_all[self.fixed_flag,:] = dist_for_fixed
            dist_for_all[~self.fixed_flag,:] = dist_nearest
            
            total_energy = dist_for_all + gibbs_energy
        return total_energy
        
    def _calc_labels_prob(self, te, t):
        """"Calculate labels probability for array of total energies (te) and totally arbitrary scalar value t."""
        
        return (np.exp(-te / t).T / np.sum(np.exp(-te / t), axis=1)).T
        
    def get_DANN_distance(self, epsilon, range_v, range_h):
        """"Calculate DANN_distance for all unknown element.
        
        Args:
            epsilon (float): the "softening" parameter for DANN.
            range_v, range_h (int): the range of the selected rectangle neighborhood.
        """
               
        if self.phyDim == 2 : 
            # horizontal and vertical index of all elements with len(num_pixels)                
            idx_h_pre, idx_v_pre = np.meshgrid(np.arange(self.phys_shp[1]), np.arange(self.phys_shp[0]))
            idx_v_pre = idx_v_pre.reshape((self.num_pixels,1))
            idx_h_pre = idx_h_pre.reshape((self.num_pixels,1))
            
            
            # vertical index of borehole elements
            idx_v_fixed = idx_v_pre[self.fixed_flag]
            # horizontal index of borehole elements
            idx_h_fixed = idx_h_pre[self.fixed_flag]
            if self.mask is None:
                # vertical index of unknown elements
                idx_v_not_fixed = idx_v_pre[~self.fixed_flag]            
                # horizontal index of unknown elements
                idx_h_not_fixed = idx_h_pre[~self.fixed_flag]
                num_not_fixed = self.num_not_fixed
            else:                
                idx_v_not_fixed = idx_v_pre[~self.fixed_flag & self.mask]            
                idx_h_not_fixed = idx_h_pre[~self.fixed_flag & self.mask]
                num_not_fixed = self.num_not_fixed - sum(~self.mask)
                                                
            if range_v is None:
                range_v = 5  
            else:
                range_v = range_v
            if range_h is None:
                range_h = self.phys_shp[1]
            else:
                range_h = range_h
            if epsilon is None:
                epsilon = 1  
            else:
                epsilon = epsilon
            # coordinates of borehole elements         
            coor_BH = np.hstack((idx_h_fixed, idx_v_fixed))
            # labels of borehole elements
            BH_labels = self.init_field[self.fixed_flag]
            # coordinates of unknown elements 
            coor_unknown = np.hstack((idx_h_not_fixed, idx_v_not_fixed))     
            
            distance_DANN = []            
            for q in trange(num_not_fixed):
                coor_unknown_i = coor_unknown[q,:] # coordinate of unknown element i
                n_dimension = len(coor_unknown_i)
                flag_coor_BH_h = abs(coor_BH[:,0]-coor_unknown_i[0]) <= range_h
                flag_coor_BH_v = abs(coor_BH[:,1]-coor_unknown_i[1]) <= range_v
                # Mark the borehole elements within the selected rectangle 
                flag_coor_BH = flag_coor_BH_h & flag_coor_BH_v 
                
                #    ----------------------------  --
                #    -                          -   ! 
                #    -                          - range_v
                #    -                          -   !
                #    -             i            -  --
                #    -      (unknown element)     -
                #    -                          -  
                #    -                          -
                #    ----------------------------
                #                  |-- range_h --|                    
                # coordinates of borehole elements within rectangle(neighborhood)                              
                coor_BH_nhood = coor_BH[flag_coor_BH, :]
                coor_BH_nhood_mean = coor_BH_nhood.mean(axis=0)
                BH_nhood_label = BH_labels[flag_coor_BH]  # labels of borehole elements within rectangle 
                labels_in_nhood = np.unique(BH_nhood_label) # Number of labels within rectangle            
                within_label_cov = np.zeros((n_dimension, n_dimension)) # within-label variance
                between_label_mdiff = np.zeros((n_dimension, n_dimension)) # between-label mean differnce
                for target_label in labels_in_nhood:
                    # index of borehole elements having target_label
                    idx_target_label = np.where(BH_nhood_label == target_label)[0]
                    label_frequencies = np.sum(BH_nhood_label == target_label) / sum(flag_coor_BH)
                    # calculate W                   
                    label_cov = np.cov(coor_BH_nhood[idx_target_label, :], rowvar=False)                   
                    within_label_cov += label_cov * label_frequencies
                    # calculate B
                    label_mean = coor_BH_nhood[idx_target_label, :].mean(axis=0)
                    between_label_mdiff += np.outer(label_mean - coor_BH_nhood_mean,
                                                  label_mean - coor_BH_nhood_mean) * label_frequencies               
                # W* = W^-.5
                # B* = W*BW*
                W_star = np.linalg.inv(sqrtm(within_label_cov) + 0.01*np.identity(n_dimension))
                B_star = W_star @ between_label_mdiff @ W_star
                identity_I = np.identity(n_dimension)
                # calculate psi
                local_matrix_psi = W_star @ (B_star + epsilon * identity_I) @ W_star
                difference = coor_BH - coor_unknown_i
                # calculate D(j, i)
                distance_DANN_i = np.sum((difference @ local_matrix_psi).T * difference.T, axis=0)
                distance_DANN.append(distance_DANN_i)              
            self.distance_DANN = np.array(distance_DANN)                          
            return self.distance_DANN      
       
   
def get_init_labels_prob(self, epsilon=None, k=None, range_v=None, range_h=None):
    """ Calculate the probability of choosing labels for unknown elements in sampling initial field.
    
    Args:
        epsilon (float): the "softening" parameter for DANN.
        k (int): the number of neighbors of each label in the KHMD rule
        range_v, range_h (int): the range of the selected rectangle neighborhood.
    
    """
    
    if k is None:
        k = 5  
    else:
        k = k
    if self.phyDim == 2 :                         
        distance_DANN = self.get_DANN_distance(epsilon, range_v, range_h)   
        # Sort by DANN distance
        distances_sort_index = distance_DANN.argsort()
        distances_sort = distance_DANN[np.arange(distance_DANN.shape[0])[:,None],distances_sort_index]            
        if self.mask is None:
            num_not_fixed = self.num_not_fixed
        else:
            num_not_fixed = self.num_not_fixed - sum(~self.mask)
        
        comp_deck = np.einsum('ijk,i->ijk',
                              np.array([np.ones((num_not_fixed, self.num_fixed)), ]*self.n_labels),
                              np.arange(self.n_labels))
        labels_fixed = np.array([self.init_field[self.fixed_flag],]*num_not_fixed)
        # sort the distances between all borehole elements and i
        labels_fixed_sort = labels_fixed[np.arange(labels_fixed.shape[0])[:,None],distances_sort_index]
        labels_fixed_sort_desk = np.array([labels_fixed_sort,]*self.n_labels)
        diff = comp_deck - labels_fixed_sort_desk
        temp = np.zeros_like(diff, dtype=float)
        temp[diff == 0] = 1
        label_flag = np.array(temp).astype(bool)            
        distances_elem_sort_desk = np.array([distances_sort,]*self.n_labels)
        # sort the distances between borehole elements having the same label and i                       
        self.distances_elem_sort_desk_for_label = []
        for i in range(self.n_labels):
            a = distances_elem_sort_desk[i,:,:][label_flag[i,:,:]].reshape(((num_not_fixed, sum(label_flag[i,0,:]))))[:,0:k]   
            self.distances_elem_sort_desk_for_label.append(a)
        distances_elem_sort_desk_for_label = np.array(self.distances_elem_sort_desk_for_label) 
        zero_flag = distances_elem_sort_desk_for_label==0
        distances_elem_sort_desk_for_label[zero_flag] = 10**(-100)
        # calculate KHMD
        reciprocal_distance_re_nearest = 1 / distances_elem_sort_desk_for_label
        sum_reciprocal_distance = np.sum(reciprocal_distance_re_nearest, axis=2).T
        dist_KHMD = k / sum_reciprocal_distance
        # calculate the probability that the unknown elements choose different labels 
        dist_for_label_exp = np.exp(-dist_KHMD)
        dist_for_label_exp_sum = np.tile(np.sum(dist_for_label_exp, axis=1).reshape((num_not_fixed,1)),
                                         self.n_labels)
        labels_prob_init = dist_for_label_exp / dist_for_label_exp_sum
        
        return dist_KHMD, labels_prob_init
    
        
def pseudocolor(physic_shp, stencil=None):
    """Graph coloring based on the physical dimensions for independent labels draw. This function is the basis for
       parallel Gibbs sampling.

    Args:
        physic_shp (:obj:`tuple` of int): physical shape of the data structure.
        stencil: the type of neighborhood

    Returns:
        1-DIMENSIONAL:
        return  color: graph color vector for parallel Gibbs sampler
        2-DIMENSIONAL:
        return  color: graph color vector for parallel Gibbs sampler
    """

    # Get the dimension of the physical space
    dim = len(physic_shp)
    # ************************************************************************************************
    # 1-DIMENSIONAL
    if dim == 1:
        i_w = np.arange(0, physic_shp[0], step=2)
        i_b = np.arange(1, physic_shp[0], step=2)

        return [i_w, i_b]

    # ************************************************************************************************
    # 2-DIMENSIONAL
    elif dim == 2:
        if (stencil is None) or (stencil == "8p"):
            # use 8 stamp as default, resulting in 4 colors
            num_of_colors = 4
            # color image
            colored_image = np.tile(np.kron([[0, 1], [2, 3]] * int(ceil(physic_shp[0] / 2)), np.ones((1, 1))),
                                    ceil(physic_shp[1] / 2))[0:physic_shp[0], 0:physic_shp[1]]
            colored_flat = colored_image.reshape(physic_shp[0] * physic_shp[1])

            # initialize storage array
            ci = []
            for c in range(num_of_colors):
                x = np.where(colored_flat == c)[0]
                ci.append(x)
            return ci
        else:
            raise Exception("In 2D space the stamp parameter needs to be either None or '8p' (defaults to 8p)")

    # ************************************************************************************************
    # 3-DIMENSIONAL
    elif dim == 3:
        if (stencil is None) or (stencil == "16p"):
            # use 16 stamp as default, resulting in 8 colors
            num_of_colors = 8
            # for even index of 0 axis in color image 
            colored_image_even = np.tile(np.kron([[0, 1], [2, 3]] * int(ceil(physic_shp[1] / 2)), np.ones((1, 1))),
                                    ceil(physic_shp[2] / 2))[0:physic_shp[1], 0:physic_shp[2]]
            # for odd index of 0 axis in color image 
            colored_image_odd = np.tile(np.kron([[4, 5], [6, 7]] * int(ceil(physic_shp[1] / 2)), np.ones((1, 1))),
                                    ceil(physic_shp[2] / 2))[0:physic_shp[1], 0:physic_shp[2]]
            colored_image_3D = np.ones(physic_shp)
            # acquire color image
            colored_image_3D[0:physic_shp[0]:2,:,:] = colored_image_even
            colored_image_3D[1:physic_shp[0]:2,:,:] = colored_image_odd
            colored_image_3D_flat = colored_image_3D.flatten()
            # initialize storage array
            ci = []
            for c in range(num_of_colors):
                x = np.where(colored_image_3D_flat == c)[0]
                ci.append(x)
            return ci
        else:
            raise Exception("In 3D space the stamp parameter needs to be either None or '16p' (defaults to 16p)")
    else:
        raise Exception("Data format appears to be wrong (neither 1-, 2- or 3-D).")





def draw_labels_vect(labels_prob):
    """Vectorized draw of the label for each elements respective labels probability.

    Args:
        labels_prob (:obj:`np.ndarray`): (n_elements x n_labels) ndarray containing the element-specific labels
            probabilities for each element.

    Returns:
        :obj:`np.array` : Flat array containing the newly drawn labels for each element.
    """

    # cumsum labels probabilities for each element
    p = np.cumsum(labels_prob, axis=1)
    p = np.concatenate((np.zeros((p.shape[0], 1)), p), axis=1)

    # draw a random number between 0 and 1 for each element
    #r = np.array([np.random.rand(p.shape[0])]).T
    r = np.random.rand(p.shape[0], 1)

    # compare and count to get label
    temp = np.sum(np.greater_equal((r @ np.ones((1, p.shape[1])) - p), 0), axis=1) - 1
    return temp


def evaluate(log_target_prop, log_target_prev):
    ratio = np.exp(np.longfloat(log_target_prop - log_target_prev))
    ratio = min(ratio, 1)

    if (ratio == 1) or (np.random.uniform() < ratio):
        return True, ratio  # if accepted

    else:
        return False, ratio  # if rejected


def estimator(betas, start_iter):
    if len(betas) > 1:
        betas = np.array(betas)
        if len(betas.shape) == 1:
            beta_est = np.mean(betas[start_iter:])
            beta_std = np.std(betas[start_iter:])
            beta_cov = None
        else:
            beta_est = np.mean(betas[start_iter:, :], axis=0)
            beta_std = np.std(betas[start_iter:, :], axis=0)
            beta_cov = np.cov(betas[start_iter:, :].T)
    elif len(betas) == 1:
        betas = np.array(betas)
        if len(betas.shape) == 1:
            beta_est = betas
            beta_std = 0
            beta_cov = None
        elif len(betas.shape) == 2:
            beta_est = np.array(betas[0])
            beta_std = np.zeros(beta_est.shape)
            beta_cov = None
        else:
            raise Exception("The shape of betas is not supported")
    else:
        raise Exception("betas is empty")
    return beta_est, beta_std, beta_cov
