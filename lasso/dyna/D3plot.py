
import os
import io
import re
import numpy as np
import logging

import ctypes

from base64 import b64encode
from zipfile import ZipFile, ZIP_DEFLATED
import tempfile
import webbrowser

from ..io.BinaryBuffer import BinaryBuffer
from .ArrayType import ArrayType
from ..plotting import plot_shell_mesh
arraytype = ArrayType


class D3plot:
    '''Class used to read LS-Dyna d3plots
    '''

    def __init__(self, filepath: str=None, use_femzip: bool=False):
        '''Constructor for a D3plot

        Parameters
        ----------
        filepath : `str`
            path to a d3plot file
        use_femzip : `bool`
            whether to use femzip decompression

        Notes
        -----
            If dyna wrote multiple files for several states,
            only give the path to the first file.
        '''
        super(D3plot, self).__init__()

        self._arrays = {}
        self._header = {}

        if filepath and not use_femzip:
            self.bb = BinaryBuffer(filepath)
            self.bb_states = None
        elif filepath and use_femzip:
            # return nstate and size of part title for header
            n_states, part_titles_size = self._read_femzipped_state(filepath)
        else:
            self.bb = None
            self.bb_states = None

        # try to determine precision
        self.charsize = 1
        self.wordsize, self.itype, self.ftype = self._determine_wordsize()

        self.geometry_section_size = 0

        # read header
        self._read_header()

        # add femzip params to header dict
        self.header["use_femzip"] = use_femzip
        if use_femzip:
            self.header["femzip"] = {
                "n_states": n_states,
                "part_titles_size": part_titles_size
            }

        # read material section
        self._read_material_section()

        # read fluid material data
        self._read_fluid_material_data()

        # SPH element data flags
        self._read_sph_element_data_flags()

        # Particle Data
        self._read_particle_data()

        # Geometry Data
        self._read_geometry_data()

        # User Material, Node, Blabla IDs
        self._read_user_ids()

        # Rigid Body Description
        self._read_rigid_body_description()

        # Adapted Element Parent List
        # TODO

        # Smooth Particle Hydrodynamcis Node and Material list
        self._read_sph_node_and_material_list()

        # Particle Geometry Data
        self._read_particle_geometry_data()

        # Rigid Road Surface Data
        self._read_rigid_road_surface()

        # Connectivity for weirdo elements
        # 10 Node Tetra
        # 8 Node Shell
        # 20 Node Solid
        # 27 Node Solid
        self._read_extra_node_connectivity()

        # Header Part & Contact Interface Titles
        self._read_header_part_contact_interface_titles()

        # correct offset if required
        # self._correct_file_offset()

        # Extra Data Types (for multi solver output)
        # TODO

        # State Data
        self._read_states()

    @property
    def arrays(self) -> dict:
        ''' Dictionary holding all d3plot arrays

        Notes
        -----
            The corresponding keys of the dictionary can
            also be found in `lasso.dyna.ArrayTypes`, which
            helps with IDE integration and code safety.

        Examples
        --------
            >>> d3plot = D3plot("some/path/to/d3plot")
            >>> d3plot.arrays.keys()
            dict_keys(['irbtyp', 'node_coordinates', ...])
            >>> # The following is good coding practice
            >>> import lasso.dyna.ArrayTypes.ArrayTypes as atypes
            >>> d3plot.arrays[atypes.node_displacmeent].shape
        '''
        return self._arrays

    @arrays.setter
    def arrays(self, array_dict: dict):
        assert(isinstance(array_dict, dict))
        self._arrays = array_dict

    @property
    def header(self):
        ''' Dictionary holding all d3plot header information

        Notes
        -----
            The header contains many informations such as number
            of elements, etc. The names are the original dyna names,
            thus sometimes confusing ... but that's dyna.

        Examples
        --------
            >>> d3plot = D3plot("some/path/to/d3plot")
            >>> d3plot.header
            dict_keys(['title', 'runtime', 'filetype', 'source_version', ...])
            >>> # number of shells
            >>> d3plot.header['nel4']
            85624
        '''
        return self._header

    @header.setter
    def header(self, new_header):
        assert(isinstance(new_header, dict))
        self._header = new_header

    def _is_end_of_file_marker(self, position: int) -> bool:
        ''' Check for the dyna eof marker at a certain position

        Notes
        -----
            The end of file marker is represented by a floating point
            number with the value -999999 (single precision hex: F02374C9,
            double precision hex: 000000007E842EC1).
        '''
        return self.bb.read_number(position, self.ftype) == self.ftype(-999999)

    def _correct_file_offset(self):
        ''' Correct the position in the bytes

        Notes
        -----
            LS-Dyna writes its files zero padded at a size of
            512 words in block size. There might be a lot of
            unused trailing data in the rear we need to skip
            in order to get to the next useful data block.
        '''

        if not self.bb:
            return

        block_count = len(self.bb) // (512*self.wordsize)

        # Warning!
        # Resets the block count!
        self.geometry_section_size = (block_count+1)*512*self.wordsize

    def _read_femzipped_state(self, file_path: str=None):
        ''' This routine reads the data for state information

        Parameters
        ----------
        filpath : string
            path to femzipped file

        Returns
        -------
        bb_states : BinaryBuffer
            New binary buffer with all states perfectly linear in memory
        n_states : int
            Number of states to be expected

        Notes
        -----
            State data is expected directly behind geometry data
            Unfortunately data is spread across multiple files.
            One file could contain geometry and state data but states
            may also be littered accross several files. This would
            not be an issue, if dyna would not always write in blocks
            of 512 words of memory, leaving zero byte padding blocks
            at the end of files. These need to be removed and/or taken
            care of.
        '''

        if not file_path:
            return

        # load lib containing the femzip routines
        if os.name == "nt":
            femzip_lib = ctypes.CDLL(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), 'femzip_buffer.dll'))
        else:
            femzip_lib = ctypes.CDLL(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), 'femzip_buffer.so'))

        # create ctype data types
        string_encode = file_path.encode('utf-8')

        # array holding buffer shape [n_timesteps, size_geo, size_state, size_part_titles]
        buffer_sizes_type = ctypes.c_int64 * 4

        # create the arrays with defined data types
        buffer_sizes = buffer_sizes_type()

        # compute the buffer dimensions using femzip
        femzip_lib.get_buffer_dimensions(
            ctypes.create_string_buffer(string_encode), buffer_sizes)

        # check that the routine succeeded
        assert(buffer_sizes[0] > 0)

        # create the binary array data types for geometry and states
        # geo_buffer also holds byte array of part titles at the end
        buffer_geom_type = ctypes.c_int * (buffer_sizes[1] + buffer_sizes[3])
        buffer_state_type = ctypes.c_int * (buffer_sizes[2]) * buffer_sizes[0]

        # create the empty arrays
        buffer_geo = buffer_geom_type()
        buffer_state = buffer_state_type()

        femzip_lib.get_buffer(ctypes.create_string_buffer(
            string_encode), buffer_sizes, buffer_geo, buffer_state)

        # save
        self.bb_states = BinaryBuffer()
        self.bb_states.memoryview = memoryview(buffer_state).cast('B')

        self.bb = BinaryBuffer()
        self.bb.memoryview = memoryview(buffer_geo).cast('B')

        return buffer_sizes[0], buffer_sizes[3]

    def _read_header(self):
        '''Reads the header of a d3plot

        Returns
        -------
        header : dict
            header data of the d3plot
        '''

        if not self.bb:
            self.header = {}
            return

        header_words = {
            "title": [0*self.wordsize, self.charsize, 10*self.wordsize],
            "runtime": [10*self.wordsize, self.itype],
            "filetype": [11*self.wordsize, self.itype],
            "source_version": [12*self.wordsize, self.itype],
            "release_version": [13*self.wordsize, self.charsize, 1*self.wordsize],
            "version": [14*self.wordsize, self.ftype],
            "ndim": [15*self.wordsize, self.itype],
            "numnp": [16*self.wordsize, self.itype],
            "icode": [17*self.wordsize, self.itype],
            "nglbv": [18*self.wordsize, self.itype],
            "it": [19*self.wordsize, self.itype],
            "iu": [20*self.wordsize, self.itype],
            "iv": [21*self.wordsize, self.itype],
            "ia": [22*self.wordsize, self.itype],
            "nel8": [23*self.wordsize, self.itype],
            "nummat8": [24*self.wordsize, self.itype],
            "numds": [25*self.wordsize, self.itype],
            "numst": [26*self.wordsize, self.itype],
            "nv3d": [27*self.wordsize, self.itype],
            "nel2": [28*self.wordsize, self.itype],
            "nummat2": [29*self.wordsize, self.itype],
            "nv1d": [30*self.wordsize, self.itype],
            "nel4": [31*self.wordsize, self.itype],
            "nummat4": [32*self.wordsize, self.itype],
            "nv2d": [33*self.wordsize, self.itype],
            "neiph": [34*self.wordsize, self.itype],
            "neips": [35*self.wordsize, self.itype],
            "maxint": [36*self.wordsize, self.itype],
            "nmsph": [37*self.wordsize, self.itype],
            "ngpsph": [38*self.wordsize, self.itype],
            "narbs": [39*self.wordsize, self.itype],
            "nelth": [40*self.wordsize, self.itype],
            "nummatt": [41*self.wordsize, self.itype],
            "nv3dt": [42*self.wordsize, self.itype],
            "ioshl1": [43*self.wordsize, self.itype],
            "ioshl2": [44*self.wordsize, self.itype],
            "ioshl3": [45*self.wordsize, self.itype],
            "ioshl4": [46*self.wordsize, self.itype],
            "ialemat": [47*self.wordsize, self.itype],
            "ncfdv1": [48*self.wordsize, self.itype],
            "ncfdv2": [49*self.wordsize, self.itype],
            "nadapt": [50*self.wordsize, self.itype],
            "nmmat": [51*self.wordsize, self.itype],
            "numfluid": [52*self.wordsize, self.itype],
            "inn": [53*self.wordsize, self.itype],
            "npefg": [54*self.wordsize, self.itype],
            "nel48": [55*self.wordsize, self.itype],
            "idtdt": [56*self.wordsize, self.itype],
            "extra": [57*self.wordsize, self.itype],
        }

        header_extra_words = {
            "nel20": [64*self.wordsize, self.itype],
            "nt3d": [65*self.wordsize, self.itype],
            "nel27": [66*self.wordsize, self.itype],
            "neipb": [67*self.wordsize, self.itype],
        }

        logging.debug("_read_header start at byte {}".format(
            self.geometry_section_size))

        # read data
        self.header = self._read_words(header_words)

        if self.header["extra"] != 0:
            self._read_words(header_extra_words, self.header)
        else:
            for name in header_extra_words.keys():
                self.header[name] = 0

        # PARSE HEADER

        # filetype
        if self.header["filetype"] > 1000:
            self.header["filetype"] -= 1000
            self.header["external_numbers_dtype"] = np.int64
        else:
            self.header["external_numbers_dtype"] = np.int32

        if self.header["filetype"] != 1 and self.header["filetype"] != 5:
            raise RuntimeError(
                "Wrong filetype %d != 1 (d3plot) or 5 (d3part) in header" % self.header["filetype"])

        # ndim
        if self.header["ndim"] == 5 or self.header["ndim"] == 7:
            self.header["mattyp"] = 1
            self.header["ndim"] = 3
            self.header['has_rigid_road_surface'] = False
            self.header['has_rigid_body_data'] = False
            # self.header['elem_connectivity_unpacked'] = True
        elif self.header["ndim"] == 4:
            self.header["mattyp"] = 0
            self.header["ndim"] = 3
            self.header['has_rigid_road_surface'] = False
            self.header['has_rigid_body_data'] = False
            # self.header['elem_connectivity_unpacked'] = True
        elif self.header["ndim"] > 5 and self.header["ndim"] < 8:
            self.header["mattyp"] = 0
            self.header['ndim'] = 3
            self.header['has_rigid_road_surface'] = True
            self.header['has_rigid_body_data'] = False
        elif self.header['ndim'] == 8 or self.header['ndim'] == 9:
            self.header["mattyp"] = 0
            self.header['ndim'] = 3
            if self.header['ndim'] == 9:
                self.header['has_rigid_road_surface'] = True
                self.header['has_reduced_rigid_body_data'] = True
            else:
                self.header['has_rigid_road_surface'] = False
                self.header['has_reduced_rigid_body_data'] = False
            self.header['has_rigid_body_data'] = True
        else:
            raise RuntimeError(
                "Invalid header entry ndim: %d" % self.header["ndim"])

        # temperature
        if self.header["it"] != 0:
            self.header["has_temperatures"] = (self.header["it"] % 10) != 0
        else:
            self.header["has_temperatures"] = False

        # mass scaling
        if self.header["it"] >= 10:
            self.header["has_mass_scaling"] = (self.header["it"] / 10) == 1
        else:
            self.header["has_mass_scaling"] = False

        # 10 node elems
        if self.header["nel8"] < 0:
            self.header["nel8"] = abs(self.header["nel8"])
            self.header["has_nel10"] = True
        else:
            self.header["has_nel10"] = False

        # integration points
        if self.header["maxint"] >= 0:
            self.header["mdlopt"] = 0
        if self.header["maxint"] < -10000:
            self.header["mdlopt"] = 2
            self.header["maxint"] = abs(self.header["maxint"]) - 10000
        if self.header["maxint"] < 0:
            self.header["mdlopt"] = 1
            self.header["maxint"] = abs(self.header["maxint"])

        # shell data
        self.header["ioshl1"] = 1 if self.header["ioshl1"] == 1000 else 0
        self.header["ioshl2"] = 1 if self.header["ioshl2"] == 1000 else 0
        self.header["ioshl3"] = 1 if self.header["ioshl3"] == 1000 else 0
        self.header["ioshl4"] = 1 if self.header["ioshl4"] == 1000 else 0

        # istrn
        # took me like 1000 years to figure this out ....
        if self.header["idtdt"] > 100:
            self.header["istrn"] = self.header["idtdt"] % 10000
        else:
            if self.header["nv2d"] > 0:
                if (self.header["nv2d"]
                    - self.header["maxint"] *
                        (6*self.header["ioshl1"] +
                         self.header["ioshl2"]+self.header["neips"])
                        - 8 * self.header["ioshl3"]
                        - 4*self.header["ioshl4"]) > 1:

                    self.header["istrn"] = 1
                else:
                    self.header["istrn"] = 0

            elif self.header["nelth"] > 0:
                if (self.header["nv3dt"]
                        - self.header["maxint"] * (6*self.header["ioshl1"]+self.header["ioshl2"]+self.header["neips"])) > 1:

                    self.header["istrn"] = 1
                else:
                    self.header["istrn"] = 0
            else:
                self.header["istrn"] = 0

        # internal energy
        shell_vars_behind_layers = (self.header["nv2d"] - self.header["maxint"] * (
            6*self.header["ioshl1"]+self.header["ioshl2"]+self.header["neips"])
            + 8 * self.header["ioshl3"] + 4*self.header["ioshl4"])

        if self.header["istrn"] == 0:

            if shell_vars_behind_layers > 1 and shell_vars_behind_layers < 6:
                self.header["has_internal_energy"] = True
            else:
                self.header["has_internal_energy"] = False

        elif self.header["istrn"] == 1:

            if shell_vars_behind_layers > 12:
                self.header["has_internal_energy"] = True
            else:
                self.header["has_internal_energy"] = False

        if "has_internal_energy" not in self.header:
            self.header["has_internal_energy"] = False

        # CHECKS
        if self.header["ncfdv1"] == 67108864:
            raise RuntimeError("Can not handle CFD Multi-Solver data. ")

        self.geometry_section_size = 64 * \
            (1 + self.header['extra']) * self.wordsize

        logging.debug("_read_header end at byte {}".format(
            self.geometry_section_size))

    def _read_material_section(self):
        ''' This function reads the material type section
        '''

        if not self.bb:
            return

        logging.debug("_read_material_section start at byte {}".format(
            self.geometry_section_size))

        memory_positions = {}

        # Material Type Data
        #
        # "This data is required because those shell elements
        # that are in a rigid body have no element data output
        # in the state data section."
        #
        # "The normal length of the shell element state data is:
        # NEL4 * NV2D, when the MATTYP flag is set the length is:
        # (NEL4 – NUMRBE) * NV2D. When reading the shell element data,
        # the material number must be check against IRBRTYP list to
        # find the element’s material type. If the type = 20, then
        # all the values for the element to zero." (Manual 03.2016)

        if self.header["mattyp"] == 0:
            self.header["numrbe"] = 0
            # irbtyp empty
        else:
            position = self.geometry_section_size

            self.header["numrbe"] = self.bb.read_number(
                position, self.itype)
            position += self.wordsize

            test_nummat = self.bb.read_number(
                position, self.itype)
            position += self.wordsize

            if test_nummat != self.header["nmmat"]:
                raise RuntimeError("nmmat (header) != nmmat (material type data): %d != %d" % (
                    self.header["nmmat"], test_nummat))

            self.arrays["irbtyp"] = self.bb.read_ndarray(position,
                                                         self.header["nmmat"]*self.wordsize, 1, self.itype)
            position += self.header["nmmat"]*self.wordsize

            self.geometry_section_size = position
            logging.debug("_read_material_section end at byte {}".format(
                self.geometry_section_size))

    def _read_fluid_material_data(self):
        ''' Read the fluid material data
        '''

        if not self.bb:
            return

        if self.header["ialemat"] == 0:
            return

        logging.debug("_read_fluid_material_data start at byte {}".format(
            self.geometry_section_size))

        position = self.geometry_section_size

        # Fluid Material Data
        array_length = self.header["ialemat"]*self.wordsize
        self.arrays[arraytype.ale_material_ids] = \
            self.bb.read_ndarray(position, array_length, 1, self.itype)
        position += array_length

        # remember position
        self.geometry_section_size = position
        logging.debug("_read_fluid_material_data end at byte {}".format(
            self.geometry_section_size))

    def _read_sph_element_data_flags(self):
        ''' Read the sph element data flags
        '''

        if not self.bb:
            return

        if self.header["nmsph"] == 0:
            self.header["num_sph_vars"] = 0
            return

        logging.debug("_read_sph_element_data_flags start at byte {}".format(
            self.geometry_section_size))

        position = self.geometry_section_size

        sph_element_data_words = {
            "isphfg1": (position, self.itype),
            "isphfg2": (position+1*self.wordsize, self.itype),
            "isphfg3": (position+2*self.wordsize, self.itype),
            "isphfg4": (position+3*self.wordsize, self.itype),
            "isphfg5": (position+4*self.wordsize, self.itype),
            "isphfg6": (position+5*self.wordsize, self.itype),
            "isphfg7": (position+6*self.wordsize, self.itype),
            "isphfg8": (position+7*self.wordsize, self.itype),
            "isphfg9": (position+8*self.wordsize, self.itype),
            "isphfg10": (position+9*self.wordsize, self.itype),
            "isphfg11": (position+10*self.wordsize, self.itype),
        }

        sph_element_data_header = self._read_words(sph_element_data_words)
        self.header.update(sph_element_data_header)

        if self.header["isphfg1"] != 11:
            raise RuntimeError("Detected inconsistency: isphfg = " +
                               str(self.header["isphfg1"]) + " but must be 11.")

        self.header["isphfg_total"] = \
            self.header["isphfg2"] \
            + self.header["isphfg3"] \
            + self.header["isphfg4"] \
            + self.header["isphfg5"] \
            + self.header["isphfg6"] \
            + self.header["isphfg7"] \
            + self.header["isphfg8"] \
            + self.header["isphfg9"] \
            + self.header["isphfg10"] \
            + self.header["isphfg11"]

        # ask the manual ...
        self.header["num_sph_vars"] = 1 + self.header["isphfg_total"]

        self.geometry_section_size += self.header["isphfg1"]*self.wordsize
        logging.debug("_read_sph_element_data_flags end at byte {}".format(
            self.geometry_section_size))

    def _read_particle_data(self):
        ''' Read the geometry section for particle data (airbags)
        '''

        if not self.bb:
            return

        if self.header['npefg'] <= 0 or self.header['npefg'] > 10000000:
            return

        logging.debug("_read_particle_data start at byte {}".format(
            self.geometry_section_size))

        position = self.geometry_section_size

        airbag_header = {
            # number of airbags
            'npartgas': self.header['npefg'] % 1000,
            # ?
            'subver': self.header['npefg'] / 1000
        }

        particle_geometry_data_words = {
            # number of geometry variables
            'ngeom': (position, self.itype),
            # number of state variables
            'nvar': (position+1*self.wordsize, self.itype),
            # number of particles
            'npart': (position+2*self.wordsize, self.itype),
            # number of state geometry variables
            'nstgeom': (position+3*self.wordsize, self.itype)
        }

        self._read_words(particle_geometry_data_words, airbag_header)
        position += 4*self.wordsize

        if airbag_header['subver'] == 4:
            # number of chambers
            airbag_header['nchamber'] = self.bb.read_number(
                position, self.itype)
            position += self.wordsize

        airbag_header['nlist'] = airbag_header['ngeom'] + \
            airbag_header['nvar'] + airbag_header['nstgeom']

        # airbag variable types
        airbag_variable_types = []
        for i_variable in range(airbag_header['nlist']):

            # find out var type for variables
            if self.bb.read_number(position+i_variable*self.wordsize, self.itype):
                airbag_variable_types.append(self.itype)
            else:
                airbag_variable_types.append(self.ftype)

        airbag_header['variable_types'] = airbag_variable_types
        position += airbag_header['nlist']*self.wordsize

        # airbag variable names
        airbag_variable_names = []
        var_width = 8

        for i_variable in range(airbag_header['nlist']):
            name = self.bb.read_text(position+(i_variable+var_width)
                                     * self.wordsize, var_width*self.wordsize)

            airbag_variable_names.append(name)

        airbag_header['variable_names'] = airbag_variable_names
        position += airbag_header['nlist']*var_width*self.wordsize

        # save airbag header
        self.header["airbag"] = airbag_header

        # update position marker
        self.geometry_section_size = position
        logging.debug("_read_particle_data start at byte {}".format(
            self.geometry_section_size))

    def _read_geometry_data(self):
        ''' Read the data from the geometry section
        '''

        if not self.bb:
            return

        logging.debug("_read_geometry_data start at byte {}".format(
            self.geometry_section_size))

        # not sure but I think never used by LS-Dyna
        # anyway needs to be detected in the header and not here,
        # though it is mentioned in this section of the database manual
        #
        # is_packed = True if self.header['ndim'] == 3 else False
        # if is_packed:
        #     raise RuntimeError("Can not deal with packed geometry data (ndim == {}).".format(self.header['ndim']))

        position = self.geometry_section_size

        # node coords
        section_word_length = self.header['ndim']*self.header['numnp']
        node_coordinates = self.bb.read_ndarray(
            position, section_word_length*self.wordsize, 1, self.ftype).reshape((self.header['numnp'], self.header['ndim']))
        self.arrays[arraytype.node_coordinates] = node_coordinates
        position += section_word_length*self.wordsize

        # solid data
        section_word_length = 9*self.header['nel8']
        elem_solid_data = self.bb.read_ndarray(
            position, section_word_length*self.wordsize, 1, self.itype).reshape((self.header['nel8'], 9))
        solid_connectivity = elem_solid_data[:, :8]
        solid_material_types = elem_solid_data[:, 8]
        self.arrays[arraytype.element_solid_material_types] = solid_material_types
        self.arrays[arraytype.element_solid_node_indexes] = solid_connectivity
        position += section_word_length*self.wordsize

        # 8 node thick shells
        section_word_length = 9*self.header['nelth']
        elem_tshell_data = self.bb.read_ndarray(
            position, section_word_length*self.wordsize, 1, self.itype).reshape((self.header['nelth'], 9))
        self.arrays[arraytype.element_tshell_node_indexes] = elem_tshell_data[:, :8]
        self.arrays[arraytype.element_tshell_material_types] = elem_tshell_data[:, 8]
        position += section_word_length*self.wordsize

        # beams
        section_word_length = 6*self.header['nel2']
        elem_beam_data = self.bb.read_ndarray(
            position, section_word_length*self.wordsize, 1, self.itype).reshape((self.header['nel2'], 6))
        self.arrays[arraytype.element_beam_material_types] = elem_beam_data[:, 5]
        self.arrays[arraytype.element_beam_node_indexes] = elem_beam_data[:, :5]
        position += section_word_length*self.wordsize

        # shells
        section_word_length = 5*self.header['nel4']
        elem_shell_data = self.bb.read_ndarray(
            position, section_word_length*self.wordsize, 1, self.itype).reshape((self.header['nel4'], 5))
        self.arrays[arraytype.element_shell_node_indexes] = elem_shell_data[:, :4]
        self.arrays[arraytype.element_shell_material_types] = elem_shell_data[:, 4]
        position += section_word_length*self.wordsize

        # update word position
        self.geometry_section_size = position

        logging.debug("_read_geometry_data end at byte {}".format(
            self.geometry_section_size))

    def _read_user_ids(self):

        if not self.bb:
            return

        if self.header['narbs'] == 0:
            return

        logging.debug("_read_user_ids start at byte {}".format(
            self.geometry_section_size))

        position = self.geometry_section_size

        numbering_words = {
            'nsort':  (position, self.itype),
            'nsrh':   (position+1*self.wordsize, self.itype),
            'nsrb':   (position+2*self.wordsize, self.itype),
            'nsrs':   (position+3*self.wordsize, self.itype),
            'nsrt':   (position+4*self.wordsize, self.itype),
            'nsortd': (position+5*self.wordsize, self.itype),
            'nsrhd':  (position+6*self.wordsize, self.itype),
            'nsrbd':  (position+7*self.wordsize, self.itype),
            'nsrsd':  (position+8*self.wordsize, self.itype),
            'nsrtd':  (position+9*self.wordsize, self.itype),
        }

        extra_numbering_words = {
            'nsrma':  (position+10*self.wordsize, self.itype),
            'nsrmu':  (position+11*self.wordsize, self.itype),
            'nsrmp':  (position+12*self.wordsize, self.itype),
            'nsrtm':  (position+13*self.wordsize, self.itype),
            'numrbs': (position+14*self.wordsize, self.itype),
            'nmmat':  (position+15*self.wordsize, self.itype),
        }

        numbering_header = self._read_words(numbering_words)
        position += len(numbering_words)*self.wordsize

        if numbering_header['nsort'] < 0:

            # read extra header
            material_section_extra_header = self._read_words(
                extra_numbering_words, numbering_header)
            position += len(extra_numbering_words)*self.wordsize

            # correct nsort
            numbering_header['nsort'] = abs(
                numbering_header['nsort'])
            numbering_header['arbitrary_material_numbering'] = True

        self.header['numbering_header'] = numbering_header

        # CHECK
        if numbering_header['nsrh'] != numbering_header['nsort'] + self.header['numnp']:
            raise RuntimeError("Encountered file inconsistency: nsrh != nsort + numnp ({} != {} + {})".format(
                numbering_header['nsrh'], numbering_header['nsort'], self.header['numnp']))

        if numbering_header['nsrb'] != numbering_header['nsrh'] + self.header['nel8']:
            raise RuntimeError("Encountered file inconsistency: nsrb != nsrh + nel8 ({} != {} + {})".format(
                numbering_header['nsrb'], numbering_header['nsrh'], self.header['nel8']))

        if numbering_header['nsrs'] != numbering_header['nsrb'] + self.header['nel2']:
            raise RuntimeError("Encountered file inconsistency: nsrs != nsrb + nel2 ({} != {} + {})".format(
                numbering_header['nsrs'], numbering_header['nsrb'], self.header['nel2']))

        if numbering_header['nsrt'] != numbering_header['nsrs'] + self.header['nel4']:
            raise RuntimeError("Encountered file inconsistency: nsrt != nsrs + nel4 ({} != {} + {})".format(
                numbering_header['nsrt'], numbering_header['nsrs'], self.header['nel4']))

        # node ids
        array_length = numbering_header['nsortd']*self.wordsize
        self.arrays[arraytype.node_ids] = self.bb.read_ndarray(
            position, array_length, 1, self.itype)
        position += array_length
        # solid ids
        array_length = self.header['nel8']*self.wordsize
        self.arrays[arraytype.element_solid_ids] = self.bb.read_ndarray(
            position, array_length, 1, self.itype)
        position += array_length
        # beam ids
        array_length = self.header['nel2']*self.wordsize
        self.arrays[arraytype.element_beam_ids] = self.bb.read_ndarray(
            position, array_length, 1, self.itype)
        position += array_length
        # shell ids
        array_length = self.header['nel4']*self.wordsize
        self.arrays[arraytype.element_shell_ids] = self.bb.read_ndarray(
            position, array_length, 1, self.itype)
        position += array_length
        # tshell ids
        array_length = self.header['nelth']*self.wordsize
        self.arrays[arraytype.element_tshell_ids] = self.bb.read_ndarray(
            position, array_length, 1, self.itype)
        position += array_length

        # part ids
        #
        # this makes no sense but materials are output three times at this section
        # but the length of the array (nmmat) is only output if nsort < 0. In
        # the other case the length is unknown ...
        #
        # Bugfix 1:
        # Indeed this is a little complicated: usually the file should contain
        # as many materials as in the input but somehow dyna generates a few
        # ghost materials itself and those are appended with a 0 ID. Therefore
        # the length should be sum of all nummat header vars but it's nmmat with:
        # nummat2+nummat4+nummat8+nummatt < nmmat. The difference are the ghost mats.
        # Took some time to find that out ... and I don't know why ...
        # oh and it is undocumented ...
        #
        # Bugfix 2:
        # The material arrays (three times) are always output, even if nsort < 0
        # which means they are not used. Quite confusing, especially since nmmat
        # is output in the main header and numbering header.
        #
        if 'nmmat' in numbering_header:

            if numbering_header["nmmat"] != self.header["nmmat"]:
                err_msg = "nmmat in the file header ({}) and in the numbering header ({}) are inconsistent."
                raise RuntimeError(err_msg.format(
                    self.header["nmmat"], numbering_header["nmmat"]))

            array_length = numbering_header['nmmat']*self.wordsize
            self.arrays[arraytype.part_ids] = self.bb.read_ndarray(
                position, numbering_header['nmmat']*self.wordsize, 1, self.itype)

            # sorted ids and sort indices are not needed

            position += 3*numbering_header["nmmat"]*self.wordsize

        else:
            position += 3*self.header["nmmat"]*self.wordsize

        # update position
        self.geometry_section_size = position
        logging.debug("_read_user_ids end at byte {}".format(
            self.geometry_section_size))

    def _read_rigid_body_description(self):
        ''' Read the rigid body description section  
        '''

        if not self.bb:
            return

        if not self.header["has_rigid_body_data"]:
            return

        logging.debug("_read_rigid_body_description start at byte {}".format(
            self.geometry_section_size))

        position = self.geometry_section_size

        rigid_body_description_header = {
            'nrigid': self.bb.read_number(position, self.itype)
        }
        position += self.wordsize

        rigid_bodies = []
        for i_rigid in range(rigid_body_description_header['nrigid']):

            rigid_body_info = {
                # rigid body part internal number
                'mrigid': self.bb.read_number(position, self.itype),
                # number of nodes in rigid body
                'numnodr': self.bb.read_number(position+self.wordsize,
                                               self.itype),
            }
            position += 2*self.wordsize

            # internal node number of rigid body
            array_length = rigid_body_info['numnodr']*self.wordsize
            rigid_body_info['noder'] = self.bb.read_ndarray(position,
                                                            array_length,
                                                            1, self.itype)
            position += array_length

            # number of active (non-rigid) nodes
            rigid_body_info['numnoda'] = self.bb.read_number(position,
                                                             self.itype)
            position += self.wordsize

            # internal node numbers of active nodes
            array_length = rigid_body_info['numnoda']*self.wordsize
            rigid_body_info['nodea'] = self.bb.read_ndarray(position,
                                                            array_length,
                                                            1,
                                                            self.itype)
            position += array_length

            # append to list
            rigid_bodies.append(rigid_body_info)

        # save rigid body info to header
        self.header['rigid_body_descriptions'] = rigid_bodies
        self.header["nrigid"] = rigid_body_description_header["nrigid"]

        # update position
        self.geometry_section_size = position
        logging.debug("_read_rigid_body_description end at byte {}".format(
            self.geometry_section_size))

    def _read_sph_node_and_material_list(self):
        ''' Read SPH node and material list
        '''

        if not self.bb:
            return

        if self.header['nmsph'] <= 0:
            return

        logging.debug("_read_sph_node_and_material_list start at byte {}".format(
            self.geometry_section_size))

        position = self.geometry_section_size

        # read info array
        array_length = self.header["nmsph"]*self.wordsize
        sph_node_matlist = self.bb.read_ndarray(
            position, array_length, 1, self.itype)
        position += array_length

        # save array
        self.arrays['sph_node_material_list'] = sph_node_matlist

        # update position
        self.geometry_section_size = position

        logging.debug("_read_sph_node_and_material_list end at byte {}".format(
            self.geometry_section_size))

    def _read_particle_geometry_data(self):
        ''' Read the particle geometry data
        '''

        if not self.bb:
            return

        if self.header['npefg'] <= 0:
            return

        logging.debug("_read_particle_geometry_data start at byte {}".format(
            self.geometry_section_size))

        airbag_header = self.header["airbag"]
        position = self.geometry_section_size

        # size of geometry section checking
        ngeom = airbag_header["ngeom"]
        if not ngeom in [4, 5]:
            raise RuntimeError(
                "variable ngeom in the airbag header must be 4 or 5.")

        # extract geometry as a single array
        array_length = ngeom*airbag_header["npartgas"]*self.wordsize
        particle_geom_data = self.bb.read_ndarray(
            position, array_length, 1, self.itype).reshape((airbag_header["npartgas"], ngeom))
        position += array_length

        # store arrays
        self.arrays["airbags_first_particle_id"] = particle_geom_data[:, 0]
        self.arrays["airbags_n_particles"] = particle_geom_data[:, 1]
        self.arrays["airbags_ids"] = particle_geom_data[:, 2]
        self.arrays["airbags_n_gas_mixtures"] = particle_geom_data[:, 3]
        if ngeom == 5:
            self.arrays["airbags_n_chambers"] = particle_geom_data[:, 4]

        # update position
        self.geometry_section_size = position

        logging.debug("_read_particle_geometry_data end at byte {}".format(
            self.geometry_section_size))

    def _read_rigid_road_surface(self):
        ''' Read rigid road surface data
        '''

        if not self.bb:
            return

        if not self.header['has_rigid_road_surface']:
            return

        logging.debug("_read_rigid_road_surface start at byte {}".format(
            self.geometry_section_size))

        position = self.geometry_section_size

        # read header
        rigid_road_surface_words = {
            'nnode': (position, self.itype),
            'nseg': (position+1*self.wordsize, self.itype),
            'nsurf': (position+2*self.wordsize, self.itype),
            'motion': (position+3*self.wordsize, self.itype),
        }

        rigid_road_header = self._read_words(rigid_road_surface_words)
        position += 4*self.wordsize

        # node ids
        array_length = rigid_road_header["nnode"]*self.wordsize
        rigid_road_node_ids = self.bb.read_ndarray(
            position, array_length, 1, self.itype)
        self.arrays[arraytype.rigid_road_node_ids] = rigid_road_node_ids
        position += array_length

        # node xyz
        array_length = rigid_road_header["nnode"]*3*self.wordsize
        rigid_road_node_coords = self.bb.read_ndarray(
            position, array_length, 1, self.ftype).reshape((rigid_road_header["nnode"], 3))
        self.arrays[arraytype.rigid_road_node_coordinates] = \
            rigid_road_node_coords
        position += array_length

        # read road segments
        # Warning: must be copied
        rigid_road_ids = np.empty(rigid_road_header["nsurf"], dtype=self.itype)
        rigid_road_nsegments = np.empty(
            rigid_road_header["nsurf"], dtype=self.itype)
        rigid_road_segment_node_ids = []
        # this array is created since the array database requires constant sized arrays
        # and we dump all segments into one array. In order to distinguish which segment
        # belongs to which road, this new array keeps track of it
        rigid_road_segment_road_id = []

        n_total_segments = 0
        for i_surf in range(rigid_road_header["nsurf"]):

            # surface id
            surf_id = self.bb.read_number(position, self.itype)
            position += self.wordsize
            rigid_road_ids[i_surf] = surf_id

            # number of segments of surface
            surf_nseg = self.bb.read_number(
                position+1*self.wordsize, self.itype)
            position += self.wordsize
            rigid_road_nsegments[i_surf] = surf_nseg

            # count total segments
            n_total_segments += surf_nseg

            # node ids of surface segments
            array_length = 4*surf_nseg*self.wordsize
            surf_segm_node_ids = self.bb.read_ndarray(position,
                                                      array_length,
                                                      1, self.itype)\
                .reshape((surf_nseg, 4))
            position += array_length
            rigid_road_segment_node_ids.append(surf_segm_node_ids)

            # remember road id for segments
            rigid_road_segment_road_id += [surf_id]*surf_nseg

        # save arrays
        self.arrays[arraytype.rigid_road_ids] = rigid_road_ids
        self.arrays[arraytype.rigid_road_n_segments] = rigid_road_nsegments
        self.arrays[arraytype.rigid_road_segment_node_ids] = np.concatenate(
            rigid_road_segment_node_ids)
        self.arrays[arraytype.rigid_road_segment_road_id] = np.asarray(
            rigid_road_segment_road_id)

        # update header
        self.header["rigid_road"] = rigid_road_header

        # update position
        self.geometry_section_size = position
        logging.debug("_read_rigid_road_surface end at byte {}".format(
            self.geometry_section_size))

    def _read_extra_node_connectivity(self):
        ''' Read the extra 2 nodes required for the 10 node tetras 
        '''

        if not self.bb:
            return

        logging.debug("_read_extra_node_connectivity start at byte {}".format(
            self.geometry_section_size))

        position = self.geometry_section_size

        # extra 2 node connectivity for 10 node tetrahedron elements
        if self.header["has_nel10"]:
            array_length = 2*self.header['nel8']*self.wordsize
            array = self.bb.read_ndarray(
                position, array_length, 1, self.itype).reshape((self.header['nel8'], 2))
            self.arrays["solid_tetra10_extra_node_ids"] = array
            position += array_length

        # extra 4 node connectivity for 8 node shell elements
        if self.header["nel48"] > 0:
            array_length = 5*self.header['nel48']*self.wordsize
            array = self.bb.read_ndarray(
                position, array_length, 1, self.itype).reshape((self.header['nel48'], 5))
            self.arrays["shell_node8_element_index"] = array[:, 0]
            self.arrays["shell_node8_extra_node_ids"] = array[:, 1:]
            position += array_length

        # extra 12 node connectivity for 20 node solid elements
        if "nel20" in self.header and self.header["nel20"] > 0:
            array_length = 13*self.header['nel20']*self.wordsize
            array = self.bb.read_ndarray(
                position, array_length, 1, self.itype).reshape((self.header['nel20'], 13))
            self.arrays["solid_node20_element_index"] = array[:, 0]
            self.arrays["solid_node20_extra_node_ids"] = array[:, 1:]
            position += array_length

        # extra 19 node connectivity for 27 node solid elements
        if "nel27" in self.header and self.header["nel27"] > 0:
            array_length = 20*self.header['nel27']*self.wordsize
            array = self.bb.read_ndarray(
                position, array_length, 1, self.itype).reshape((self.header['nel27'], 20))
            self.arrays["solid_node27_element_index"] = array[:, 0]
            self.arrays["solid_node27_extra_node_ids"] = array[:, 1:]
            position += array_length

        # update position
        self.geometry_section_size = position

        logging.debug("_read_extra_node_connectivity end at byte {}".format(
            self.geometry_section_size))

    def _read_header_part_contact_interface_titles(self):
        ''' Read the header for the parts, contacts and interfaces
        '''

        if not self.bb:
            return

        # filetype
        # 1 - d3plot
        # 4 - intfor
        # 5 - d3part
        if not self.header["filetype"] in [1, 4, 5]:
            return

        logging.debug("_read_header_part_contact_interface_titles start at byte {}".format(
            self.geometry_section_size))

        position = self.geometry_section_size

        # there is only output if there is an eof marker
        # at least I think I fixed such a bug in the past
        if not self._is_end_of_file_marker(position):
            return
        position += self.wordsize

        # section have types here according to what is inside
        ntypes = []
        self.header["ntype"] = ntypes

        # read first ntype
        current_ntype = self.bb.read_number(position, self.itype)

        while current_ntype in [90000, 90001, 90002, 90020]:

            # title output
            if current_ntype == 90000:

                ntypes.append(current_ntype)
                position += self.wordsize

                # Bugfix:
                # the titles are always 18*4 bytes, even if the wordsize
                # is 8 bytes for the entire file.
                titles_wordsize = 4

                array_length = 18*titles_wordsize
                self.header["title2"] = self.bb.read_text(
                    position, array_length)
                position += array_length

            # some title output
            elif current_ntype in [90001, 90002, 90020]:

                ntypes.append(current_ntype)
                position += self.wordsize

                # number of parts
                entry_count = self.bb.read_number(position, self.itype)
                position += self.wordsize

                # Bugfix:
                # the titles are always 18*4 bytes, even if the wordsize
                # is 8 bytes for the entire file.
                titles_wordsize = 4

                # part ids and corresponding titles
                array_type = np.dtype([("ids", self.itype),
                                       ("titles", "S"+str(18*titles_wordsize))])
                array_length = (self.wordsize+18*titles_wordsize)*entry_count
                tmp_arrays = self.bb.read_ndarray(
                    position, array_length, 1, array_type)
                position += array_length

                # save stuff
                if current_ntype == 90001:
                    self.arrays[arraytype.part_titles_ids] = tmp_arrays["ids"]
                    if not arraytype.part_ids in self.arrays.keys():
                        self.arrays[arraytype.part_ids] = tmp_arrays["ids"]
                    self.arrays[arraytype.part_titles] = tmp_arrays["titles"]
                elif current_ntype == 90002:
                    self.arrays[arraytype.contact_title_ids] = tmp_arrays["ids"]
                    self.arrays[arraytype.contact_titles] = tmp_arrays["titles"]
                elif current_ntype == 90020:
                    self.arrays["icfd_part_title_ids"] = tmp_arrays["ids"]
                    self.arrays["icfd_part_titles"] = tmp_arrays["titles"]

            # d3prop
            elif current_ntype == 90100:

                ntypes.append(current_ntype)
                position += self.wordsize

                # number of keywords
                self.header["nline"] = self.bb.read_number(
                    position, self.itype)
                position += self.wordsize

                # Bugfix:
                # the titles are always 18*4 bytes, even if the wordsize
                # is 8 bytes for the entire file.
                titles_wordsize = 4

                # keywords
                array_length = 20*titles_wordsize*self.header["nline"]
                d3prop_keywords = self.bb.read_ndarray(
                    position, array_length, 1, np.dtype("S"+str(titles_wordsize*20)))
                position += array_length

                # save
                self.arrays["d3prop_keywords"] = d3prop_keywords

            # not sure whether there is an eof file here
            # do not have a test file to check ...
            if self._is_end_of_file_marker(position):
                position += self.wordsize

            # next one
            current_ntype = self.bb.read_number(position, self.itype)

        # remember position
        self.geometry_section_size = position
        logging.debug("_read_header_part_contact_interface_titles end at byte {}".format(
            self.geometry_section_size))

    def _read_states(self, n_states: int=None):
        ''' Read the states of a d3plot*
        '''

        if not self.bb:
            self.header["n_timesteps"] = 0
            return

        logging.debug("-"*80)
        logging.debug("_read_states with geom offset {}".format(
            self.geometry_section_size))

        # (0) OFFSETS
        # timestep
        timestep_offset = 1*self.wordsize
        # global vars
        global_vars_offset = self.header["nglbv"]*self.wordsize
        # node vars
        n_node_vars = (self.header["iu"]+self.header["iv"]+self.header["ia"])\
            * self.header["ndim"]

        if self.header["it"] == 1:
            n_node_temp_vars = 1
        elif self.header["it"] == 2:
            n_node_temp_vars = 4
        elif self.header["it"] == 3:
            n_node_temp_vars = 6
        else:
            n_node_temp_vars = 0

        if self.header["has_mass_scaling"]:
            n_node_temp_vars += 1

        node_data_offset = (n_node_vars+n_node_temp_vars) * \
            self.header["numnp"]*self.wordsize
        # thermal shit
        therm_data_offset = self.header["nt3d"]*self.header["nel8"]*self.wordsize \
            if "nt3d" in self.header else 0
        # solids
        solid_offset = self.header["nel8"]*self.header["nv3d"]*self.wordsize
        # tshells
        tshell_offset = self.header["nelth"]*self.header["nv3dt"]*self.wordsize
        # bea,s
        beam_offset = self.header["nel2"]*self.header["nv1d"]*self.wordsize
        # shells
        shell_offset = (
            self.header["nel4"]-self.header["numrbe"])*self.header["nv2d"]*self.wordsize
        # Manual
        # "NOTE: This CFDDATA is no longer output by ls-dyna."
        cfd_data_offset = 0
        # sph
        sph_offset = self.header["nmsph"] * \
            self.header["num_sph_vars"]*self.wordsize
        # deleted nodes and elems ... or nothing
        elem_deletion_offset = 0
        if self.header["mdlopt"] == 1:
            elem_deletion_offset = self.header["numnp"]*self.wordsize
        elif self.header["mdlopt"] == 2:
            elem_deletion_offset = (self.header["nel2"]
                                    + self.header["nel4"]
                                    + self.header["nel8"]
                                    + self.header["nelth"])*self.wordsize
        else:
            err_msg = "Unexpected value of mdlop: {}, expected was 0, 1 or 2."
            raise RuntimeError(err_msg.format(self.header["mdlopt"]))
        # airbag particle offset
        if "airbag" in self.header:
            particle_state_offset = \
                (self.header["airbag"]["npartgas"]*self.header["airbag"]["nstgeom"]
                 + self.header["airbag"]["npart"]*self.header["airbag"]["nvar"]) \
                * self.wordsize
        else:
            particle_state_offset = 0
        # rigid road stuff whoever uses this
        road_surface_offset = self.header["rigid_road"]["nsurf"]*6*self.wordsize \
            if "rigid_road" in self.header else 0
        # rigid body motion data
        if self.header["has_rigid_body_data"]:
            n_rigids = self.header["nrigid"]
            n_rigid_vars = 12 if self.header["has_reduced_rigid_body_data"] else 24
            rigid_body_motion_offset = n_rigids*n_rigid_vars*self.wordsize
        else:
            rigid_body_motion_offset = 0
        # TODO
        extra_data_offset = 0

        bytes_per_state = timestep_offset \
            + global_vars_offset \
            + node_data_offset \
            + therm_data_offset \
            + solid_offset \
            + tshell_offset \
            + beam_offset \
            + shell_offset \
            + cfd_data_offset \
            + sph_offset \
            + elem_deletion_offset \
            + particle_state_offset \
            + road_surface_offset \
            + rigid_body_motion_offset \
            + extra_data_offset \

        # (1) READ STATE DATA
        if not self.header["use_femzip"]:
            self.bb_states, n_states = self._read_state_bytebuffer(
                bytes_per_state)
        else:
            n_states = self.header["femzip"]["n_states"]
            # end marker + part section size
            # + 1!! dont why, but one day we will
            if os.name == "posix":
                bytes_per_state += (self.header["femzip"]
                                    ["part_titles_size"]+1)*self.wordsize
            elif os.name == "nt":
                # end marker is always in here
                bytes_per_state += 1*self.wordsize

        logging.debug("bytes_per_state: {}".format(bytes_per_state))

        self.header["n_timesteps"] = n_states

        # state data as array
        array_length = int(n_states)*int(bytes_per_state)
        state_data = self.bb_states.read_ndarray(
            0, array_length, 1, self.ftype)
        state_data = state_data.reshape((n_states, -1))

        # here -1, also no idea why
        if os.name == "nt":
            var_index = 0
        else:
            var_index = 0 if not self.header["use_femzip"] \
                else (self.header["femzip"]["part_titles_size"] - 1)

        # global state header
        var_index = self._read_states_global_vars(state_data, var_index)

        # node data
        var_index = self._read_states_nodes(state_data, var_index)

        # thermal solid data
        var_index = self._read_states_solids_thermal(state_data, var_index)

        # cfddata was originally here

        # solids
        var_index = self._read_states_solids(state_data, var_index)

        # tshells
        var_index = self._read_states_thsell(state_data, var_index)

        # beams
        var_index = self._read_states_beams(state_data, var_index)

        # shells
        var_index = self._read_states_shell(state_data, var_index)

        # element and node deletion info
        var_index = self._read_states_is_alive(state_data, var_index)

        # sph
        var_index = self._read_states_sph(state_data, var_index)

        # airbag particle data
        var_index = self._read_states_airbags(state_data, var_index)

        # road surface data
        var_index = self._read_states_road_surfaces(state_data, var_index)

        # rigid body motion
        var_index = self._read_states_rigid_body_motion(state_data, var_index)

        # extra data
        # TODO

    def _read_states_global_vars(self, state_data: np.ndarray, var_index: int) -> int:
        ''' Read the global vars for the state

        Parameters
        ----------
        state_data : np.ndarray
            array with entire state data
        var_index : int
            variable index in the state data array

        Returns
        -------
        var_index : int
            updated variable index after reading the section
        '''

        if not self.bb_states:
            return 0

        logging.debug(
            "_read_states_global_vars start at var_index {}".format(var_index))

        n_states = state_data.shape[0]

        # global stuff
        self.arrays[arraytype.global_timesteps] = state_data[:, var_index + 0]
        self.arrays[arraytype.global_kinetic_energy] = state_data[:, var_index + 1]
        self.arrays[arraytype.global_internal_energy] = state_data[:, var_index + 2]
        self.arrays[arraytype.global_total_energy] = state_data[:, var_index + 3]
        self.arrays[arraytype.global_velocity] = state_data[:, var_index + 4:var_index + 7]\
            .transpose((1, 0))

        var_index += 7

        # part infos ... whoever calls this global data
        n_parts = self.header["nummat8"] \
            + self.header["nummat2"] \
            + self.header["nummat4"] \
            + self.header["nummatt"]
        if "numrbs" in self.header["numbering_header"]:
            n_parts += self.header["numbering_header"]["numrbs"]

        # part internal energy
        self.arrays[arraytype.part_internal_energy] = \
            state_data[:, var_index:var_index+n_parts].T
        var_index += n_parts

        # part kinetic energy
        self.arrays[arraytype.part_kinetic_energy] = \
            state_data[:, var_index:var_index+n_parts].T
        var_index += n_parts

        # part velocity
        self.arrays[arraytype.part_velocity] = \
            state_data[:, var_index:var_index+3*n_parts]\
            .reshape((n_states, n_parts, 3))\
            .transpose((1, 2, 0))
        var_index += 3*n_parts

        # part mass
        self.arrays[arraytype.part_mass] = \
            state_data[:, var_index:var_index+n_parts].T
        var_index += n_parts

        # part hourglass energy
        self.arrays[arraytype.part_hourglass_energy] = \
            state_data[:, var_index:var_index+n_parts].T
        var_index += n_parts

        # rigid walls
        previous_global_vars = (6+7*n_parts)
        n_rigid_wall_vars = 4 if self.header["version"] >= 971 else 1
        # +1 is timestep which is not considered a global var ... seriously
        n_rigid_walls = (self.header["nglbv"] -
                         previous_global_vars) // n_rigid_wall_vars
        self.header["n_rigid_walls"] = n_rigid_walls
        self.header["n_rigid_wall_vars"] = n_rigid_wall_vars
        if previous_global_vars + n_rigid_walls*n_rigid_wall_vars != self.header["nglbv"]:
            logging.warn(
                "Bug while reading global data for rigid walls. Skipping this data.")
            var_index += self.header["nglbv"]-previous_global_vars
        else:

            # rigid wall force
            self.arrays[arraytype.rigid_wall_force] = \
                state_data[:, var_index:var_index+n_rigid_walls].T
            var_index += n_rigid_walls

            # rigid wall position
            if n_rigid_wall_vars > 1:
                self.arrays[arraytype.rigid_wall_position] = \
                    state_data[:, var_index:var_index+3*n_rigid_walls]\
                    .reshape(1, 2, 0)
                var_index += 3*n_rigid_walls

        logging.debug(
            "_read_states_global_vars end at var_index {}".format(var_index))

        return var_index

    def _read_states_nodes(self, state_data: np.ndarray, var_index: int) -> int:
        ''' Read the node data in the state sectio

        Parameters
        ----------
        state_data : np.ndarray
            array with entire state data
        var_index : int
            variable index in the state data array

        Returns
        -------
        var_index : int
            updated variable index after reading the section
        '''

        if not self.bb_states:
            return var_index

        if self.header["numnp"] <= 0:
            return var_index

        logging.debug(
            "_read_states_nodes start at var_index {}".format(var_index))

        n_dim = self.header["ndim"]
        n_states = state_data.shape[0]
        n_nodes = self.header["numnp"]

        # only node temperatures
        if self.header["it"] == 1:
            self.arrays[arraytype.node_temperature] = \
                state_data[:, var_index:var_index+n_nodes].T
            var_index += n_nodes

        # node temperature and node flux
        if self.header["it"] == 2:
            self.arrays[arraytype.node_temperature] = \
                state_data[:, var_index:var_index+n_nodes].T
            var_index += n_nodes

            tmp_array = state_data[:, var_index:var_index+3*n_nodes]\
                .reshape((n_states, n_nodes, 3))\
                .transpose((1, 2, 0))
            self.arrays[arraytype.node_heat_flux] = tmp_array
            var_index += 3*n_nodes

        # 3 temperatures per node and node flux
        # temperatures at inner, middle and outer layer
        if self.header["it"] == 3:
            tmp_array = state_data[:, var_index:var_index+3*n_nodes]\
                .reshape((n_states, n_nodes, 3))\
                .transpose((1, 2, 0))
            self.arrays[arraytype.node_temperature] = tmp_array
            var_index += 3*n_nodes

            tmp_array = state_data[:, var_index:var_index+3*n_nodes]\
                .reshape((n_states, n_nodes, 3))\
                .transpose((1, 2, 0))
            self.arrays[arraytype.node_heat_flux] = tmp_array
            var_index += 3*n_nodes

        # mass scaling
        if self.header["has_mass_scaling"]:
            self.arrays[arraytype.node_mass_scaling] = \
                state_data[:, var_index:var_index+n_nodes].T
            var_index += n_nodes

        # displacement
        if self.header["iu"]:
            tmp_array = state_data[:, var_index:var_index+n_dim*n_nodes]\
                .reshape((n_states, n_nodes, n_dim))\
                .transpose((1, 2, 0))
            self.arrays[arraytype.node_displacement] = tmp_array
            var_index += n_dim*n_nodes

        # velocity
        if self.header["iv"]:
            tmp_array = state_data[:, var_index:var_index+n_dim*n_nodes]\
                .reshape((n_states, n_nodes, n_dim))\
                .transpose((1, 2, 0))
            self.arrays[arraytype.node_velocity] = tmp_array
            var_index += n_dim*n_nodes

        # acceleration
        if self.header["ia"]:
            tmp_array = state_data[:, var_index:var_index+n_dim*n_nodes]\
                .reshape((n_states, n_nodes, n_dim))\
                .transpose((1, 2, 0))
            self.arrays[arraytype.node_acceleration] = tmp_array
            var_index += n_dim*n_nodes

        logging.debug(
            "_read_states_nodes end at var_index {}".format(var_index))

        return var_index

    def _read_states_solids_thermal(self, state_data: np.ndarray, var_index: int) -> int:
        ''' Read the thermal data for solids

        Parameters
        ----------
        state_data : np.ndarray
            array with entire state data
        var_index : int
            variable index in the state data array

        Returns
        -------
        var_index : int
            updated variable index after reading the section
        '''

        if not self.bb_states:
            return var_index

        if not "nt3d" in self.header:
            return var_index

        if self.header["nt3d"] <= 0:
            return var_index

        logging.debug(
            "_read_states_solids_thermal start at var_index {}".format(var_index))

        n_states = state_data.shape[0]
        n_solids = self.header["nel8"]
        n_thermal_vars = self.header["nt3d"]

        tmp_array = state_data[:, var_index:var_index+n_solids*n_thermal_vars]
        self.arrays[arraytype.element_solid_thermal_data] = \
            tmp_array\
            .reshape((n_states, n_solids, n_thermal_vars))\
            .transpose((1, 2, 0))
        var_index += n_thermal_vars*n_solids

        logging.debug(
            "_read_states_solids_thermal end at var_index {}".format(var_index))

        return var_index

    def _read_states_solids(self, state_data: np.ndarray, var_index: int) -> int:
        ''' Read the state data of the solid elements

        Parameters
        ----------
        state_data : np.ndarray
            array with entire state data
        var_index : int
            variable index in the state data array

        Returns
        -------
        var_index : int
            updated variable index after reading the section
        '''

        if not self.bb_states:
            return var_index

        if self.header["nel8"] <= 0 or self.header["nv3d"] <= 0:
            return var_index

        logging.debug(
            "_read_states_solids start at var_index {}".format(var_index))

        n_solid_vars = self.header["nv3d"]
        n_solids = self.header["nel8"]
        n_states = state_data.shape[0]
        n_strain_vars = 6*self.header["istrn"]
        n_history_vars = self.header["neiph"]-n_strain_vars

        solid_state_data = \
            state_data[:, var_index:var_index+n_solid_vars*n_solids]\
            .reshape((n_states, n_solids, n_solid_vars))
        var_index += n_solids*n_solid_vars

        # stress
        i_solid_var = 0
        self.arrays[arraytype.element_solid_stress] = \
            solid_state_data[:, :, :6]\
            .reshape((n_states, n_solids, 6))\
            .transpose((1, 2, 0))
        i_solid_var += 6

        # effective plastic strain
        self.arrays[arraytype.element_solid_effective_plastic_strain] = \
            solid_state_data[:, :, i_solid_var]\
            .reshape((n_states, n_solids)).T
        i_solid_var += 1

        # history vars
        if n_history_vars:
            self.arrays[arraytype.element_solid_history_variables] = \
                solid_state_data[i_solid_var:i_solid_var+n_history_vars]\
                .reshape((n_states, n_solids, n_history_vars))\
                .transpose((1, 2, 0))
            i_solid_var += n_history_vars

        # strain
        if n_strain_vars:
            self.arrays[arraytype.element_solid_strain] = \
                solid_state_data[:, :, i_solid_var:i_solid_var+n_strain_vars]\
                .reshape((n_states, n_solids, n_strain_vars))\
                .transpose((1, 2, 0))
            i_solid_var += n_strain_vars

        logging.debug(
            "_read_states_solids end at var_index {}".format(var_index))

        # a skip by nv3d*n_solids might be more robust
        return var_index

    def _read_states_thsell(self, state_data: np.ndarray, var_index: int) -> int:
        ''' Read the state data for thick shell elements 

        Parameters
        ----------
        state_data : np.ndarray
            array with entire state data
        var_index : int
            variable index in the state data array

        Returns
        -------
        var_index : int
            updated variable index after reading the section
        '''

        if not self.bb_states:
            return var_index

        if self.header["nelth"] <= 0 or self.header["nv3dt"] <= 0:
            return var_index

        logging.debug(
            "_read_states_thsell start at var_index {}".format(var_index))

        n_states = state_data.shape[0]
        n_tshells = self.header["nelth"]
        n_history_vars = self.header["neips"]
        n_layers = self.header["maxint"]
        n_layer_vars = n_layers \
            * (6*self.header["ioshl1"]
               + self.header["ioshl2"]
               + n_history_vars)
        n_strain_vars = 12*self.header["istrn"]
        n_thsell_vars = self.header["nv3dt"]
        has_stress = self.header["ioshl1"]
        has_pstrain = self.header["ioshl2"]

        # thick shell element data
        tshell_data = state_data[:,
                                 var_index:var_index+n_thsell_vars*n_tshells]
        tshell_data = tshell_data.reshape((n_states, n_tshells, n_thsell_vars))
        var_index += n_thsell_vars*n_tshells

        # extract layer data
        tshell_layer_data = tshell_data[:, :, slice(0, n_layer_vars)]
        tshell_layer_data = tshell_layer_data.reshape((n_states, -1))
        tshell_nonlayer_data = tshell_data[:, :, n_layer_vars:]

        # assemble dtype for reading layers
        tshell_dtype = [
            ('stress', (self.ftype, 6*has_stress)),
            ('plastic_strain', self.ftype, has_pstrain),
            ('history_vars', (self.ftype, n_history_vars))
        ]

        # read the shit
        tshell_layer_data = np.array(tshell_layer_data,
                                     copy=False,
                                     dtype=tshell_dtype)

        # save it
        if has_stress:
            self.arrays[arraytype.element_tshell_stress] = \
                tshell_layer_data["stress"]\
                .reshape((n_states, n_tshells, n_layers, 6))\
                .transpose((1, 3, 2, 0))
        if has_pstrain:
            self.arrays[arraytype.element_tshell_effective_plastic_strain] = \
                tshell_layer_data["plastic_strain"]\
                .reshape((n_states, n_tshells, n_layers))\
                .transpose((1, 2, 0))
        if n_history_vars:
            self.arrays[arraytype.element_tshell_history_variables] = \
                tshell_layer_data["history_vars"]\
                .reshape((n_states, n_tshells, n_layers, n_history_vars))\
                .transpose((1, 3, 2, 0))

        # strain data (only non layer data for tshells)
        if n_strain_vars:
            tshell_nonlayer_data = tshell_nonlayer_data[:, :, :n_strain_vars]
            tshell_nonlayer_data = tshell_nonlayer_data\
                .reshape((n_states, n_tshells, 2, 6))\
                .transpose((1, 3, 2, 0))

        logging.debug(
            "_read_states_thsell end at var_index {}".format(var_index))

        return var_index

    def _read_states_beams(self, state_data: np.ndarray, var_index: int) -> int:
        ''' Read the state data for beams 

        Parameters
        ----------
        state_data : np.ndarray
            array with entire state data
        var_index : int
            variable index in the state data array

        Returns
        -------
        var_index : int
            updated variable index after reading the section
        '''

        if not self.bb_states:
            return var_index

        if self.header["nel2"] <= 0 or self.header["nv1d"] <= 0:
            return var_index

        logging.debug(
            "_read_states_beams start at var_index {}".format(var_index))

        n_states = state_data.shape[0]
        n_beams = self.header["nel2"]
        n_history_vars = self.header["neipb"]
        n_beam_vars = self.header["nv1d"]
        n_layers = int((-3*n_history_vars+n_beam_vars-6)
                       / (n_history_vars+5))
        n_layer_vars = 6+5*n_layers\
                        + n_history_vars*(3+n_layers)

        # beam element data
        beam_data = state_data[:, var_index:var_index+n_beam_vars*n_beams]
        beam_data = beam_data.reshape((n_states, n_beams, n_beam_vars))
        var_index += n_beams*n_beam_vars

        # extract layer data
        beam_layer_data = beam_data[:, :, slice(6, 6+n_layer_vars)]
        beam_layer_data = beam_layer_data.reshape((n_states, -1))
        beam_nonlayer_data = beam_data[:, :, :6]

        # handle layers
        beam_dtype_layers = [
            ('shear_stress', (self.ftype, 2)),
            ('axial_stress', (self.ftype, 1)),
            ('plastic_strain', (self.ftype, 1)),
            ('axial_strain', (self.ftype, 1)),
            ('history_vars', (self.ftype,
                              n_history_vars*(3+n_layers))),
        ]
        beam_layer_data = np.array(
            beam_layer_data, copy=False, dtype=beam_dtype_layers)

        # axial force
        self.arrays[arraytype.element_beam_axial_force] = \
            beam_nonlayer_data[:, :, 0]\
            .reshape((n_states, n_beams))\
            .transpose((1, 0))
        # shear force
        self.arrays[arraytype.element_beam_shear_force] = \
            beam_nonlayer_data[:, :, 1:3]\
            .reshape((n_states, n_beams, 2))\
            .transpose((1, 2, 0))
        # bending moment
        self.arrays[arraytype.element_beam_bending_moment] = \
            beam_nonlayer_data[:, :, 3:5]\
            .reshape((n_states, n_beams, 2))\
            .transpose((1, 2, 0))
        # torsion moment
        self.arrays[arraytype.element_beam_torsion_moment] = \
            beam_nonlayer_data[:, :, 5]\
            .reshape((n_states, n_beams))\
            .transpose((1, 0))
        if n_layers:
            # shear stress
            self.arrays[arraytype.element_beam_shear_stress] = \
                beam_layer_data["shear_stress"]\
                .reshape((n_states, n_beams, n_layers, 2))\
                .transpose((1, 3, 2, 0))
            # axial stress
            self.arrays[arraytype.element_beam_axial_stress] = \
                beam_layer_data["axial_stress"]\
                .reshape((n_states, n_beams, n_layers))\
                .transpose((1, 2, 0))
            # eff. plastic strain
            self.arrays[arraytype.element_beam_plastic_strain] = \
                beam_layer_data["plastic_strain"]\
                .reshape((n_states, n_beams, n_layers))\
                .transpose((1, 2, 0))
            # axial strain
            self.arrays[arraytype.element_beam_axial_strain] = \
                beam_layer_data["axial_strain"]\
                .reshape((n_states, n_beams, n_layers, 2))\
                .transpose((1, 3, 2, 0))
            # history vars
            if n_history_vars:
                self.arrays[arraytype.element_beam_history_vars] = \
                    beam_layer_data["history_vars"]\
                    .reshape((n_states, n_beams, n_layers, n_history_vars))\
                    .transpose((1, 3, 2, 0))

        logging.debug(
            "_read_states_beams end at var_index {}".format(var_index))

        return var_index

    def _read_states_shell(self, state_data: np.ndarray, var_index: int) -> int:
        ''' Read the state data for shell elements 

        Parameters
        ----------
        state_data : np.ndarray
            array with entire state data
        var_index : int
            variable index in the state data array

        Returns
        -------
        var_index : int
            updated variable index after reading the section
        '''

        if not self.bb_states:
            return var_index

        if self.header["nel4"] <= 0 \
           or self.header["nv2d"]-self.header["numrbe"] <= 0:
            return var_index

        logging.debug(
            "_read_states_shell start at var_index {}".format(var_index))

        n_states = state_data.shape[0]
        n_shells = self.header["nel4"]-self.header["numrbe"]
        n_shell_vars = self.header["nv2d"]

        # what is in the file?
        n_layers = self.header["maxint"]
        n_history_vars = self.header["neips"]
        n_stress_vars = 6*self.header["ioshl1"]
        n_pstrain_vars = self.header["ioshl2"]
        n_force_variables = 8*self.header["ioshl3"]
        n_extra_variables = 4*self.header["ioshl4"]
        n_strain_vars = 12*self.header["istrn"]

        # this is a sanity check if the manual was understood correctly
        n_shell_vars2 = n_layers*(n_stress_vars+n_pstrain_vars+n_history_vars)\
            + n_force_variables+n_extra_variables+n_strain_vars
        if n_shell_vars != n_shell_vars2:
            msg = "n_shell_vars != n_shell_vars_computed: {} != {}."\
                + " Shell variables might be wrong."
            raise Warning(msg.format(n_shell_vars, n_shell_vars2))

        n_layer_vars = n_layers \
            * (n_stress_vars
               + n_pstrain_vars
               + n_history_vars)

        # shell element data
        shell_data = state_data[:, var_index:var_index+n_shell_vars*n_shells]
        shell_data = shell_data.reshape((n_states, n_shells, n_shell_vars))
        var_index += n_shell_vars*n_shells

        # extract layer data
        shell_layer_data = shell_data[:, :, :n_layer_vars]
        shell_layer_data = \
            shell_layer_data.reshape((n_states, n_shells, n_layers, -1))
        shell_nonlayer_data = shell_data[:, :, n_layer_vars:]

        # save layer stuff
        # stress
        layer_var_index = 0
        if n_stress_vars:
            self.arrays[arraytype.element_shell_stress] = \
                shell_layer_data[:, :, :, :n_stress_vars]\
                .reshape((n_states, n_shells, n_layers, n_stress_vars))\
                .transpose((1, 3, 2, 0))
            layer_var_index += n_stress_vars
        # pstrain
        if n_pstrain_vars:
            self.arrays[arraytype.element_shell_effective_plastic_strain] = \
                shell_layer_data[:, :, :, layer_var_index]\
                .reshape((n_states, n_shells, n_layers))\
                .transpose((1, 2, 0))
            layer_var_index += 1
        # history vars
        if n_history_vars:
            self.arrays[arraytype.element_shell_history_vars] = \
                shell_layer_data[:, :, :, layer_var_index:layer_var_index+n_history_vars]\
                .reshape((n_states, n_shells, n_layers, n_history_vars))\
                .transpose((1, 3, 2, 0))
            layer_var_index += n_history_vars

        # save nonlayer stuff
        # forces
        nonlayer_var_index = 0
        if n_force_variables:
            self.arrays[arraytype.element_shell_bending_moment] = \
                shell_nonlayer_data[:, :, 0:3]\
                .reshape((n_states, n_shells, 3))\
                .transpose((1, 2, 0))
            self.arrays[arraytype.element_shell_shear_force] = \
                shell_nonlayer_data[:, :, 3:5]\
                .reshape((n_states, n_shells, 2))\
                .transpose((1, 2, 0))
            self.arrays[arraytype.element_shell_normal_force] = \
                shell_nonlayer_data[:, :, 5:8]\
                .reshape((n_states, n_shells, 3))\
                .transpose((1, 2, 0))
            nonlayer_var_index += n_force_variables

        # weird stuff
        if n_extra_variables:
            self.arrays[arraytype.element_shell_thickness] = \
                shell_nonlayer_data[:, :, nonlayer_var_index]\
                .reshape((n_states, n_shells))\
                .transpose((1, 0))
            self.arrays[arraytype.element_shell_unknown_variables] = \
                shell_nonlayer_data[:, :, nonlayer_var_index+1:nonlayer_var_index+3]\
                .reshape((n_states, n_shells, 2))\
                .transpose((1, 2, 0))
            nonlayer_var_index += 3
            if self.header["istrn"] == 0:
                self.arrays[arraytype.element_shell_internal_energy] = \
                    shell_nonlayer_data[:, :, nonlayer_var_index]\
                    .reshape((n_states, n_shells))\
                    .transpose((1, 0))
                nonlayer_var_index += 1

        # strain
        if n_strain_vars:
            shell_strain = shell_nonlayer_data[:, :,
                                               nonlayer_var_index:nonlayer_var_index+n_strain_vars]
            self.arrays[arraytype.element_shell_strain] = \
                shell_strain.reshape((n_states, n_shells, 2, 6))\
                .reshape((n_states, n_shells, 2, 6))\
                .transpose((1, 3, 2, 0))
            nonlayer_var_index += n_strain_vars

            # internal energy is behind strain if strain is written
            # ... says the manual ...
            if n_shell_vars >= 45:
                self.arrays[arraytype.element_shell_internal_energy] = \
                    shell_nonlayer_data[:, :, nonlayer_var_index]\
                    .reshape((n_states, n_shells))\
                    .transpose((1, 0))

        logging.debug(
            "_read_states_shell end at var_index {}".format(var_index))

        return var_index

    def _read_states_is_alive(self, state_data: np.ndarray, var_index: int) -> int:
        ''' Read deletion info for nodes, elements, etc

        Parameters
        ----------
        state_data : np.ndarray
            array with entire state data
        var_index : int
            variable index in the state data array

        Returns
        -------
        var_index : int
            updated variable index after reading the section
        '''

        if not self.bb_states:
            return var_index

        if self.header["mdlopt"] <= 0:
            return var_index

        logging.debug(
            "_read_states_is_alive start at var_index {}".format(var_index))

        n_states = state_data.shape[0]

        # node deletion info
        if self.header["mdlopt"] == 1:
            n_nodes = self.header["numnp"]

            if n_nodes > 0:
                self.arrays[arraytype.node_is_alive] = \
                    state_data[:, var_index:var_index+n_nodes]
                var_index += n_nodes

        # element deletion info
        elif self.header["mdlopt"] == 2:
            n_solids = self.header["nel8"]
            n_tshells = self.header["nelth"]
            n_shells = self.header["nel4"]
            n_beams = self.header["nel2"]
            n_elems = n_solids + n_tshells + n_shells + n_beams

            # solids
            if n_solids > 0:
                self.arrays[arraytype.element_solid_is_alive] = \
                    state_data[:, var_index:var_index+n_solids]\
                    .reshape((n_states, n_solids))\
                    .transpose((1, 0))
                var_index += n_solids

            # tshells
            if n_tshells > 0:
                self.arrays[arraytype.element_tshell_is_alive] = \
                    state_data[:, var_index:var_index+n_tshells]\
                    .reshape((n_states, n_tshells))\
                    .transpose((1, 0))
                var_index += n_tshells

            # shells
            if n_shells > 0:
                self.arrays[arraytype.element_shell_is_alive] = \
                    state_data[:, var_index:var_index+n_shells]\
                    .reshape((n_states, n_shells))\
                    .transpose((1, 0))
                var_index += n_shells

            # beams
            if n_beams > 0:
                self.arrays[arraytype.element_beam_is_alive] = \
                    state_data[:, var_index:var_index+n_beams]\
                    .reshape((n_states, n_beams))\
                    .transpose((1, 0))
                var_index += n_beams

        logging.debug(
            "_read_states_is_alive end at var_index {}".format(var_index))

        return var_index

    def _read_states_sph(self, state_data: np.ndarray, var_index: int) -> int:
        ''' Read the sph state data

        Parameters
        ----------
        state_data : np.ndarray
            array with entire state data
        var_index : int
            variable index in the state data array

        Returns
        -------
        var_index : int
            updated variable index after reading the section
        '''

        if not self.bb_states:
            return var_index

        if self.header["nmsph"] <= 0:
            return var_index

        logging.debug(
            "_read_states_sph start at var_index {}".format(var_index))

        n_states = state_data.shape[0]
        n_particles = self.header["numsph"]
        n_variables = self.header["num_sph_vars"]

        # extract data
        sph_data = state_data[:, var_index:var_index+n_particles*n_variables]
        var_index += n_particles*n_variables

        # deletion
        self.arrays[arraytype.sph_deletion] = sph_data[:, 0] < 0
        i_var = 1

        # particle radius
        if self.header["isphfg2"]:
            self.arrays[arraytype.sph_radius] = sph_data[:, i_var]\
                .transpose((1, 0))
            i_var += 1

        # pressure
        if self.header["isphfg3"]:
            self.arrays[arraytype.sph_pressure] = sph_data[:, i_var]\
                .transpose((1, 0))
            i_var += 1

        # stress
        if self.header["isphfg4"]:
            self.arrays[arraytype.sph_stress] = sph_data[:, i_var:i_var+6]\
                .reshape((n_states, n_particles, 6))\
                .transpose((1, 2, 0))
            i_var += 6

        # eff. plastic strain
        if self.header["isphfg5"]:
            self.arrays[arraytype.sph_effective_plastic_strain] = \
                sph_data[:, i_var]\
                .transpose((1, 0))
            i_var += 1

        # density
        if self.header["isphfg6"]:
            self.arrays[arraytype.sph_density] = sph_data[:, i_var]\
                .transpose((1, 0))
            i_var += 1

        # internal energy
        if self.header["isphfg7"]:
            self.arrays[arraytype.sph_internal_energy] = sph_data[:, i_var]\
                .transpose((1, 0))
            i_var += 1

        # number of neighbors
        if self.header["isphfg8"]:
            self.arrays[arraytype.sph_n_neighbors] = sph_data[:, i_var]\
                .transpose((1, 0))
            i_var += 1

        # pressure
        if self.header["isphfg9"]:
            self.arrays[arraytype.sph_strain] = sph_data[:, i_var:i_var+6]\
                .reshape((n_states, n_particles, 6))\
                .transpose((1, 2, 0))
            i_var += 6

        # pressure
        if self.header["isphfg10"]:
            self.arrays[arraytype.sph_mass] = sph_data[:, i_var]\
                .transpose((1, 0))
            i_var += 1

        logging.debug("_read_states_sph end at var_index {}".format(var_index))

        return var_index

    def _read_states_airbags(self, state_data: np.ndarray, var_index: int) -> int:
        ''' Read the airbag state data

        Parameters
        ----------
        state_data : np.ndarray
            array with entire state data
        var_index : int
            variable index in the state data array

        Returns
        -------
        var_index : int
            updated variable index after reading the section
        '''

        if not self.bb_states:
            return var_index

        if self.header["npefg"] <= 0:
            return var_index

        logging.debug(
            "_read_states_airbags start at var_index {}".format(var_index))

        n_states = state_data.shape[0]
        n_airbag_geom_vars = self.header["airbag"]["ngeom"]
        n_airbags = self.header["airbag"]["npartgas"]
        n_state_airbag_vars = self.header["airbag"]["nstgeom"]
        n_particles = self.header["airbag"]["npart"]
        n_particle_vars = self.header["airbag"]["nvar"]

        # types
        # nlist = ngeom + nvar + nstgeom
        airbag_var_types = self.header["airbag"]["variable_types"]
        geom_var_types = airbag_var_types[:n_airbag_geom_vars]
        particle_var_types = airbag_var_types[
            n_airbag_geom_vars:n_airbag_geom_vars+n_particle_vars]
        airbag_state_var_types = \
            airbag_var_types[n_airbag_geom_vars+n_particle_vars:]

        # required for dynamic reading
        def get_dtype(
            type_flag): return self.itype if type_flag == 1 else self.ftype

        # Warning
        # Im not sure if this is right ...
        n_total_vars = \
            n_airbags*n_state_airbag_vars\
            + n_particles*n_particle_vars

        # extract airbag data
        airbag_state_data = state_data[:, var_index:var_index+n_total_vars]
        var_index += n_total_vars

        # airbag data
        airbag_data = airbag_state_data[:, :n_airbags*n_state_airbag_vars]\
            .reshape((n_states, n_airbags, n_state_airbag_vars))
        airbag_state_offset = n_airbags*n_state_airbag_vars

        # particle data
        particle_data = \
            airbag_state_data[:,
                              airbag_state_offset:
                              airbag_state_offset+n_particles*n_particle_vars]\
            .reshape((n_states, n_particles, n_particle_vars))\
            .transpose((1, 2, 0))

        # save sh...

        # airbag active particles
        if n_state_airbag_vars >= 1:
            self.arrays[arraytype.airbag_n_active_particles] = \
                airbag_data[:, :, 0]\
                .view(get_dtype(airbag_state_var_types[0]))\
                .transpose((1, 0))

        # airbag bag volumne
        if n_state_airbag_vars >= 2:
            dtype = airbag_state_var_types[1]
            self.arrays[arraytype.airbag_bag_volume] = \
                airbag_data[:, :, 1]\
                .view(get_dtype(airbag_state_var_types[1]))\
                .transpose((1, 0))

        # particle gas id
        if n_particle_vars >= 1:
            self.arrays[arraytype.airbag_particle_gas_id] = \
                particle_data[:, 0, :]\
                .view(get_dtype(particle_var_types[0]))

        # particle chamber id
        if n_particle_vars >= 2:
            self.arrays[arraytype.airbag_particle_chamber_id] = \
                particle_data[:, 1, :]\
                .view(get_dtype(particle_var_types[1]))

        # particle leakage
        if n_particle_vars >= 3:
            self.arrays[arraytype.airbag_particle_leakage] = \
                particle_data[:, 2, :]\
                .view(get_dtype(particle_var_types[2]))

        # particle mass
        if n_particle_vars >= 4:
            self.arrays[arraytype.airbag_particle_mass] = \
                particle_data[:, 3, :]\
                .view(get_dtype(particle_var_types[3]))

        # particle radius
        if n_particle_vars >= 5:
            self.arrays[arraytype.airbag_particle_radius] = \
                particle_data[:, 4, :]\
                .view(get_dtype(particle_var_types[4]))

        # particle spin energy
        if n_particle_vars >= 6:
            self.arrays[arraytype.airbag_particle_spin_energy] = \
                particle_data[:, 5, :]\
                .view(get_dtype(particle_var_types[5]))

        # particle translational energy
        if n_particle_vars >= 7:
            self.arrays[arraytype.airbag_particle_translation_energy] = \
                particle_data[:, 6, :]\
                .view(get_dtype(particle_var_types[6]))

        # particle segment distance
        if n_particle_vars >= 8:
            self.arrays[arraytype.airbag_particle_nearest_segment_distance] = \
                particle_data[:, 7, :]\
                .view(get_dtype(particle_var_types[7]))

        # particle position
        if n_particle_vars >= 11:
            self.arrays[arraytype.airbag_particle_position] = \
                particle_data[:, 8:11, :]\
                .view(get_dtype(particle_var_types[8]))

        # particle velocity
        if n_particle_vars >= 14:
            self.arrays[arraytype.airbag_particle_nearest_segment_distance] = \
                particle_data[:, 11:14, :]\
                .view(get_dtype(particle_var_types[11]))

        logging.debug(
            "_read_states_airbags end at var_index {}".format(var_index))

        return var_index

    def _read_states_road_surfaces(self, state_data: np.ndarray, var_index: int) -> int:
        ''' Read the road surfaces state data for whoever want this ...

        Parameters
        ----------
        state_data : np.ndarray
            array with entire state data
        var_index : int
            variable index in the state data array

        Returns
        -------
        var_index : int
            updated variable index after reading the section
        '''

        if not self.bb_states:
            return var_index

        if not self.header["has_rigid_road_surface"]:
            return var_index

        logging.debug(
            "_read_states_road_surfaces start at var_index {}".format(var_index))

        n_states = state_data.shape[0]
        n_roads = self.header["rigid_road"]["nsurf"]

        # read road data
        road_data = state_data[:, var_index:var_index+6*n_roads]\
            .reshape((n_states, n_roads, 2, 3))
        var_index += 6*n_roads

        # road displacement
        self.arrays[arraytype.rigid_road_displacement] = \
            road_data[:, :, 0, :]\
            .transpose(1, 2, 0)

        # road velocity
        self.arrays[arraytype.rigid_road_velocity] = \
            road_data[:, :, 1, :]\
            .transpose(1, 2, 0)

        logging.debug(
            "_read_states_road_surfaces end at var_index {}".format(var_index))

        return var_index

    def _read_states_rigid_body_motion(self, state_data: np.ndarray, var_index: int) -> int :
        ''' Read the road surfaces state data for whoever want this ...

        Parameters
        ----------
        state_data : np.ndarray
            array with entire state data
        var_index : int
            variable index in the state data array

        Returns
        -------
        var_index : int
            updated variable index after reading the section
        '''

        if not self.bb_states:
            return var_index

        if not self.header["has_rigid_body_data"]:
            return var_index

        logging.debug(
            "_read_states_rigid_body_motion start at var_index {}".format(var_index))

        n_states = state_data.shape[0]
        n_rigids = self.header["nrigid"]
        n_rigid_vars = 12 if self.header["has_reduced_rigid_body_data"] else 24

        # do the thing
        rigid_body_data = state_data[:, var_index:var_index+n_rigids*n_rigid_vars]\
            .reshape((n_states, n_rigids, n_rigid_vars))
        var_index += n_rigids*n_rigid_vars

        # let the party begin
        # rigid coordinates
        self.arrays[arraytype.rigid_body_coordinates] = \
            rigid_body_data[:, :, :3]\
            .transpose((1, 2, 0))
        i_var = 3

        # rotation matrix
        self.arrays[arraytype.rigid_body_rotation_matrix] = \
            rigid_body_data[:, :, i_var:i_var+9]\
            .transpose((1, 2, 0))
        i_var += 9

        if self.header["has_reduced_rigid_body_data"]:
            return var_index

        # velocity pewpew
        self.arrays[arraytype.rigid_body_velocity] = \
            rigid_body_data[:, :, i_var:i_var+3]\
            .transpose((1, 2, 0))
        i_var += 3

        # rotational velocity
        self.arrays[arraytype.rigid_body_rot_velocity] = \
            rigid_body_data[:, :, i_var:i_var+3]\
            .transpose((1, 2, 0))
        i_var += 3

        # acceleration
        self.arrays[arraytype.rigid_body_acceleration] = \
            rigid_body_data[:, :, i_var:i_var+3]\
            .transpose((1, 2, 0))
        i_var += 3

        # rotational acceleration
        self.arrays[arraytype.rigid_body_rot_acceleration] = \
            rigid_body_data[:, :, i_var:i_var+3]\
            .transpose((1, 2, 0))
        i_var += 3

        logging.debug(
            "_read_states_rigid_body_motion end at var_index {}".format(var_index))

        return var_index

    def _read_state_bytebuffer(self, size_per_state: int):
        ''' This routine reads the data for state information

        Parameters
        ----------
        size_per_state : int
            size of every state to be read

        Returns
        -------
        bb_states : BinaryBuffer
            New binary buffer with all states perfectly linear in memory
        n_states : int
            Number of states to be expected

        Notes
        -----
            State data is expected directly behind geometry data
            Unfortunately data is spread across multiple files.
            One file could contain geometry and state data but states
            may also be littered accross several files. This would
            not be an issue, if dyna would not always write in blocks
            of 512 words of memory, leaving zero byte padding blocks
            at the end of files. These need to be removed and/or taken
            care of.
        '''

        if not self.bb:
            return

        base_filepath = self.bb.filepath_[0]
        base_filesize = len(self.bb)

        # bugfix
        # If you encounter these int casts more often here this is why:
        # Some ints around here are numpy.int32 which can overflow
        # (sometimes there is a warning ... sometimes not ...)
        # we cast to python ints in order to prevent overflow.
        size_per_state = int(size_per_state)

        # settings
        block_size = 512*self.wordsize

        # query for state files
        filepaths = self._find_dyna_result_files(base_filepath)

        # compute state data in first file
        n_states_beyond_geom = (
            base_filesize - self.geometry_section_size) // size_per_state
        n_states_beyond_geom = int(n_states_beyond_geom)

        # memory required later
        memory_infos = [{
            "start": self.geometry_section_size,
            "length": n_states_beyond_geom*size_per_state,
            "offset": 0,
            "filepath": base_filepath,
            "n_states": n_states_beyond_geom
        }]

        # compute amount of state data in every further file
        for filepath in filepaths:
            filesize = os.path.getsize(filepath)
            n_states_in_file = filesize // size_per_state
            memory_infos.append({
                "start": 0,
                "length": size_per_state*(n_states_in_file),
                "offset": memory_infos[-1]["length"],
                "filepath": filepath,
                "n_states": n_states_in_file
            })

        # allocate memory
        # bugfix: casting to int prevents int32 overflow for large files
        memory_required = 0
        for mem in memory_infos:
            memory_required += int(mem["length"])
        mview = memoryview(bytearray(memory_required))

        # transfer memory from first file
        n_states = memory_infos[0]["n_states"]
        start = memory_infos[0]["start"]
        length = memory_infos[0]["length"]
        end = start + length
        mview[:length] = self.bb.memoryview[start:end]

        # transfer memory for other files
        total_offset = length
        for minfo in memory_infos[1:]:
            start = minfo["start"]
            length = minfo["length"]
            offset = minfo["offset"]
            filepath = minfo["filepath"]

            with open(filepath, "br") as fp:
                fp.readinto(mview[total_offset:total_offset+length])

            total_offset += length
            n_states += minfo["n_states"]

        # save
        bb_states = BinaryBuffer()
        bb_states.memoryview = mview
        return bb_states, n_states

    def _read_words(self, words_to_read: dict, storage_dict: dict =None):
        ''' Read several words described by a dict

        Parameters
        ----------
        words_to_read : dict
            this dict describes the words to be read. One entry
            must be a tuple of len two (byte position and dtype)
        storage_dict : dict
            in this dict the read words will be saved

        Returns
        -------
        storage_dict : dict
            the storage dict given as arg or a new dict if none was given
        '''

        if storage_dict == None:
            storage_dict = {}

        for name, data in words_to_read.items():
            if data[1] == self.itype:
                storage_dict[name] = self.bb.read_number(
                    data[0], data[1])
            elif data[1] == self.ftype:
                storage_dict[name] = self.bb.read_number(
                    data[0], data[1])
            elif data[1] == self.charsize:
                try:
                    storage_dict[name] = self.bb.read_text(
                        data[0], data[1]*data[2])
                except UnicodeDecodeError as err:
                    storage_dict[name] = ""

            else:
                raise RuntimeError(
                    "Encountered unknown dtype {} during reading.".format(str(data[1])))

        return storage_dict

    def _find_dyna_result_files(self, filepath: str):
        '''Searches all dyna result files

        Parameters
        ----------
        filepath : str
            path to the first basic d3plot file

        Returns
        -------
        filepaths : list of str
            path to all dyna files

        Notes
        -----
            The dyna files usually follow a scheme to
            simply have the base name and numbers appended
            e.g. (d3plot, d3plot0001, d3plot0002, etc.)
        '''

        file_dir = os.path.dirname(filepath)
        file_dir = file_dir if len(file_dir) != 0 else '.'
        file_basename = os.path.basename(filepath)

        # pattern = "({path})([0-9]+)?".format(path=file_basename)
        pattern = "({path})[0-9]+".format(path=file_basename)
        reg = re.compile(pattern)

        filepaths = [os.path.join(file_dir, path) for path in os.listdir(file_dir)
                     if os.path.isfile(os.path.join(file_dir, path))
                     and reg.match(path)]

        return sorted(filepaths)

    def _determine_wordsize(self):
        '''Determine the precision of the file

        Returns
        -------
        wordsize : int
            size of each word in bytes
        '''

        if not self.bb:
            return 4, np.int32, np.float32

        # test file type flag (1=d3plot, 5=d3part)

        # single precision
        value = self.bb.read_number(44, np.int32)
        if value > 1000:
            value -= 1000
        if value == 1 or value == 5:
            return 4, np.int32, np.float32

        # double precision
        value = self.bb.read_number(88, np.int64)
        if value > 1000:
            value -= 1000
        if value == 1 or value == 5:
            return 8, np.int64, np.float64

        raise RuntimeError("Unknown file type.")

    def plot(self, i_timestep: int = 0, field=None, is_element_field: bool = True):
        ''' Plot the d3plot geometry

        Parameters
        ----------
        i_timestep : int
            timestep index to plot
        field : np.ndarray
            Array containing a field value for every element

        Notes
        -----
            Currently only shell elements can be plotted, since for
            solids the surface needs extraction.
        '''

        assert(i_timestep < self.header["n_timesteps"])

        # shell nodes
        shell_node_indexes = self.arrays[arraytype.element_shell_node_indexes]-1

        # get node displacement
        node_xyz = self.arrays[arraytype.node_displacement][:, :, i_timestep]

        # check for correct field size
        if isinstance(field, np.ndarray):
            assert(field.ndim == 1)
            if is_element_field and len(shell_node_indexes) != len(field):
                msg = "Element indexes and field have different len: {} != {}"
                raise ValueError(msg.format(
                    shell_node_indexes.shape, field.shape))
            elif not is_element_field and len(node_xyz) != len(field):
                msg = "Node field and coords have different len: {} != {}"
                raise ValueError(msg.format(node_xyz.shape, field.shape))

        # create plot
        _html = plot_shell_mesh(
            node_xyz, shell_node_indexes, field, is_element_field)

        # store in a temporary file
        tempdir = tempfile.gettempdir()
        tempdir = os.path.join(tempdir, "lasso")
        if not os.path.isdir(tempdir):
            os.mkdir(tempdir)

        for tmpfile in os.listdir(tempdir):
            tmpfile = os.path.join(tempdir, tmpfile)
            if os.path.isfile(tmpfile):
                os.remove(tmpfile)

        # create new temp file
        with tempfile.NamedTemporaryFile(dir=tempdir, suffix=".html", mode="w", delete=False) as fp:
            fp.write(_html)
            webbrowser.open(fp.name)

    @staticmethod
    def enable_logger(enable: bool):
        ''' Enable the logger for this class

        Parameters
        ----------
        enable : `bool`
            whether to enable logging for this class
        '''

        if enable:
            logging.getLogger().setLevel(logging.DEBUG)
        else:
            logging.getLogger().setLevel(logging.NOTSET)
