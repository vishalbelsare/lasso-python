
import ctypes
import logging
import os
import pprint
import re
import struct
import sys
import tempfile
import traceback
import typing
import webbrowser
from typing import Any, ByteString, Dict, Iterable, List, Tuple, Union

import numpy as np

from ..io.BinaryBuffer import BinaryBuffer
from ..plotting import plot_shell_mesh
from .ArrayType import ArrayType
from .FilterType import FilterType

arraytype = ArrayType
FORTRAN_OFFSET = 1


class D3plot:
    '''Class used to read LS-Dyna d3plots
    '''

    def __init__(self,
                 filepath: str = None,
                 use_femzip: bool = False,
                 n_files_to_load_at_once=-1,
                 state_array_filter=None):
        '''Constructor for a D3plot

        Parameters
        ----------
        filepath : `str`
            path to a d3plot file
        use_femzip : `bool`
            whether to use femzip decompression
        n_files_to_load_at_once : `int`
            number of d3plot files to load into memory at once. By default -1 means all are loaded
        state_array_filter : `list` of `str` or `None`
            names of arrays which will be the only ones loaded from state data
        Notes
        -----
            If dyna wrote multiple files for several states,
            only give the path to the first file.
        '''
        super(D3plot, self).__init__()

        self._arrays = {}
        self._header = {}

        # how many files to load into memory at once
        self.n_files_to_load_at_once = n_files_to_load_at_once

        # arrays to filter out
        self.state_array_filter = state_array_filter

        # femzip
        if filepath and not use_femzip:
            self.bb_generator = self._read_d3plot_file_generator(
                filepath, n_files_to_load_at_once)
            self.bb = next(self.bb_generator)
            self.bb_states = None
        elif filepath and use_femzip:
            self.bb_generator = self._read_femzip_file_generator(
                filepath, n_files_to_load_at_once)
            self.bb = next(self.bb_generator)
            self.bb_states = None
        else:
            self.bb_generator = None
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

        block_count = len(self.bb) // (512 * self.wordsize)

        # Warning!
        # Resets the block count!
        self.geometry_section_size = (block_count + 1) * 512 * self.wordsize

    def _get_n_parts(self) -> int:
        ''' Get the number of parts contained in the d3plot

        Returns
        -------
        n_parts : `int`
            number of total parts
        '''

        n_parts = self.header["nummat8"] \
            + self.header["nummat2"] \
            + self.header["nummat4"] \
            + self.header["nummatt"]
        if "numrbs" in self.header["numbering_header"]:
            n_parts += self.header["numbering_header"]["numrbs"]

        return n_parts

    def _get_n_rigid_walls(self) -> int:
        ''' Get the number of rigid walls in the d3plot

        Returns
        -------
        n_rigid_walls : `int`
            number of rigid walls
        '''
        previous_global_vars = (6 + 7 * self._get_n_parts())
        n_rigid_wall_vars = 4 if self.header["version"] >= 971 else 1
        # +1 is timestep which is not considered a global var ... seriously
        n_rigid_walls = (self.header["nglbv"] -
                         previous_global_vars) // n_rigid_wall_vars

        return n_rigid_walls

    def _read_d3plot_file_generator(self,
                                    filepath: str,
                                    n_files_to_load_at_once: int) -> typing.Any:
        ''' Generator function for reading bare d3plot files
        '''

        # (1) GEOMETRY
        bb = BinaryBuffer(filepath)
        yield bb

        # (2) STATES
        file_infos = self._collect_file_infos(
            D3plot._compute_n_bytes_per_state(self.header, self.wordsize))
        n_states = sum(
            map(lambda file_info: file_info["n_states"], file_infos))

        n_files = len(file_infos) if n_files_to_load_at_once <= -1 \
            else n_files_to_load_at_once
        sub_file_infos = [file_infos[index:index + n_files]
                          for index in range(0, len(file_infos), n_files)]

        n_states = sum(map(lambda info: info["n_states"], file_infos))

        logging.debug("n_files found: {0}".format(n_files))
        logging.debug("n_states estimated: {0}".format(n_states))
        logging.debug("files: {0}".format(pprint.pformat([info for info in file_infos])))

        # number of states and if buffered reading is used
        yield n_states, len(sub_file_infos) > 1

        for sub_file_info_list in sub_file_infos:
            bb, n_states = D3plot._read_file_from_memory_info(
                sub_file_info_list)
            yield bb, n_states

    def _read_femzip_file_generator(self,
                                    filepath: str,
                                    n_files_to_load_at_once: int) -> typing.Any:
        ''' Generator function for reading femzipped d3plot files
        '''

        # windows
        if os.name == "nt":
            libname = 'femzip_buffer.dll'

            # catch if cx_freeze is used
            if hasattr(sys, 'frozen'):
                femzip_lib = ctypes.CDLL(
                    os.path.join(os.path.abspath(os.path.dirname(sys.executable)), libname))
            else:
                femzip_lib = ctypes.CDLL(
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), libname))

        # linux .. hopefully
        else:
            libname = 'femzip_buffer.so'

            if hasattr(sys, 'frozen'):
                femzip_lib = ctypes.CDLL(
                    os.path.join(os.path.abspath(os.path.dirname(sys.executable)), libname))
            else:
                femzip_lib = ctypes.CDLL(
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), libname))

        if not filepath:
            return

        # (1) GEOMETRY

        # create ctype data types
        string_encode = filepath.encode('utf-8')

        # array holding buffer shape [n_timesteps, size_geo, size_state, size_part_titles]
        buffer_sizes_type = ctypes.c_int64 * 4

        # create the arrays with defined data types
        buffer_sizes = buffer_sizes_type()

        # compute the buffer dimensions using femzip
        femzip_lib.get_buffer_dimensions(
            ctypes.create_string_buffer(string_encode), buffer_sizes)

        n_timesteps = buffer_sizes[0]
        size_geometry = buffer_sizes[1]
        size_states = buffer_sizes[2]
        size_part_titles = buffer_sizes[3]

        self.header["femzip"] = {
            "n_states": n_timesteps,
            "size_geometry": size_geometry,
            "part_titles_size": size_part_titles,
            "size_states": size_states,
        }

        # geo_buffer also holds byte array of part titles at the end
        buffer_geom_type = ctypes.c_int * (size_geometry + size_part_titles)
        buffer_geo = buffer_geom_type()

        # read geometry
        femzip_lib.read_geom(
            ctypes.create_string_buffer(string_encode), buffer_sizes, buffer_geo)

        # save
        bb = BinaryBuffer()
        bb.memoryview = memoryview(buffer_geo).cast('B')

        yield bb

        # (2) STATES
        # number of states and if buffered reading is used
        yield size_part_titles

        if n_files_to_load_at_once <= -1:
            n_files_to_load_at_once = n_timesteps

        yield n_timesteps, n_timesteps != n_files_to_load_at_once

        buffer_state_type = ctypes.c_int * size_states * n_files_to_load_at_once
        buffer_state = buffer_state_type()

        bb = BinaryBuffer()
        bb.memoryview = memoryview(buffer_state).cast('B')

        # get the state buffer size
        size_states = ctypes.c_int(size_states * n_files_to_load_at_once)

        # do the thing
        FORTRAN_OFFSET = 1
        for i_timestep in range(FORTRAN_OFFSET,
                                n_timesteps + FORTRAN_OFFSET,
                                n_files_to_load_at_once):

            if ((n_timesteps - i_timestep) // n_files_to_load_at_once) == 0:
                n_states_to_load = n_timesteps + FORTRAN_OFFSET - i_timestep
            else:
                n_states_to_load = n_files_to_load_at_once

            n_timesteps_read = femzip_lib.read_states(
                i_timestep, n_states_to_load, buffer_state, size_states)
            yield bb, n_timesteps_read

        # do the thing
        femzip_lib.finish_reading(buffer_state, size_states)

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
            "title": [0 * self.wordsize, self.charsize, 10 * self.wordsize],
            "runtime": [10 * self.wordsize, self.itype],
            "filetype": [11 * self.wordsize, self.itype],
            "source_version": [12 * self.wordsize, self.itype],
            "release_version": [13 * self.wordsize, self.charsize, 1 * self.wordsize],
            "version": [14 * self.wordsize, self.ftype],
            "ndim": [15 * self.wordsize, self.itype],
            "numnp": [16 * self.wordsize, self.itype],
            "icode": [17 * self.wordsize, self.itype],
            "nglbv": [18 * self.wordsize, self.itype],
            "it": [19 * self.wordsize, self.itype],
            "iu": [20 * self.wordsize, self.itype],
            "iv": [21 * self.wordsize, self.itype],
            "ia": [22 * self.wordsize, self.itype],
            "nel8": [23 * self.wordsize, self.itype],
            "nummat8": [24 * self.wordsize, self.itype],
            "numds": [25 * self.wordsize, self.itype],
            "numst": [26 * self.wordsize, self.itype],
            "nv3d": [27 * self.wordsize, self.itype],
            "nel2": [28 * self.wordsize, self.itype],
            "nummat2": [29 * self.wordsize, self.itype],
            "nv1d": [30 * self.wordsize, self.itype],
            "nel4": [31 * self.wordsize, self.itype],
            "nummat4": [32 * self.wordsize, self.itype],
            "nv2d": [33 * self.wordsize, self.itype],
            "neiph": [34 * self.wordsize, self.itype],
            "neips": [35 * self.wordsize, self.itype],
            "maxint": [36 * self.wordsize, self.itype],
            "nmsph": [37 * self.wordsize, self.itype],
            "ngpsph": [38 * self.wordsize, self.itype],
            "narbs": [39 * self.wordsize, self.itype],
            "nelth": [40 * self.wordsize, self.itype],
            "nummatt": [41 * self.wordsize, self.itype],
            "nv3dt": [42 * self.wordsize, self.itype],
            "ioshl1": [43 * self.wordsize, self.itype],
            "ioshl2": [44 * self.wordsize, self.itype],
            "ioshl3": [45 * self.wordsize, self.itype],
            "ioshl4": [46 * self.wordsize, self.itype],
            "ialemat": [47 * self.wordsize, self.itype],
            "ncfdv1": [48 * self.wordsize, self.itype],
            "ncfdv2": [49 * self.wordsize, self.itype],
            "nadapt": [50 * self.wordsize, self.itype],
            "nmmat": [51 * self.wordsize, self.itype],
            "numfluid": [52 * self.wordsize, self.itype],
            "inn": [53 * self.wordsize, self.itype],
            "npefg": [54 * self.wordsize, self.itype],
            "nel48": [55 * self.wordsize, self.itype],
            "idtdt": [56 * self.wordsize, self.itype],
            "extra": [57 * self.wordsize, self.itype],
        }

        header_extra_words = {
            "nel20": [64 * self.wordsize, self.itype],
            "nt3d": [65 * self.wordsize, self.itype],
            "nel27": [66 * self.wordsize, self.itype],
            "neipb": [67 * self.wordsize, self.itype],
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

        # store raw header for checks during writing
        self._raw_header = {}
        self._raw_header.update(self.header)

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
        if self.header["ndim"] == 4:
            self.header["mattyp"] = 0
            self.header["ndim"] = 3
            self.header['has_rigid_road_surface'] = False
            self.header['has_rigid_body_data'] = False
            # self.header['elem_connectivity_unpacked'] = True
        if 5 < self.header["ndim"] < 8:
            self.header["mattyp"] = 0
            self.header['ndim'] = 3
            self.header['has_rigid_road_surface'] = True
            self.header['has_rigid_body_data'] = False
        if self.header['ndim'] == 8 or self.header['ndim'] == 9:
            self.header["mattyp"] = 0
            self.header['ndim'] = 3
            if self.header['ndim'] == 9:
                self.header['has_rigid_road_surface'] = True
                self.header['has_reduced_rigid_body_data'] = True
            else:
                self.header['has_rigid_road_surface'] = False
                self.header['has_reduced_rigid_body_data'] = False
            self.header['has_rigid_body_data'] = True
        if self.header["ndim"] != 3:
            raise RuntimeError(
                "Invalid header entry ndim: %d" % self.header["ndim"])

        # mass scaling
        if self.header["it"] >= 10:
            self.header["has_mass_scaling"] = (self.header["it"] / 10) == 1
        else:
            self.header["has_mass_scaling"] = False

        # temperature
        if self.header["it"] != 0:
            self.header["has_temperatures"] = (self.header["it"] % 10) != 0
        else:
            self.header["has_temperatures"] = False

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
        # FIXME
        # make it correct ... once more
        # took me like 1000 years to figure this out ....
        if self.header["idtdt"] > 100:
            self.header["istrn"] = self.header["idtdt"] % 10000
        else:
            if self.header["nv2d"] > 0:
                if (self.header["nv2d"] -
                    self.header["maxint"] *
                        (6 * self.header["ioshl1"] +
                         self.header["ioshl2"] +
                         self.header["neips"]) -
                        8 * self.header["ioshl3"] -
                        4 * self.header["ioshl4"]) > 1:

                    self.header["istrn"] = 1
                else:
                    self.header["istrn"] = 0

            elif self.header["nelth"] > 0:
                if (self.header["nv3dt"] -
                        self.header["maxint"] * (6 * self.header["ioshl1"] +
                                                 self.header["ioshl2"] +
                                                 self.header["neips"])) > 1:

                    self.header["istrn"] = 1
                else:
                    self.header["istrn"] = 0
            else:
                self.header["istrn"] = 0

        # internal energy
        shell_vars_behind_layers = (self.header["nv2d"] - self.header["maxint"] * (
            6 * self.header["ioshl1"] + self.header["ioshl2"] + self.header["neips"]) +
            8 * self.header["ioshl3"] + 4 * self.header["ioshl4"])

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

        # node temperature gradient
        # TODO
        # idtdt // 1 == 1

        # node residual forces and moments
        # TODO
        # idtdt % 10 == 1

        # pstrain tensor
        # TODO
        # idtdt % 100 == 1
        #  > solid 6 values
        #  > shell 6*3 values (layers)

        # thermal strain tensor
        # TODO
        # idtdt % 1000 == 1
        #  > solid 6 values
        #  > shells 6 values

        # CHECKS
        if self.header["ncfdv1"] == 67108864:
            raise RuntimeError("Can not handle CFD Multi-Solver data. ")

        self.geometry_section_size = 64 * \
            (1 + (self.header['extra'] != 0)) * self.wordsize

        logging.debug("_read_header end at byte {}".format(
            self.geometry_section_size))

    def _read_material_section(self):
        ''' This function reads the material type section
        '''

        if not self.bb:
            return

        logging.debug("_read_material_section start at byte {}".format(
            self.geometry_section_size))

        position = self.geometry_section_size

        # failsafe
        original_position = self.geometry_section_size
        blocksize = (2 + self.header["nmmat"]) * self.wordsize

        try:

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

                self.header["numrbe"] = self.bb.read_number(
                    position, self.itype)
                position += self.wordsize

                test_nummat = self.bb.read_number(
                    position, self.itype)
                position += self.wordsize

                if test_nummat != self.header["nmmat"]:
                    raise RuntimeError("nmmat (header) != nmmat (material type data): %d != %d" % (
                        self.header["nmmat"], test_nummat))

                self.arrays[arraytype.part_material_type] = \
                    self.bb.read_ndarray(
                        position, self.header["nmmat"] * self.wordsize, 1, self.itype)
                position += self.header["nmmat"] * self.wordsize

        except Exception:
            # print info
            trb_msg = traceback.format_exc()
            msg = "A failure in {0} was caught:\n{1}"
            logging.warning(msg.format("_read_material_section", trb_msg))

            # fix position
            position = original_position + blocksize

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

        # safety
        original_position = position
        blocksize = self.header["ialemat"] * self.wordsize

        try:
            # Fluid Material Data
            array_length = self.header["ialemat"] * self.wordsize
            self.arrays[arraytype.ale_material_ids] = \
                self.bb.read_ndarray(position, array_length, 1, self.itype)
            position += array_length

        except Exception:

            # print info
            trb_msg = traceback.format_exc()
            msg = "A failure in {0} was caught:\n{1}"
            logging.warning(
                msg.format("_read_fluid_material_data", trb_msg))

            # fix position
            position = original_position + blocksize

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
            "isphfg2": (position + 1 * self.wordsize, self.itype),
            "isphfg3": (position + 2 * self.wordsize, self.itype),
            "isphfg4": (position + 3 * self.wordsize, self.itype),
            "isphfg5": (position + 4 * self.wordsize, self.itype),
            "isphfg6": (position + 5 * self.wordsize, self.itype),
            "isphfg7": (position + 6 * self.wordsize, self.itype),
            "isphfg8": (position + 7 * self.wordsize, self.itype),
            "isphfg9": (position + 8 * self.wordsize, self.itype),
            "isphfg10": (position + 9 * self.wordsize, self.itype),
            "isphfg11": (position + 10 * self.wordsize, self.itype),
        }

        sph_element_data_header = self._read_words(sph_element_data_words)
        self.header.update(**sph_element_data_header)

        if self.header["isphfg1"] != 11:
            msg = "Detected inconsistency: isphfg = {0} but must be 11."
            raise RuntimeError(msg.format(self.header["isphfg1"]))

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

        self.geometry_section_size += self.header["isphfg1"] * self.wordsize
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
            'subver': self.header['npefg'] // 1000
        }

        particle_geometry_data_words = {
            # number of geometry variables
            'ngeom': (position, self.itype),
            # number of state variables
            'nvar': (position + 1 * self.wordsize, self.itype),
            # number of particles
            'npart': (position + 2 * self.wordsize, self.itype),
            # number of state geometry variables
            'nstgeom': (position + 3 * self.wordsize, self.itype)
        }

        self._read_words(particle_geometry_data_words, airbag_header)
        position += 4 * self.wordsize

        if airbag_header['subver'] == 4:
            # number of chambers
            airbag_header['nchamber'] = self.bb.read_number(
                position, self.itype)
            position += self.wordsize

        airbag_header['nlist'] = airbag_header['ngeom'] + \
            airbag_header['nvar'] + airbag_header['nstgeom']

        # safety
        # from here on the code may fail
        original_position = position
        blocksize = 9 * airbag_header['nlist'] * self.wordsize

        try:
            # variable typecodes
            self.arrays[arraytype.airbag_variable_types] = \
                self.bb.read_ndarray(position,
                                     airbag_header['nlist'] * self.wordsize,
                                     1,
                                     self.itype)
            position += airbag_header['nlist'] * self.wordsize

            # airbag variable names
            # every word is an ascii char
            airbag_variable_names = []
            var_width = 8

            for i_variable in range(airbag_header['nlist']):
                name = self.bb.read_text(position + (i_variable * var_width) *
                                         self.wordsize, var_width * self.wordsize)
                airbag_variable_names.append(name[::self.wordsize])

            self.arrays[arraytype.airbag_variable_names] = airbag_variable_names
            position += airbag_header['nlist'] * var_width * self.wordsize

        except Exception:

            # print info
            trb_msg = traceback.format_exc()
            msg = "A failure in {0} was caught:\n{1}"
            logging.warning(msg.format("_read_particle_data", trb_msg))

            # fix position
            position = original_position + blocksize

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
        section_word_length = self.header['ndim'] * self.header['numnp']
        try:
            node_coordinates = \
                self.bb.read_ndarray(position,
                                     section_word_length * self.wordsize,
                                     1,
                                     self.ftype)\
                .reshape((self.header['numnp'], self.header['ndim']))
            self.arrays[arraytype.node_coordinates] = node_coordinates
        except Exception:
            trb_msg = traceback.format_exc()
            msg = "A failure in {0} was caught:\n{1}"
            logging.warning(msg.format("_read_geometry_data, node_coordinates", trb_msg))
        finally:
            position += section_word_length * self.wordsize

        # solid data
        section_word_length = 9 * self.header['nel8']
        try:
            elem_solid_data = \
                self.bb.read_ndarray(position,
                                     section_word_length * self.wordsize,
                                     1,
                                     self.itype)\
                .reshape((self.header['nel8'], 9))
            solid_connectivity = elem_solid_data[:, :8]
            solid_part_indexes = elem_solid_data[:, 8]
            self.arrays[arraytype.element_solid_node_indexes] = solid_connectivity - FORTRAN_OFFSET
            self.arrays[arraytype.element_solid_part_indexes] = solid_part_indexes - FORTRAN_OFFSET
        except Exception:
            trb_msg = traceback.format_exc()
            msg = "A failure in {0} was caught:\n{1}"
            logging.warning(msg.format("_read_geometry_data, solids_geometry", trb_msg))
        finally:
            position += section_word_length * self.wordsize

        # ten node solids extra nodes
        if self.header["has_nel10"]:
            section_word_length = 2 * self.header["nel8"]
            try:
                self.arrays[arraytype.element_solid_extra_nodes] = \
                    elem_solid_data = \
                    self.bb.read_ndarray(position,
                                         section_word_length * self.wordsize,
                                         1,
                                         self.itype)\
                    .reshape((self.header['nel8'], 2))
            except Exception:
                trb_msg = traceback.format_exc()
                msg = "A failure in {0} was caught:\n{1}"
                logging.warning(msg.format("_read_geometry_data, ten_node_solids", trb_msg))
            finally:
                position += section_word_length * self.wordsize

        # 8 node thick shells
        section_word_length = 9 * self.header['nelth']
        try:
            elem_tshell_data = self.bb.read_ndarray(
                position, section_word_length * self.wordsize, 1, self.itype).reshape((self.header['nelth'], 9))
            self.arrays[arraytype.element_tshell_node_indexes] = elem_tshell_data[:, :8] - FORTRAN_OFFSET
            self.arrays[arraytype.element_tshell_part_indexes] = elem_tshell_data[:, 8] - FORTRAN_OFFSET
        except Exception:
            trb_msg = traceback.format_exc()
            msg = "A failure in {0} was caught:\n{1}"
            logging.warning(msg.format("_read_geometry_data, tshells_geometry", trb_msg))
        finally:
            position += section_word_length * self.wordsize

        # beams
        section_word_length = 6 * self.header['nel2']
        try:
            elem_beam_data = self.bb.read_ndarray(
                position,
                section_word_length * self.wordsize,
                1, self.itype).reshape((self.header['nel2'], 6))
            self.arrays[arraytype.element_beam_part_indexes] = elem_beam_data[:, 5] - FORTRAN_OFFSET
            self.arrays[arraytype.element_beam_node_indexes] = elem_beam_data[:, :5] - FORTRAN_OFFSET
        except Exception:
            trb_msg = traceback.format_exc()
            msg = "A failure in {0} was caught:\n{1}"
            logging.warning(msg.format("_read_geometry_data, beams_geometry", trb_msg))
        finally:
            position += section_word_length * self.wordsize

        # shells
        section_word_length = 5 * self.header['nel4']
        try:
            elem_shell_data = self.bb.read_ndarray(
                position, section_word_length * self.wordsize, 1, self.itype).reshape((self.header['nel4'], 5))
            self.arrays[arraytype.element_shell_node_indexes] = elem_shell_data[:, :4] - FORTRAN_OFFSET
            self.arrays[arraytype.element_shell_part_indexes] = elem_shell_data[:, 4] - FORTRAN_OFFSET
        except Exception:
            trb_msg = traceback.format_exc()
            msg = "A failure in {0} was caught:\n{1}"
            logging.warning(msg.format("_read_geometry_data, shells_geometry", trb_msg))
        finally:
            position += section_word_length * self.wordsize

        # update word position
        self.geometry_section_size = position

        logging.debug("_read_geometry_data end at byte {}".format(
            self.geometry_section_size))

    def _read_user_ids(self):

        if not self.bb:
            return

        if self.header['narbs'] <= 0:
            return

        logging.debug("_read_user_ids start at byte {}".format(
            self.geometry_section_size))

        position = self.geometry_section_size

        # safety
        original_position = position
        blocksize = self.header["narbs"] * self.wordsize

        try:
            numbering_words = {
                'nsort': (position, self.itype),
                'nsrh': (position + 1 * self.wordsize, self.itype),
                'nsrb': (position + 2 * self.wordsize, self.itype),
                'nsrs': (position + 3 * self.wordsize, self.itype),
                'nsrt': (position + 4 * self.wordsize, self.itype),
                'nsortd': (position + 5 * self.wordsize, self.itype),
                'nsrhd': (position + 6 * self.wordsize, self.itype),
                'nsrbd': (position + 7 * self.wordsize, self.itype),
                'nsrsd': (position + 8 * self.wordsize, self.itype),
                'nsrtd': (position + 9 * self.wordsize, self.itype),
            }

            extra_numbering_words = {
                'nsrma': (position + 10 * self.wordsize, self.itype),
                'nsrmu': (position + 11 * self.wordsize, self.itype),
                'nsrmp': (position + 12 * self.wordsize, self.itype),
                'nsrtm': (position + 13 * self.wordsize, self.itype),
                'numrbs': (position + 14 * self.wordsize, self.itype),
                'nmmat': (position + 15 * self.wordsize, self.itype),
            }

            numbering_header = self._read_words(numbering_words)
            position += len(numbering_words) * self.wordsize

            if numbering_header['nsort'] < 0:

                # read extra header
                self._read_words(extra_numbering_words, numbering_header)
                position += len(extra_numbering_words) * self.wordsize

                # correct nsort
                numbering_header['nsort'] = abs(
                    numbering_header['nsort'])
                numbering_header['arbitrary_material_numbering'] = True

            self.header['numbering_header'] = numbering_header

            # node ids
            array_length = numbering_header['nsortd'] * self.wordsize
            self.arrays[arraytype.node_ids] = self.bb.read_ndarray(
                position, array_length, 1, self.itype)
            position += array_length
            # solid ids
            array_length = self.header['nel8'] * self.wordsize
            self.arrays[arraytype.element_solid_ids] = self.bb.read_ndarray(
                position, array_length, 1, self.itype)
            position += array_length
            # beam ids
            array_length = self.header['nel2'] * self.wordsize
            self.arrays[arraytype.element_beam_ids] = self.bb.read_ndarray(
                position, array_length, 1, self.itype)
            position += array_length
            # shell ids
            array_length = self.header['nel4'] * self.wordsize
            self.arrays[arraytype.element_shell_ids] = self.bb.read_ndarray(
                position, array_length, 1, self.itype)
            position += array_length
            # tshell ids
            array_length = self.header['nelth'] * self.wordsize
            self.arrays[arraytype.element_tshell_ids] = self.bb.read_ndarray(
                position, array_length, 1, self.itype)
            position += array_length

            # part ids
            #
            # this makes no sense but materials are output three times at this section
            # but the length of the array (nmmat) is only output if nsort < 0. In
            # the other case the length is unknown ...
            #
            # Bugfix:
            # The material arrays (three times) are always output, even if nsort < 0
            # which means they are not used. Quite confusing, especially since nmmat
            # is output in the main header and numbering header.
            #
            if 'nmmat' in numbering_header:

                if numbering_header["nmmat"] != self.header["nmmat"]:
                    err_msg = "nmmat in the file header ({}) and in the numbering header ({}) are inconsistent."
                    raise RuntimeError(err_msg.format(
                        self.header["nmmat"], numbering_header["nmmat"]))

                array_length = numbering_header['nmmat'] * self.wordsize

                self.arrays[arraytype.part_ids] = self.bb.read_ndarray(
                    position, numbering_header['nmmat'] * self.wordsize, 1, self.itype)
                position += numbering_header["nmmat"] * self.wordsize

                self.arrays[arraytype.part_ids_unordered] = self.bb.read_ndarray(
                    position, numbering_header['nmmat'] * self.wordsize, 1, self.itype)
                position += numbering_header["nmmat"] * self.wordsize

                self.arrays[arraytype.part_ids_cross_references] = self.bb.read_ndarray(
                    position, numbering_header['nmmat'] * self.wordsize, 1, self.itype)
                position += numbering_header["nmmat"] * self.wordsize

            else:
                position += 3 * self.header["nmmat"] * self.wordsize

        except Exception:
            # print info
            trb_msg = traceback.format_exc()
            msg = "A failure in {0} was caught:\n{1}"
            logging.warning(msg.format("_read_user_ids", trb_msg))

            # fix position
            position = original_position + blocksize

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
        for _ in range(rigid_body_description_header['nrigid']):

            rigid_body_info = {
                # rigid body part internal number
                'mrigid': self.bb.read_number(position, self.itype),
                # number of nodes in rigid body
                'numnodr': self.bb.read_number(position + self.wordsize,
                                               self.itype),
            }
            position += 2 * self.wordsize

            # internal node number of rigid body
            array_length = rigid_body_info['numnodr'] * self.wordsize
            rigid_body_info['noder'] = self.bb.read_ndarray(position,
                                                            array_length,
                                                            1, self.itype)
            position += array_length

            # number of active (non-rigid) nodes
            rigid_body_info['numnoda'] = self.bb.read_number(position,
                                                             self.itype)
            position += self.wordsize

            # internal node numbers of active nodes
            array_length = rigid_body_info['numnoda'] * self.wordsize
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

        # save arrays
        rigid_body_n_nodes = []
        rigid_body_part_indexes = []
        rigid_body_n_active_nodes = []
        rigid_body_node_indexes_list = []
        rigid_body_active_node_indexes_list = []
        for rigid_body_info in rigid_bodies:
            rigid_body_part_indexes.append(rigid_body_info["mrigid"])
            rigid_body_n_nodes.append(rigid_body_info["numnodr"])
            rigid_body_node_indexes_list.append(rigid_body_info["noder"] - FORTRAN_OFFSET)
            rigid_body_n_active_nodes.append(rigid_body_info["numnoda"])
            rigid_body_active_node_indexes_list.append(
                rigid_body_info["nodea"])

        self.arrays[arraytype.rigid_body_part_indexes] = \
            np.array(rigid_body_part_indexes, dtype=self.itype) - FORTRAN_OFFSET
        self.arrays[arraytype.rigid_body_n_nodes] = \
            np.array(rigid_body_n_nodes, dtype=self.itype)
        self.arrays[arraytype.rigid_body_n_active_nodes] = \
            np.array(rigid_body_n_active_nodes, dtype=self.itype)
        self.arrays[arraytype.rigid_body_node_indexes_list] = \
            rigid_body_node_indexes_list
        self.arrays[arraytype.rigid_body_active_node_indexes_list] = \
            rigid_body_active_node_indexes_list

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

        array_length = self.header["nmsph"] * self.wordsize * 2
        try:
            # read info array
            sph_node_matlist = self.bb.read_ndarray(
                position, array_length, 1, self.itype).reshape((self.header["nmsph"], 2))

            # save array
            self.arrays[arraytype.sph_node_indexes] = sph_node_matlist[:, 0] - FORTRAN_OFFSET
            self.arrays[arraytype.sph_node_material_index] = sph_node_matlist[:, 1] - FORTRAN_OFFSET

        except Exception:
            # print info
            trb_msg = traceback.format_exc()
            msg = "A failure in {0} was caught:\n{1}"
            logging.warning(msg.format("_read_sph_node_and_material_list", trb_msg))

        finally:
            # update position
            self.geometry_section_size += array_length

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
        if ngeom not in [4, 5]:
            raise RuntimeError(
                "variable ngeom in the airbag header must be 4 or 5.")

        original_position = position
        blocksize = airbag_header["npartgas"] * ngeom * self.wordsize
        try:

            # extract geometry as a single array
            array_length = blocksize
            particle_geom_data = self.bb.read_ndarray(position, array_length, 1, self.itype)\
                .reshape((airbag_header["npartgas"], ngeom))
            position += array_length

            # store arrays
            self.arrays[arraytype.airbags_first_particle_id] = particle_geom_data[:, 0]
            self.arrays[arraytype.airbags_n_particles] = particle_geom_data[:, 1]
            self.arrays[arraytype.airbags_ids] = particle_geom_data[:, 2]
            self.arrays[arraytype.airbags_n_gas_mixtures] = particle_geom_data[:, 3]
            if ngeom == 5:
                self.arrays[arraytype.airbags_n_chambers] = particle_geom_data[:, 4]

        except Exception:
            # print info
            trb_msg = traceback.format_exc()
            msg = "A failure in {0} was caught:\n{1}"
            logging.warning(msg.format("_read_particle_geometry_data", trb_msg))

            # fix position
            position = original_position + blocksize

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
            'nseg': (position + 1 * self.wordsize, self.itype),
            'nsurf': (position + 2 * self.wordsize, self.itype),
            'motion': (position + 3 * self.wordsize, self.itype),
        }

        rigid_road_header = self._read_words(rigid_road_surface_words)
        position += 4 * self.wordsize

        # node ids
        array_length = rigid_road_header["nnode"] * self.wordsize
        rigid_road_node_ids = self.bb.read_ndarray(
            position, array_length, 1, self.itype)
        self.arrays[arraytype.rigid_road_node_ids] = rigid_road_node_ids
        position += array_length

        # node xyz
        array_length = rigid_road_header["nnode"] * 3 * self.wordsize
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

        # this array is created since the array database requires
        # constant sized arrays and we dump all segments into one
        # array. In order to distinguish which segment
        # belongs to which road, this new array keeps track of it
        rigid_road_segment_road_id = []

        # n_total_segments = 0
        for i_surf in range(rigid_road_header["nsurf"]):

            # surface id
            surf_id = self.bb.read_number(position, self.itype)
            position += self.wordsize
            rigid_road_ids[i_surf] = surf_id

            # number of segments of surface
            surf_nseg = self.bb.read_number(
                position + 1 * self.wordsize, self.itype)
            position += self.wordsize
            rigid_road_nsegments[i_surf] = surf_nseg

            # count total segments
            # n_total_segments += surf_nseg

            # node ids of surface segments
            array_length = 4 * surf_nseg * self.wordsize
            surf_segm_node_ids = self.bb.read_ndarray(position,
                                                      array_length,
                                                      1, self.itype)\
                .reshape((surf_nseg, 4))
            position += array_length
            rigid_road_segment_node_ids.append(surf_segm_node_ids)

            # remember road id for segments
            rigid_road_segment_road_id += [surf_id] * surf_nseg

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
            array_length = 2 * self.header['nel8'] * self.wordsize
            try:
                array = self.bb.read_ndarray(
                    position, array_length, 1, self.itype).reshape((self.header['nel8'], 2))
                self.arrays[arraytype.element_solid_node10_extra_node_indexes] = array - FORTRAN_OFFSET
            except Exception:
                trb_msg = traceback.format_exc()
                msg = "A failure in {0} was caught:\n{1}"
                logging.warning(msg.format("_read_extra_node_connectivity, solid10", trb_msg))
            finally:
                position += array_length

        # extra 4 node connectivity for 8 node shell elements
        if self.header["nel48"] > 0:
            array_length = 5 * self.header['nel48'] * self.wordsize
            try:
                array = self.bb.read_ndarray(
                    position, array_length, 1, self.itype).reshape((self.header['nel48'], 5))
                self.arrays[arraytype.element_shell_node8_element_index] = \
                    array[:, 0] - FORTRAN_OFFSET
                self.arrays[arraytype.element_shell_node8_extra_node_indexes] = \
                    array[:, 1:] - FORTRAN_OFFSET
            except Exception:
                trb_msg = traceback.format_exc()
                msg = "A failure in {0} was caught:\n{1}"
                logging.warning(msg.format("_read_extra_node_connectivity, shell8", trb_msg))
            finally:
                position += array_length

        # extra 12 node connectivity for 20 node solid elements
        if "nel20" in self.header and self.header["nel20"] > 0:
            array_length = 13 * self.header['nel20'] * self.wordsize
            try:
                array = self.bb.read_ndarray(
                    position, array_length, 1, self.itype).reshape((self.header['nel20'], 13))
                self.arrays[arraytype.element_solid_node20_element_index] = \
                    array[:, 0] - FORTRAN_OFFSET
                self.arrays[arraytype.element_solid_node20_extra_node_indexes] = \
                    array[:, 1:] - FORTRAN_OFFSET
            except Exception:
                trb_msg = traceback.format_exc()
                msg = "A failure in {0} was caught:\n{1}"
                logging.warning(msg.format("_read_extra_node_connectivity, solid20", trb_msg))
            finally:
                position += array_length

        # extra 19 node connectivity for 27 node solid elements
        if "nel27" in self.header and self.header["nel27"] > 0:
            array_length = 20 * self.header['nel27'] * self.wordsize
            try:
                array = self.bb.read_ndarray(
                    position, array_length, 1, self.itype).reshape((self.header['nel27'], 20))
                self.arrays[arraytype.element_solid_node27_element_index] = \
                    array[:, 0] - FORTRAN_OFFSET
                self.arrays[arraytype.element_solid_node27_extra_node_indexes] = \
                    array[:, 1:] - FORTRAN_OFFSET
            except Exception:
                trb_msg = traceback.format_exc()
                msg = "A failure in {0} was caught:\n{1}"
                logging.warning(msg.format("_read_extra_node_connectivity, solid27", trb_msg))
            finally:
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

        # Security
        #
        # we try to read the titles ahead. If dyna writes multiple files
        # then the first file is geometry only thus failing here has no
        # impact on further state reading.
        # If though states are compressed into the first file then we are
        # in trouble here even when catching here.
        try:
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

                    array_length = 18 * titles_wordsize
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
                                           ("titles", "S" + str(18 * titles_wordsize))])
                    array_length = (self.wordsize + 18 * titles_wordsize) * entry_count
                    tmp_arrays = self.bb.read_ndarray(
                        position, array_length, 1, array_type)
                    position += array_length

                    # save stuff
                    if current_ntype == 90001:
                        self.arrays[arraytype.part_titles_ids] = tmp_arrays["ids"]
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
                    array_length = 20 * titles_wordsize * self.header["nline"]
                    d3prop_keywords = self.bb.read_ndarray(
                        position, array_length, 1, np.dtype("S" + str(titles_wordsize * 20)))
                    position += array_length

                    # save
                    self.arrays["d3prop_keywords"] = d3prop_keywords

                # not sure whether there is an eof file here
                # do not have a test file to check ...
                if self._is_end_of_file_marker(position):
                    position += self.wordsize

                # next one
                if self.bb.size <= position:
                    break
                current_ntype = self.bb.read_number(position, self.itype)

        except Exception:
            trb_msg = traceback.format_exc()
            msg = "A failure in {0} was caught:\n{1}"
            logging.warning(
                msg.format("_read_header_part_contact_interface_titles", trb_msg))

        # remember position
        self.geometry_section_size = position
        logging.debug("_read_header_part_contact_interface_titles end at byte {}".format(
            self.geometry_section_size))

    def _read_states_allocate_arrays(self, n_states, array_names, array_dict=None) -> dict:
        ''' Allocate the state arrays

        Parameters
        ----------
        n_states : `int`
            number of states to allocate memory for
        array_names : `list` of `str`
            names of state arrays to allocate
        array_dict : `dict`
            dictionary to allocate arrays into

        Returns
        -------
        array_dict : `dict`
            dictionary in which the arrays have been allocated.
            This is the same array as in the input.
        '''

        # do the argument thing
        if array_dict is None:
            array_dict = {}

        # (1) ARRAY SHAPES
        # general
        n_dim = self.header["ndim"]
        # parts
        n_parts = self._get_n_parts()
        # rigid walls
        n_rigid_walls = self._get_n_rigid_walls()
        # nodes
        n_nodes = self.header["numnp"]
        # solids
        n_solids = self.header["nel8"]
        n_solids_thermal_vars = self.header["nt3d"]
        n_solids_strain_vars = 6 * self.header["istrn"]
        n_solids_history_vars = self.header["neiph"] - n_solids_strain_vars
        # thick shells
        n_tshells = self.header["nelth"]
        n_tshells_history_vars = self.header["neips"]
        n_tshells_layers = self.header["maxint"]
        # beams
        n_beams = self.header["nel2"]
        n_beams_history_vars = self.header["neipb"]
        n_beam_vars = self.header["nv1d"]
        n_beams_layers = int((-3 * n_beams_history_vars + n_beam_vars - 6) /
                             (n_beams_history_vars + 5))
        # shells
        n_shells = self.header["nel4"]
        n_shells_reduced = self.header["nel4"] - self.header["numrbe"]
        n_shell_layers = self.header["maxint"]
        n_shell_history_vars = self.header["neips"]
        # sph
        allocate_sph = "nmsph" in self.header
        n_sph_particles = self.header["nmsph"] if allocate_sph else 0
        # airbags
        allocate_airbags = "airbag" in self.header
        n_airbags = self.header["airbag"]["npartgas"] if allocate_airbags else 0
        n_airbag_particles = self.header["airbag"]["npart"] if allocate_airbags else 0
        # rigid roads
        allocate_rigid_roads = "rigid_road" in self.header
        n_roads = self.header["rigid_road"]["nsurf"] if allocate_rigid_roads else 0
        # rigid bodies
        n_rigid_bodies = self.header["nrigid"] if "nrigid" in self.header else 0

        # dictionary to lookup array types
        state_array_shapes = {
            # global
            arraytype.global_timesteps: [n_states],
            arraytype.global_kinetic_energy: [n_states],
            arraytype.global_internal_energy: [n_states],
            arraytype.global_total_energy: [n_states],
            arraytype.global_velocity: [n_states, 3],
            # parts
            arraytype.part_internal_energy: [n_states, n_parts],
            arraytype.part_kinetic_energy: [n_states, n_parts],
            arraytype.part_velocity: [n_states, n_parts, 3],
            arraytype.part_mass: [n_states, n_parts],
            arraytype.part_hourglass_energy: [n_states, n_parts],
            # rigid wall
            arraytype.rigid_wall_force: [n_states, n_rigid_walls],
            arraytype.rigid_wall_position: [n_states, n_rigid_walls, 3],
            # nodes
            arraytype.node_temperature: [n_states, n_nodes, 3]
            if self.header["it"] == 3 \
            else [n_states, n_nodes],
            arraytype.node_heat_flux: [n_states, n_nodes, 3],
            arraytype.node_mass_scaling: [n_states, n_nodes],
            arraytype.node_displacement: [n_states, n_nodes, n_dim],
            arraytype.node_velocity: [n_states, n_nodes, n_dim],
            arraytype.node_acceleration: [n_states, n_nodes, n_dim],
            # solids
            arraytype.element_solid_thermal_data: [n_states, n_solids, n_solids_thermal_vars],
            arraytype.element_solid_stress: [n_states, n_solids, 6],
            arraytype.element_solid_effective_plastic_strain: [n_states, n_solids],
            arraytype.element_solid_history_variables: \
            [n_states, n_solids, n_solids_history_vars],
            arraytype.element_solid_strain: [n_states, n_solids, 6],
            arraytype.element_solid_is_alive: [n_states, n_solids],
            # thick shells
            arraytype.element_tshell_stress: [n_states, n_tshells, n_tshells_layers, 6],
            arraytype.element_tshell_effective_plastic_strain: \
            [n_states, n_tshells, n_tshells_layers],
            arraytype.element_tshell_history_variables: \
            [n_states, n_tshells, n_tshells_layers, n_tshells_history_vars],
            arraytype.element_tshell_strain: [n_states, n_tshells, 2, 6],
            arraytype.element_tshell_is_alive: [n_states, n_tshells],
            # beams
            arraytype.element_beam_axial_force: [n_states, n_beams],
            arraytype.element_beam_shear_force: [n_states, n_beams, 2],
            arraytype.element_beam_bending_moment: [n_states, n_beams, 2],
            arraytype.element_beam_torsion_moment: [n_states, n_beams],
            arraytype.element_beam_shear_stress: [n_states, n_beams, n_beams_layers, 2],
            arraytype.element_beam_axial_stress: [n_states, n_beams, n_beams_layers],
            arraytype.element_beam_plastic_strain: [n_states, n_beams, n_beams_layers],
            arraytype.element_beam_axial_strain: [n_states, n_beams, n_beams_layers],
            arraytype.element_beam_history_vars: \
            [n_states, n_beams, n_beams_layers + 3, n_beams_history_vars],
            arraytype.element_beam_is_alive: [n_states, n_beams],
            # shells
            arraytype.element_shell_stress: [n_states, n_shells_reduced, n_shell_layers, 6],
            arraytype.element_shell_effective_plastic_strain: \
            [n_states, n_shells_reduced, n_shell_layers],
            arraytype.element_shell_history_vars: \
            [n_states, n_shells_reduced, n_shell_layers, n_shell_history_vars],
            arraytype.element_shell_bending_moment: [n_states, n_shells_reduced, 3],
            arraytype.element_shell_shear_force: [n_states, n_shells_reduced, 2],
            arraytype.element_shell_normal_force: [n_states, n_shells_reduced, 3],
            arraytype.element_shell_thickness: [n_states, n_shells_reduced],
            arraytype.element_shell_unknown_variables: [n_states, n_shells_reduced, 2],
            arraytype.element_shell_internal_energy: [n_states, n_shells_reduced],
            arraytype.element_shell_strain: [n_states, n_shells_reduced, 2, 6],
            arraytype.element_shell_is_alive: [n_states, n_shells],
            # sph
            arraytype.sph_deletion: [n_states, n_sph_particles],
            arraytype.sph_radius: [n_states, n_sph_particles],
            arraytype.sph_pressure: [n_states, n_sph_particles],
            arraytype.sph_stress: [n_states, n_sph_particles, 6],
            arraytype.sph_effective_plastic_strain: [n_states, n_sph_particles],
            arraytype.sph_density: [n_states, n_sph_particles],
            arraytype.sph_internal_energy: [n_states, n_sph_particles],
            arraytype.sph_n_neighbors: [n_states, n_sph_particles],
            arraytype.sph_strain: [n_states, n_sph_particles, 6],
            arraytype.sph_mass: [n_states, n_sph_particles],
            # airbag
            arraytype.airbag_n_active_particles: [n_states, n_airbags],
            arraytype.airbag_bag_volume: [n_states, n_airbags],
            arraytype.airbag_particle_gas_id: [n_states, n_airbag_particles],
            arraytype.airbag_particle_chamber_id: [n_states, n_airbag_particles],
            arraytype.airbag_particle_leakage: [n_states, n_airbag_particles],
            arraytype.airbag_particle_mass: [n_states, n_airbag_particles],
            arraytype.airbag_particle_radius: [n_states, n_airbag_particles],
            arraytype.airbag_particle_spin_energy: [n_states, n_airbag_particles],
            arraytype.airbag_particle_translation_energy: [n_states, n_airbag_particles],
            arraytype.airbag_particle_nearest_segment_distance: [n_states, n_airbag_particles],
            arraytype.airbag_particle_position: [n_states, n_airbag_particles, 3],
            arraytype.airbag_particle_velocity: [n_states, n_airbag_particles, 3],
            # rigid road
            arraytype.rigid_road_displacement: [n_states, n_roads, 3],
            arraytype.rigid_road_velocity: [n_states, n_roads, 3],
            # rigid body
            arraytype.rigid_body_coordinates: [n_states, n_rigid_bodies, 3],
            arraytype.rigid_body_rotation_matrix: [n_states, n_rigid_bodies, 9],
            arraytype.rigid_body_velocity: [n_states, n_rigid_bodies, 3],
            arraytype.rigid_body_rot_velocity: [n_states, n_rigid_bodies, 3],
            arraytype.rigid_body_acceleration: [n_states, n_rigid_bodies, 3],
            arraytype.rigid_body_rot_acceleration: [n_states, n_rigid_bodies, 3],
        }

        # only allocate available arrays
        if array_names is None:
            array_names = arraytype.get_state_array_names()

        # BUGFIX
        # These arrays are actually integer types, all other state arrays
        # are floats
        int_state_arrays = [
            arraytype.airbag_n_active_particles,
            arraytype.airbag_particle_gas_id,
            arraytype.airbag_particle_chamber_id,
            arraytype.airbag_particle_leakage,
        ]

        # (2) ALLOCATE ARRAYS
        # this looper allocates the arrays specified by the user.
        for array_name in array_names:

            array_dtype = self.ftype \
                if array_name not in int_state_arrays\
                else self.itype

            if array_name in state_array_shapes:
                array_dict[array_name] = np.empty(
                    state_array_shapes[array_name], dtype=array_dtype)
            else:
                raise ValueError("Array '{0}' is not a state array. Please try one of: {1}".format(
                    array_name, list(state_array_shapes.keys())))

        return array_dict

    def _read_states_transfer_memory(self, i_state: int, buffer_array_dict: dict, master_array_dict: dict):
        ''' Transfers the memory from smaller buffer arrays with only a few timesteps into the major one

        Parameters
        ----------
        i_state : `int`
            current state index
        buffer_array_dict : `dict`
            dict with arrays of only a few timesteps
        master_array_dict : `dict`
            dict with the parent master arrays

        Notes
        -----
            If an array in the master dict is not found in the buffer dict
            then this array is set to `None`.
        '''

        state_array_names = arraytype.get_state_array_names()

        arrays_to_delete = []
        for array_name, array in master_array_dict.items():

            # copy memory to big array
            if array_name in buffer_array_dict:
                buffer_array = buffer_array_dict[array_name]
                n_states_buffer_array = buffer_array.shape[0]
                array[i_state:i_state + n_states_buffer_array] = buffer_array
            else:
                # remove unnecesary state arrays (not geometry arrays!)
                # we "could" deal with this in the allocate function
                # by not allocating them but this would replicate code
                # in the reading functions
                if array_name in state_array_names:
                    arrays_to_delete.append(array_name)

        for array_name in arrays_to_delete:
            del master_array_dict[array_name]

    @staticmethod
    def _compute_n_bytes_per_state(header: dict, wordsize: int) -> int:
        ''' Computes the number of bytes for every state

        Parameters
        ----------
        header : `dict`
            header information of a d3plot
        wordsize : `int`
            size of every word in bytes in the d3plot

        Returns
        -------
        n_bytes_per_state : `int`
            number of bytes of every state
        '''

        if not header:
            return 0

        # timestep
        timestep_offset = 1 * wordsize
        # global vars
        global_vars_offset = header["nglbv"] * wordsize
        # node vars
        n_node_vars = (header["iu"] +
                       header["iv"] +
                       header["ia"])\
            * header["ndim"]

        if header["it"] == 1:
            n_node_temp_vars = 1
        elif header["it"] == 2:
            n_node_temp_vars = 4
        elif header["it"] == 3:
            n_node_temp_vars = 6
        else:
            n_node_temp_vars = 0

        if header["has_mass_scaling"]:
            n_node_temp_vars += 1

        node_data_offset = int(n_node_vars + n_node_temp_vars) * \
            int(header["numnp"]) * int(wordsize)
        # thermal shit
        therm_data_offset = header["nt3d"] * header["nel8"] * wordsize \
            if "nt3d" in header else 0
        # solids
        solid_offset = header["nel8"] * header["nv3d"] * wordsize
        # tshells
        tshell_offset = header["nelth"] * header["nv3dt"] * wordsize
        # beams
        beam_offset = header["nel2"] * header["nv1d"] * wordsize
        # shells
        shell_offset = (
            header["nel4"] - header["numrbe"]) * header["nv2d"] * wordsize
        # Manual
        # "NOTE: This CFDDATA is no longer output by ls-dyna."
        cfd_data_offset = 0
        # sph
        sph_offset = header["nmsph"] * \
            header["num_sph_vars"] * wordsize
        # deleted nodes and elems ... or nothing
        elem_deletion_offset = 0
        if header["mdlopt"] == 1:
            elem_deletion_offset = header["numnp"] * wordsize
        elif header["mdlopt"] == 2:
            elem_deletion_offset = (header["nel2"] +
                                    header["nel4"] +
                                    header["nel8"] +
                                    header["nelth"]) * wordsize
        elif header["mdlopt"] == 0:
            pass
        else:
            err_msg = "Unexpected value of mdlop: {}, expected was 0, 1 or 2."
            raise RuntimeError(err_msg.format(header["mdlopt"]))
        # airbag particle offset
        if "airbag" in header:
            particle_state_offset = \
                (header["airbag"]["npartgas"] * header["airbag"]["nstgeom"] +
                 header["airbag"]["npart"] * header["airbag"]["nvar"]) \
                * wordsize
        else:
            particle_state_offset = 0
        # rigid road stuff whoever uses this
        road_surface_offset = header["rigid_road"]["nsurf"] * 6 * wordsize \
            if "rigid_road" in header else 0
        # rigid body motion data
        if header["has_rigid_body_data"]:
            n_rigids = header["nrigid"]
            n_rigid_vars = 12 if header["has_reduced_rigid_body_data"] else 24
            rigid_body_motion_offset = n_rigids * n_rigid_vars * wordsize
        else:
            rigid_body_motion_offset = 0
        # TODO
        extra_data_offset = 0

        n_bytes_per_state = timestep_offset \
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

        return n_bytes_per_state

    def _read_states(self):
        ''' Read the states from the d3plot
        '''

        if not self.bb:
            self.header["n_timesteps"] = 0
            return

        logging.debug("-" * 80)
        logging.debug("_read_states with geom offset {}".format(
            self.geometry_section_size))

        # (0) OFFSETS
        bytes_per_state = D3plot._compute_n_bytes_per_state(
            self.header, self.wordsize)
        logging.debug("bytes_per_state: {}".format(bytes_per_state))

        # load the memory from the files
        # part_titles_size = 0
        if self.header["use_femzip"]:

            # part_titles_size = next(self.bb_generator)
            next(self.bb_generator)

            # end marker + part section size
            # + 1!! dont why, but one day we will
            # if os.name == "posix":
            #     bytes_per_state += (part_titles_size + 1) * self.wordsize
            # elif os.name == "nt":
            #     # end marker is always in here
            #     bytes_per_state += 1 * self.wordsize

            # BUGFIX
            # femzip version 10 always omits part titles ... finally
            bytes_per_state += 1 * self.wordsize

        # (1) READ STATE DATA
        # TODO we load the first file twice which is already in memory!
        n_states, buffered_reading = next(self.bb_generator)

        # determine whether to transfer arrays
        if not buffered_reading:
            transfer_arrays = False
        else:
            transfer_arrays = True
        if self.state_array_filter:
            transfer_arrays = True

        # arrays need to be preallocated if we transfer them
        if transfer_arrays:
            self._read_states_allocate_arrays(
                n_states, self.state_array_filter, self.arrays)

        i_state = 0
        for bb_states, n_states in self.bb_generator:

            # dictionary to store the temporary, partial arrays
            # if we do not transfer any arrays we store them directly
            # in the classes main dict
            array_dict = {} if transfer_arrays else self.arrays

            # sometimes there is just a geometry in the file
            if n_states == 0:
                continue

            # state data as array
            array_length = int(n_states) * int(bytes_per_state)
            state_data = bb_states.read_ndarray(0, array_length, 1, self.ftype)
            state_data = state_data.reshape((n_states, -1))

            # BUGFIX: changed in femzip 10 without telling me :/
            # parts are not written in front of every state anymore

            # here -1, also no idea why
            # if os.name == "nt":
            #     var_index = 0
            # else:
            #     var_index = 0 if not self.header["use_femzip"] \
            #         else (part_titles_size - 1)
            var_index = 0

            # global state header
            var_index = self._read_states_global_vars(
                state_data, var_index, array_dict)

            # node data
            var_index = self._read_states_nodes(
                state_data, var_index, array_dict)

            # thermal solid data
            var_index = self._read_states_solids_thermal(
                state_data, var_index, array_dict)

            # cfddata was originally here

            # solids
            var_index = self._read_states_solids(
                state_data, var_index, array_dict)

            # tshells
            var_index = self._read_states_tshell(
                state_data, var_index, array_dict)

            # beams
            var_index = self._read_states_beams(
                state_data, var_index, array_dict)

            # shells
            var_index = self._read_states_shell(
                state_data, var_index, array_dict)

            # element and node deletion info
            var_index = self._read_states_is_alive(
                state_data, var_index, array_dict)

            # sph
            var_index = self._read_states_sph(
                state_data, var_index, array_dict)

            # airbag particle data
            var_index = self._read_states_airbags(
                state_data, var_index, array_dict)

            # road surface data
            var_index = self._read_states_road_surfaces(
                state_data, var_index, array_dict)

            # rigid body motion
            var_index = self._read_states_rigid_body_motion(
                state_data, var_index, array_dict)

            # transfer memory
            if transfer_arrays:
                self._read_states_transfer_memory(
                    i_state, array_dict, self.arrays)

            # increment state counter
            i_state += n_states
            self.header["n_timesteps"] = i_state

        if transfer_arrays:
            self.bb = None
            self.bb_states = None

    def _read_states_global_vars(self,
                                 state_data: np.ndarray,
                                 var_index: int,
                                 array_dict: dict) -> int:
        ''' Read the global vars for the state

        Parameters
        ----------
        state_data : np.ndarray
            array with entire state data
        var_index : int
            variable index in the state data array
        array_dict : `dict`
            dictionary to store the loaded arrays in

        Returns
        -------
        var_index : int
            updated variable index after reading the section
        '''

        logging.debug(
            "_read_states_global_vars start at var_index {}".format(var_index))

        n_states = state_data.shape[0]

        original_var_index = var_index
        try:
            # global stuff
            array_dict[arraytype.global_timesteps] = state_data[:, var_index + 0]
            array_dict[arraytype.global_kinetic_energy] = state_data[:, var_index + 1]
            array_dict[arraytype.global_internal_energy] = state_data[:, var_index + 2]
            array_dict[arraytype.global_total_energy] = state_data[:, var_index + 3]
            array_dict[arraytype.global_velocity] = \
                state_data[:, var_index + 4:var_index + 7]
            var_index += 7

            # part infos ... whoever calls this global data
            # n_parts = self._get_n_parts()
            n_parts = self.header["nmmat"]

            # part internal energy
            array_dict[arraytype.part_internal_energy] = \
                state_data[:, var_index:var_index + n_parts]
            var_index += n_parts

            # part kinetic energy
            array_dict[arraytype.part_kinetic_energy] = \
                state_data[:, var_index:var_index + n_parts]
            var_index += n_parts

            # part velocity
            array_dict[arraytype.part_velocity] = \
                state_data[:, var_index:var_index + 3 * n_parts]\
                .reshape((n_states, n_parts, 3))
            var_index += 3 * n_parts

            # part mass
            array_dict[arraytype.part_mass] = \
                state_data[:, var_index:var_index + n_parts]
            var_index += n_parts

            # part hourglass energy
            array_dict[arraytype.part_hourglass_energy] = \
                state_data[:, var_index:var_index + n_parts]
            var_index += n_parts

            # rigid walls
            previous_global_vars = (6 + 7 * n_parts)
            n_rigid_wall_vars = 4 if self.header["version"] >= 971 else 1
            # +1 is timestep which is not considered a global var ... seriously
            n_rigid_walls = (self.header["nglbv"] - previous_global_vars) // n_rigid_wall_vars
            self.header["n_rigid_walls"] = n_rigid_walls
            self.header["n_rigid_wall_vars"] = n_rigid_wall_vars
            if previous_global_vars + n_rigid_walls * n_rigid_wall_vars != self.header["nglbv"]:
                logging.warning(
                    "Bug while reading global data for rigid walls. Skipping this data.")
                var_index += self.header["nglbv"] - previous_global_vars
            else:

                # rigid wall force
                array_dict[arraytype.rigid_wall_force] = \
                    state_data[:, var_index:var_index + n_rigid_walls]
                var_index += n_rigid_walls

                # rigid wall position
                if n_rigid_wall_vars > 1:
                    array_dict[arraytype.rigid_wall_position] = \
                        state_data[:, var_index:var_index + 3 * n_rigid_walls]\
                        .reshape(n_states, n_rigid_walls, 3)
                    var_index += 3 * n_rigid_walls

        except Exception:
            # print
            trb_msg = traceback.format_exc()
            msg = "A failure in {0} was caught:\n{1}"
            logging.warning(
                msg.format("_read_states_global_vars", trb_msg))
            # fix var_index
            var_index = original_var_index + self.header["nglbv"]

        logging.debug(
            "_read_states_global_vars end at var_index {}".format(var_index))

        return var_index

    def _read_states_nodes(self,
                           state_data: np.ndarray,
                           var_index: int,
                           array_dict: dict) -> int:
        ''' Read the node data in the state sectio

        Parameters
        ----------
        state_data : np.ndarray
            array with entire state data
        var_index : int
            variable index in the state data array
        array_dict : `dict`
            dictionary to store the loaded arrays in

        Returns
        -------
        var_index : int
            updated variable index after reading the section
        '''

        if self.header["numnp"] <= 0:
            return var_index

        logging.debug(
            "_read_states_nodes start at var_index {}".format(var_index))

        n_dim = self.header["ndim"]
        n_states = state_data.shape[0]
        n_nodes = self.header["numnp"]

        # only node temperatures
        if self.header["it"] == 1:
            try:
                array_dict[arraytype.node_temperature] = \
                    state_data[:, var_index:var_index + n_nodes]
            except Exception:
                trb_msg = traceback.format_exc()
                msg = "A failure in {0} was caught:\n{1}"
                logging.warning(
                    msg.format("_read_states_nodes, node_temperatures", trb_msg))
            finally:
                var_index += n_nodes

        # node temperature and node flux
        if self.header["it"] == 2:
            try:
                array_dict[arraytype.node_temperature] = \
                    state_data[:, var_index:var_index + n_nodes]
            except Exception:
                trb_msg = traceback.format_exc()
                msg = "A failure in {0} was caught:\n{1}"
                logging.warning(
                    msg.format("_read_states_nodes, node_temperatures & node_heat_flux",
                               trb_msg))
            finally:
                var_index += n_nodes

            tmp_array = state_data[:, var_index:var_index + 3 * n_nodes]\
                .reshape((n_states, n_nodes, 3))
            array_dict[arraytype.node_heat_flux] = tmp_array
            var_index += 3 * n_nodes

        # 3 temperatures per node and node flux
        # temperatures at inner, middle and outer layer
        if self.header["it"] == 3:
            try:
                tmp_array = state_data[:, var_index:var_index + 3 * n_nodes]\
                    .reshape((n_states, n_nodes, 3))
                array_dict[arraytype.node_temperature] = tmp_array
            except Exception:
                trb_msg = traceback.format_exc()
                msg = "A failure in {0} was caught:\n{1}"
                logging.warning(
                    msg.format("_read_states_nodes, node_temperatures & node_heat_flux",
                               trb_msg))
            finally:
                var_index += 3 * n_nodes

            try:
                tmp_array = state_data[:, var_index:var_index + 3 * n_nodes]\
                    .reshape((n_states, n_nodes, 3))
                array_dict[arraytype.node_heat_flux] = tmp_array
            except Exception:
                trb_msg = traceback.format_exc()
                msg = "A failure in {0} was caught:\n{1}"
                logging.warning(
                    msg.format("_read_states_nodes, node_temperatures & node_heat_flux",
                               trb_msg))
            finally:
                var_index += 3 * n_nodes

        # mass scaling
        if self.header["has_mass_scaling"]:
            try:
                array_dict[arraytype.node_mass_scaling] = \
                    state_data[:, var_index:var_index + n_nodes]
            except Exception:
                trb_msg = traceback.format_exc()
                msg = "A failure in {0} was caught:\n{1}"
                logging.warning(
                    msg.format("_read_states_nodes, node_mass_scaling",
                               trb_msg))
            finally:
                var_index += n_nodes

        # displacement
        if self.header["iu"]:
            try:
                tmp_array = state_data[:, var_index:var_index + n_dim * n_nodes]\
                    .reshape((n_states, n_nodes, n_dim))
                array_dict[arraytype.node_displacement] = tmp_array
            except Exception:
                trb_msg = traceback.format_exc()
                msg = "A failure in {0} was caught:\n{1}"
                logging.warning(
                    msg.format("_read_states_nodes, node_displacement",
                               trb_msg))
            finally:
                var_index += n_dim * n_nodes

        # velocity
        if self.header["iv"]:
            try:
                tmp_array = state_data[:, var_index:var_index + n_dim * n_nodes]\
                    .reshape((n_states, n_nodes, n_dim))
                array_dict[arraytype.node_velocity] = tmp_array
            except Exception:
                trb_msg = traceback.format_exc()
                msg = "A failure in {0} was caught:\n{1}"
                logging.warning(
                    msg.format("_read_states_nodes, node_velocity",
                               trb_msg))
            finally:
                var_index += n_dim * n_nodes

        # acceleration
        if self.header["ia"]:
            try:
                tmp_array = state_data[:, var_index:var_index + n_dim * n_nodes]\
                    .reshape((n_states, n_nodes, n_dim))
                array_dict[arraytype.node_acceleration] = tmp_array
            except Exception:
                trb_msg = traceback.format_exc()
                msg = "A failure in {0} was caught:\n{1}"
                logging.warning(
                    msg.format("_read_states_nodes, node_acceleration",
                               trb_msg))
            finally:
                var_index += n_dim * n_nodes

        logging.debug(
            "_read_states_nodes end at var_index {}".format(var_index))

        return var_index

    def _read_states_solids_thermal(self,
                                    state_data: np.ndarray,
                                    var_index: int,
                                    array_dict: dict) -> int:
        ''' Read the thermal data for solids

        Parameters
        ----------
        state_data : np.ndarray
            array with entire state data
        var_index : int
            variable index in the state data array
        array_dict : `dict`
            dictionary to store the loaded arrays in

        Returns
        -------
        var_index : int
            updated variable index after reading the section
        '''

        if "nt3d" not in self.header:
            return var_index

        if self.header["nt3d"] <= 0:
            return var_index

        logging.debug(
            "_read_states_solids_thermal start at var_index {}".format(var_index))

        n_states = state_data.shape[0]
        n_solids = self.header["nel8"]
        n_thermal_vars = self.header["nt3d"]

        try:
            tmp_array = state_data[:, var_index:var_index + n_solids * n_thermal_vars]
            array_dict[arraytype.element_solid_thermal_data] = \
                tmp_array\
                .reshape((n_states, n_solids, n_thermal_vars))
        except Exception:
            trb_msg = traceback.format_exc()
            msg = "A failure in {0} was caught:\n{1}"
            logging.warning(
                msg.format("_read_states_solids_thermal",
                           trb_msg))
        finally:
            var_index += n_thermal_vars * n_solids

        logging.debug(
            "_read_states_solids_thermal end at var_index {}".format(var_index))

        return var_index

    def _read_states_solids(self,
                            state_data: np.ndarray,
                            var_index: int,
                            array_dict: dict) -> int:
        ''' Read the state data of the solid elements

        Parameters
        ----------
        state_data : np.ndarray
            array with entire state data
        var_index : int
            variable index in the state data array
        array_dict : `dict`
            dictionary to store the loaded arrays in

        Returns
        -------
        var_index : int
            updated variable index after reading the section
        '''

        if self.header["nel8"] <= 0 or self.header["nv3d"] <= 0:
            return var_index

        logging.debug(
            "_read_states_solids start at var_index {}".format(var_index))

        n_solid_vars = self.header["nv3d"]
        n_solids = self.header["nel8"]
        n_states = state_data.shape[0]
        n_strain_vars = 6 * self.header["istrn"]
        n_history_vars = self.header["neiph"]

        # double safety here, if either the formatting of the solid state data
        # or individual arrays fails then we catch it
        try:
            solid_state_data = \
                state_data[:, var_index:var_index + n_solid_vars * n_solids]\
                .reshape((n_states, n_solids, n_solid_vars))

            i_solid_var = 0

            # stress
            try:
                array_dict[arraytype.element_solid_stress] = \
                    solid_state_data[:, :, :6]\
                    .reshape((n_states, n_solids, 6))
            except Exception:
                trb_msg = traceback.format_exc()
                msg = "A failure in {0} was caught:\n{1}"
                logging.warning(
                    msg.format("_read_states_solids, stress",
                               trb_msg))
            finally:
                i_solid_var += 6

            # effective plastic strain
            try:
                array_dict[arraytype.element_solid_effective_plastic_strain] = \
                    solid_state_data[:, :, i_solid_var]\
                    .reshape((n_states, n_solids))
            except Exception:
                trb_msg = traceback.format_exc()
                msg = "A failure in {0} was caught:\n{1}"
                logging.warning(
                    msg.format("_read_states_solids, eff_plastic_strain",
                               trb_msg))
            finally:
                i_solid_var += 1

            # history vars
            if n_history_vars:
                try:
                    array_dict[arraytype.element_solid_history_variables] = \
                        solid_state_data[:, :, i_solid_var:i_solid_var + n_history_vars]\
                        .reshape((n_states, n_solids, n_history_vars))
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in {0} was caught:\n{1}"
                    logging.warning(
                        msg.format("_read_states_solids, history_variables",
                                   trb_msg))
                finally:
                    i_solid_var += n_history_vars

            # strain
            # they are the last 6 entries of the history vars
            if n_strain_vars:
                try:
                    array_dict[arraytype.element_solid_strain] = \
                        array_dict[arraytype.element_solid_history_variables][:, :, -n_strain_vars:]

                    array_dict[arraytype.element_solid_history_variables] = \
                        array_dict[arraytype.element_solid_history_variables][:, :, :-n_strain_vars]
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in {0} was caught:\n{1}"
                    logging.warning(
                        msg.format("_read_states_solids, strain",
                                   trb_msg))

        # catch formatting in solid_state_datra
        except Exception:
            trb_msg = traceback.format_exc()
            msg = "A failure in {0} was caught:\n{1}"
            logging.warning(
                msg.format("_read_states_solids, solid_state_data",
                           trb_msg))
        # always increment variable count
        finally:
            var_index += n_solids * n_solid_vars

        logging.debug(
            "_read_states_solids end at var_index {}".format(var_index))

        return var_index

    def _read_states_tshell(self,
                            state_data: np.ndarray,
                            var_index: int,
                            array_dict: dict) -> int:
        ''' Read the state data for thick shell elements

        Parameters
        ----------
        state_data : np.ndarray
            array with entire state data
        var_index : int
            variable index in the state data array
        array_dict : `dict`
            dictionary to store the loaded arrays in

        Returns
        -------
        var_index : int
            updated variable index after reading the section
        '''

        if self.header["nelth"] <= 0 or self.header["nv3dt"] <= 0:
            return var_index

        logging.debug(
            "_read_states_tshell start at var_index {}".format(var_index))

        n_states = state_data.shape[0]
        n_tshells = self.header["nelth"]
        n_history_vars = self.header["neips"]
        n_layers = self.header["maxint"]
        n_layer_vars = n_layers \
            * (6 * self.header["ioshl1"] +
               self.header["ioshl2"] +
               n_history_vars)
        n_strain_vars = 12 * self.header["istrn"]
        n_thsell_vars = self.header["nv3dt"]
        has_stress = self.header["ioshl1"]
        has_pstrain = self.header["ioshl2"]

        try:
            # thick shell element data
            tshell_data = state_data[:,
                                     var_index:var_index + n_thsell_vars * n_tshells]
            tshell_data = tshell_data.reshape((n_states, n_tshells, n_thsell_vars))

            # extract layer data
            tshell_layer_data = tshell_data[:, :, slice(0, n_layer_vars)]
            tshell_layer_data = tshell_layer_data\
                .reshape((n_states, n_tshells, n_layers, -1))
            tshell_nonlayer_data = tshell_data[:, :, n_layer_vars:]

            # STRESS
            i_tshell_layer_var = 0
            if has_stress:
                try:
                    array_dict[arraytype.element_tshell_stress] = \
                        tshell_layer_data[:, :, :, i_tshell_layer_var:i_tshell_layer_var + 6]\
                        .reshape((n_states, n_tshells, n_layers, 6))
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in {0} was caught:\n{1}"
                    logging.warning(
                        msg.format("_read_states_tshell, stress", trb_msg))
                finally:
                    i_tshell_layer_var += 6

            # PSTRAIN
            if has_pstrain:
                try:
                    array_dict[arraytype.element_tshell_effective_plastic_strain] = \
                        tshell_layer_data[:, :, :, i_tshell_layer_var]\
                        .reshape((n_states, n_tshells, n_layers))
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in {0} was caught:\n{1}"
                    logging.warning(
                        msg.format("_read_states_tshell, eff_plastic_strain", trb_msg))
                finally:
                    i_tshell_layer_var += 1

            # HISTORY VARS
            if n_history_vars:
                try:
                    array_dict[arraytype.element_tshell_history_variables] = \
                        tshell_layer_data[:, :, :,
                                          i_tshell_layer_var:i_tshell_layer_var + n_history_vars]\
                        .reshape((n_states, n_tshells, n_layers, n_history_vars))
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in {0} was caught:\n{1}"
                    logging.warning(
                        msg.format("_read_states_tshell, history_variables", trb_msg))

            # STRAIN (only non layer data for tshells)
            if n_strain_vars:
                try:
                    tshell_nonlayer_data = tshell_nonlayer_data[:, :, :n_strain_vars]
                    array_dict[arraytype.element_tshell_strain] = tshell_nonlayer_data\
                        .reshape((n_states, n_tshells, 2, 6))
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in {0} was caught:\n{1}"
                    logging.warning(
                        msg.format("_read_states_tshell, strain", trb_msg))

        except Exception:
            trb_msg = traceback.format_exc()
            msg = "A failure in {0} was caught:\n{1}"
            logging.warning(
                msg.format("_read_states_tshell, tshell_data", trb_msg))
        finally:
            var_index += n_thsell_vars * n_tshells

        logging.debug(
            "_read_states_tshell end at var_index {}".format(var_index))

        return var_index

    def _read_states_beams(self,
                           state_data: np.ndarray,
                           var_index: int,
                           array_dict: dict) -> int:
        ''' Read the state data for beams

        Parameters
        ----------
        state_data : np.ndarray
            array with entire state data
        var_index : int
            variable index in the state data array
        array_dict : `dict`
            dictionary to store the loaded arrays in

        Returns
        -------
        var_index : int
            updated variable index after reading the section
        '''

        if self.header["nel2"] <= 0 or self.header["nv1d"] <= 0:
            return var_index

        logging.debug(
            "_read_states_beams start at var_index {}".format(var_index))

        # beam intergration point vars
        N_BEAM_IP_VARS = 5

        n_states = state_data.shape[0]
        n_beams = self.header["nel2"]
        n_history_vars = self.header["neipb"]
        n_beam_vars = self.header["nv1d"]
        n_layers = int((-3 * n_history_vars + n_beam_vars - 6) /
                       (n_history_vars + 5))
        n_layer_vars = 6 + N_BEAM_IP_VARS * n_layers

        try:
            # beam element data
            beam_data = state_data[:, var_index:var_index + n_beam_vars * n_beams]
            beam_data = beam_data.reshape((n_states, n_beams, n_beam_vars))

            # extract layer data
            beam_nonlayer_data = beam_data[:, :, :6]
            beam_layer_data = beam_data[:, :, 6:6 + n_layer_vars]
            beam_layer_data = beam_layer_data\
                .reshape((n_states, n_beams, n_layers, N_BEAM_IP_VARS))

            # axial force
            try:
                array_dict[arraytype.element_beam_axial_force] = \
                    beam_nonlayer_data[:, :, 0]\
                    .reshape((n_states, n_beams))
            except Exception:
                trb_msg = traceback.format_exc()
                msg = "A failure in {0} was caught:\n{1}"
                logging.warning(
                    msg.format("_read_states_beams, axial_force", trb_msg))

            # shear force
            try:
                array_dict[arraytype.element_beam_shear_force] = \
                    beam_nonlayer_data[:, :, 1:3]\
                    .reshape((n_states, n_beams, 2))
            except Exception:
                trb_msg = traceback.format_exc()
                msg = "A failure in {0} was caught:\n{1}"
                logging.warning(
                    msg.format("_read_states_beams, shear_force", trb_msg))

            # bending moment
            try:
                array_dict[arraytype.element_beam_bending_moment] = \
                    beam_nonlayer_data[:, :, 3:5]\
                    .reshape((n_states, n_beams, 2))
            except Exception:
                trb_msg = traceback.format_exc()
                msg = "A failure in {0} was caught:\n{1}"
                logging.warning(
                    msg.format("_read_states_beams, bending_moment", trb_msg))

            # torsion moment
            try:
                array_dict[arraytype.element_beam_torsion_moment] = \
                    beam_nonlayer_data[:, :, 5]\
                    .reshape((n_states, n_beams))
            except Exception:
                trb_msg = traceback.format_exc()
                msg = "A failure in {0} was caught:\n{1}"
                logging.warning(
                    msg.format("_read_states_beams, torsion_moment", trb_msg))

            if n_layers:

                # shear stress
                try:
                    array_dict[arraytype.element_beam_shear_stress] = \
                        beam_layer_data[:, :, :, 0:2]
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in {0} was caught:\n{1}"
                    logging.warning(
                        msg.format("_read_states_beams, shear_stress", trb_msg))

                # axial stress
                try:
                    array_dict[arraytype.element_beam_axial_stress] = \
                        beam_layer_data[:, :, :, 2]
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in {0} was caught:\n{1}"
                    logging.warning(
                        msg.format("_read_states_beams, axial_stress", trb_msg))

                # eff. plastic strain
                try:
                    array_dict[arraytype.element_beam_plastic_strain] = \
                        beam_layer_data[:, :, :, 3]
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in {0} was caught:\n{1}"
                    logging.warning(
                        msg.format("_read_states_beams, eff_plastic_strain", trb_msg))

                # axial strain
                try:
                    array_dict[arraytype.element_beam_axial_strain] = \
                        beam_layer_data[:, :, :, 4]
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in {0} was caught:\n{1}"
                    logging.warning(
                        msg.format("_read_states_beams, axial_strain", trb_msg))

            # history vars
            if n_history_vars:
                try:
                    array_dict[arraytype.element_beam_history_vars] = \
                        beam_data[:, :, 6 + n_layer_vars:]\
                        .reshape((n_states, n_beams, 3 + n_layers, n_history_vars))
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in {0} was caught:\n{1}"
                    logging.warning(
                        msg.format("_read_states_beams, history_variables", trb_msg))

        # failure of formatting beam state data
        except Exception:
            trb_msg = traceback.format_exc()
            msg = "A failure in {0} was caught:\n{1}"
            logging.warning(
                msg.format("_read_states_beams, beam_state_data", trb_msg))
        # always increment variable index
        finally:
            var_index += n_beams * n_beam_vars

        logging.debug(
            "_read_states_beams end at var_index {}".format(var_index))

        return var_index

    def _read_states_shell(self,
                           state_data: np.ndarray,
                           var_index: int,
                           array_dict: dict) -> int:
        ''' Read the state data for shell elements

        Parameters
        ----------
        state_data : np.ndarray
            array with entire state data
        var_index : int
            variable index in the state data array
        array_dict : `dict`
            dictionary to store the loaded arrays in

        Returns
        -------
        var_index : int
            updated variable index after reading the section
        '''

        # bugfix
        #
        # Interestingly, dyna seems to write result values for rigid shells in
        # the d3part file, but not in the d3plot. Of course this is not
        # documented ...
        n_reduced_shells = self.header["nel4"] \
            if self.header["filetype"] == 5 \
            else self.header["nel4"] - self.header["numrbe"]

        if self.header["nv2d"] <= 0 \
           or n_reduced_shells <= 0:
            return var_index

        logging.debug(
            "_read_states_shell start at var_index {}".format(var_index))

        n_states = state_data.shape[0]
        n_shells = n_reduced_shells
        n_shell_vars = self.header["nv2d"]

        # what is in the file?
        n_layers = self.header["maxint"]
        n_history_vars = self.header["neips"]
        n_stress_vars = 6 * self.header["ioshl1"]
        n_pstrain_vars = self.header["ioshl2"]
        n_force_variables = 8 * self.header["ioshl3"]
        n_extra_variables = 4 * self.header["ioshl4"]
        n_strain_vars = 12 * self.header["istrn"]

        try:
            # this is a sanity check if the manual was understood correctly
            n_shell_vars2 = \
                n_layers * (n_stress_vars + n_pstrain_vars + n_history_vars)\
                + n_force_variables + n_extra_variables + n_strain_vars

            if n_shell_vars != n_shell_vars2:
                msg = "n_shell_vars != n_shell_vars_computed: {} != {}."\
                    + " Shell variables might be wrong."
                logging.warning(msg.format(n_shell_vars, n_shell_vars2))

            n_layer_vars = \
                n_layers * (n_stress_vars + n_pstrain_vars + n_history_vars)

            # shell element data
            shell_data = state_data[:, var_index:var_index + n_shell_vars * n_shells]
            shell_data = shell_data.reshape((n_states, n_shells, n_shell_vars))

            # extract layer data
            shell_layer_data = shell_data[:, :, :n_layer_vars]
            shell_layer_data = \
                shell_layer_data.reshape((n_states, n_shells, n_layers, -1))
            shell_nonlayer_data = shell_data[:, :, n_layer_vars:]

            # save layer stuff
            # STRESS
            layer_var_index = 0
            if n_stress_vars:
                try:
                    array_dict[arraytype.element_shell_stress] = \
                        shell_layer_data[:, :, :, :n_stress_vars]\
                        .reshape((n_states, n_shells, n_layers, n_stress_vars))
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in {0} was caught:\n{1}"
                    logging.warning(
                        msg.format("_read_states_shells, stress",
                                   trb_msg))
                finally:
                    layer_var_index += n_stress_vars

            # PSTRAIN
            if n_pstrain_vars:
                try:
                    array_dict[arraytype.element_shell_effective_plastic_strain] = \
                        shell_layer_data[:, :, :, layer_var_index]\
                        .reshape((n_states, n_shells, n_layers))
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in {0} was caught:\n{1}"
                    logging.warning(
                        msg.format("_read_states_shells, stress",
                                   trb_msg))
                finally:
                    layer_var_index += 1

            # HISTORY VARIABLES
            if n_history_vars:
                try:
                    array_dict[arraytype.element_shell_history_vars] = \
                        shell_layer_data[:, :, :, layer_var_index:layer_var_index + n_history_vars]\
                        .reshape((n_states, n_shells, n_layers, n_history_vars))
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in {0} was caught:\n{1}"
                    logging.warning(
                        msg.format("_read_states_shells, history_variables",
                                   trb_msg))
                finally:
                    layer_var_index += n_history_vars

            # save nonlayer stuff
            # forces
            nonlayer_var_index = 0
            if n_force_variables:
                try:
                    array_dict[arraytype.element_shell_bending_moment] = \
                        shell_nonlayer_data[:, :, 0:3]\
                        .reshape((n_states, n_shells, 3))
                    array_dict[arraytype.element_shell_shear_force] = \
                        shell_nonlayer_data[:, :, 3:5]\
                        .reshape((n_states, n_shells, 2))
                    array_dict[arraytype.element_shell_normal_force] = \
                        shell_nonlayer_data[:, :, 5:8]\
                        .reshape((n_states, n_shells, 3))
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in {0} was caught:\n{1}"
                    logging.warning(
                        msg.format("_read_states_shells, forces",
                                   trb_msg))
                finally:
                    nonlayer_var_index += n_force_variables

            # weird stuff
            if n_extra_variables:
                try:
                    array_dict[arraytype.element_shell_thickness] = \
                        shell_nonlayer_data[:, :, nonlayer_var_index]\
                        .reshape((n_states, n_shells))
                    array_dict[arraytype.element_shell_unknown_variables] = \
                        shell_nonlayer_data[:, :, nonlayer_var_index + 1:nonlayer_var_index + 3]\
                        .reshape((n_states, n_shells, 2))
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in {0} was caught:\n{1}"
                    logging.warning(
                        msg.format("_read_states_shells, history_variables",
                                   trb_msg))
                finally:
                    nonlayer_var_index += 3

            # INTERNAL_ENERGY
            # STRAIN
            if self.header["istrn"] == 0:
                try:
                    array_dict[arraytype.element_shell_internal_energy] = \
                        shell_nonlayer_data[:, :, nonlayer_var_index]\
                        .reshape((n_states, n_shells))
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in {0} was caught:\n{1}"
                    logging.warning(
                        msg.format("_read_states_shells, strain",
                                   trb_msg))
                finally:
                    nonlayer_var_index += 1

            # strain present
            if n_strain_vars:
                try:
                    shell_strain = \
                        shell_nonlayer_data[:, :,
                                            nonlayer_var_index:nonlayer_var_index + n_strain_vars]
                    array_dict[arraytype.element_shell_strain] = \
                        shell_strain.reshape((n_states, n_shells, 2, 6))\
                        .reshape((n_states, n_shells, 2, 6))
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in {0} was caught:\n{1}"
                    logging.warning(
                        msg.format("_read_states_shells, strains",
                                   trb_msg))
                finally:
                    nonlayer_var_index += n_strain_vars

                # internal energy is behind strain if strain is written
                # ... says the manual ...
                if n_shell_vars >= 45:
                    try:
                        array_dict[arraytype.element_shell_internal_energy] = \
                            shell_nonlayer_data[:, :, nonlayer_var_index]\
                            .reshape((n_states, n_shells))
                    except Exception:
                        trb_msg = traceback.format_exc()
                        msg = "A failure in {0} was caught:\n{1}"
                        logging.warning(
                            msg.format("_read_states_shells, strains",
                                       trb_msg))

        # error in formatting shell state data
        except Exception:
            trb_msg = traceback.format_exc()
            msg = "A failure in {0} was caught:\n{1}"
            logging.warning(
                msg.format("_read_states_shell, shell_state_data", trb_msg))

        # always increment variable index
        finally:
            var_index += n_shell_vars * n_shells

        logging.debug(
            "_read_states_shell end at var_index {}".format(var_index))

        return var_index

    def _read_states_is_alive(self,
                              state_data: np.ndarray,
                              var_index: int,
                              array_dict: dict) -> int:
        ''' Read deletion info for nodes, elements, etc

        Parameters
        ----------
        state_data : np.ndarray
            array with entire state data
        var_index : int
            variable index in the state data array
        array_dict : `dict`
            dictionary to store the loaded arrays in

        Returns
        -------
        var_index : int
            updated variable index after reading the section
        '''

        if self.header["mdlopt"] <= 0:
            return var_index

        logging.debug(
            "_read_states_is_alive start at var_index {}".format(var_index))

        n_states = state_data.shape[0]

        # NODES
        if self.header["mdlopt"] == 1:
            n_nodes = self.header["numnp"]

            if n_nodes > 0:
                try:
                    array_dict[arraytype.node_is_alive] = \
                        state_data[:, var_index:var_index + n_nodes]
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in {0} was caught:\n{1}"
                    logging.warning(msg.format("_read_states_is_alive, nodes",
                                               trb_msg))
                finally:
                    var_index += n_nodes

        # element deletion info
        elif self.header["mdlopt"] == 2:
            n_solids = self.header["nel8"]
            n_tshells = self.header["nelth"]
            n_shells = self.header["nel4"]
            n_beams = self.header["nel2"]
            # n_elems = n_solids + n_tshells + n_shells + n_beams

            # SOLIDS
            if n_solids > 0:
                try:
                    array_dict[arraytype.element_solid_is_alive] = \
                        state_data[:, var_index:var_index + n_solids]\
                        .reshape((n_states, n_solids))
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in {0} was caught:\n{1}"
                    logging.warning(
                        msg.format("_read_states_is_alive, solids",
                                   trb_msg))
                finally:
                    var_index += n_solids

            # TSHELLS
            if n_tshells > 0:
                try:
                    array_dict[arraytype.element_tshell_is_alive] = \
                        state_data[:, var_index:var_index + n_tshells]\
                        .reshape((n_states, n_tshells))
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in {0} was caught:\n{1}"
                    logging.warning(
                        msg.format("_read_states_is_alive, solids",
                                   trb_msg))
                finally:
                    var_index += n_tshells

            # SHELLS
            if n_shells > 0:
                try:
                    array_dict[arraytype.element_shell_is_alive] = \
                        state_data[:, var_index:var_index + n_shells]\
                        .reshape((n_states, n_shells))
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in {0} was caught:\n{1}"
                    logging.warning(
                        msg.format("_read_states_is_alive, shells",
                                   trb_msg))
                finally:
                    var_index += n_shells

            # BEAMS
            if n_beams > 0:
                try:
                    array_dict[arraytype.element_beam_is_alive] = \
                        state_data[:, var_index:var_index + n_beams]\
                        .reshape((n_states, n_beams))
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in {0} was caught:\n{1}"
                    logging.warning(
                        msg.format("_read_states_is_alive, beams",
                                   trb_msg))
                finally:
                    var_index += n_beams

        logging.debug(
            "_read_states_is_alive end at var_index {}".format(var_index))

        return var_index

    def _read_states_sph(self,
                         state_data: np.ndarray,
                         var_index: int,
                         array_dict: dict) -> int:
        ''' Read the sph state data

        Parameters
        ----------
        state_data : np.ndarray
            array with entire state data
        var_index : int
            variable index in the state data array
        array_dict : `dict`
            dictionary to store the loaded arrays in

        Returns
        -------
        var_index : int
            updated variable index after reading the section
        '''

        if self.header["nmsph"] <= 0:
            return var_index

        logging.debug(
            "_read_states_sph start at var_index {}".format(var_index))

        n_states = state_data.shape[0]
        n_particles = self.header["nmsph"]
        n_variables = self.header["num_sph_vars"]

        # extract data
        try:
            sph_data = state_data[:, var_index:var_index + n_particles * n_variables]

            i_var = 1

            # deletion
            try:
                array_dict[arraytype.sph_deletion] = sph_data[:, 0] < 0
            except Exception:
                trb_msg = traceback.format_exc()
                msg = "A failure in {0} was caught:\n{1}"
                logging.warning(
                    msg.format("_read_states_sph, deletion",
                               trb_msg))

            # particle radius
            if self.header["isphfg2"]:
                try:
                    array_dict[arraytype.sph_radius] = sph_data[:, i_var]
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in {0} was caught:\n{1}"
                    logging.warning(
                        msg.format("_read_states_sph, radius",
                                   trb_msg))
                finally:
                    i_var += 1

            # pressure
            if self.header["isphfg3"]:
                try:
                    array_dict[arraytype.sph_pressure] = sph_data[:, i_var]
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in {0} was caught:\n{1}"
                    logging.warning(
                        msg.format("_read_states_sph, pressure",
                                   trb_msg))
                finally:
                    i_var += 1

            # stress
            if self.header["isphfg4"]:
                try:
                    array_dict[arraytype.sph_stress] = sph_data[:, i_var:i_var + 6]\
                        .reshape((n_states, n_particles, 6))
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in {0} was caught:\n{1}"
                    logging.warning(
                        msg.format("_read_states_sph, pressure",
                                   trb_msg))
                finally:
                    i_var += 6

            # eff. plastic strain
            if self.header["isphfg5"]:
                try:
                    array_dict[arraytype.sph_effective_plastic_strain] = \
                        sph_data[:, i_var]
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in {0} was caught:\n{1}"
                    logging.warning(
                        msg.format("_read_states_sph, eff_plastic_strain",
                                   trb_msg))
                finally:
                    i_var += 1

            # density
            if self.header["isphfg6"]:
                try:
                    array_dict[arraytype.sph_density] = sph_data[:, i_var]
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in {0} was caught:\n{1}"
                    logging.warning(
                        msg.format("_read_states_sph, density",
                                   trb_msg))
                finally:
                    i_var += 1

            # internal energy
            if self.header["isphfg7"]:
                try:
                    array_dict[arraytype.sph_internal_energy] = sph_data[:, i_var]
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in {0} was caught:\n{1}"
                    logging.warning(
                        msg.format("_read_states_sph, internal_energy",
                                   trb_msg))
                finally:
                    i_var += 1

            # number of neighbors
            if self.header["isphfg8"]:
                try:
                    array_dict[arraytype.sph_n_neighbors] = sph_data[:, i_var]
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in {0} was caught:\n{1}"
                    logging.warning(
                        msg.format("_read_states_sph, n_neighbors",
                                   trb_msg))
                finally:
                    i_var += 1

            # strain
            if self.header["isphfg9"]:
                try:
                    array_dict[arraytype.sph_strain] = sph_data[:, i_var:i_var + 6]\
                        .reshape((n_states, n_particles, 6))
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in {0} was caught:\n{1}"
                    logging.warning(
                        msg.format("_read_states_sph, strain",
                                   trb_msg))
                finally:
                    i_var += 6

            # mass
            if self.header["isphfg10"]:
                try:
                    array_dict[arraytype.sph_mass] = sph_data[:, i_var]
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in {0} was caught:\n{1}"
                    logging.warning(
                        msg.format("_read_states_sph, pressure",
                                   trb_msg))
                finally:
                    i_var += 1

        except Exception:
            trb_msg = traceback.format_exc()
            msg = "A failure in {0} was caught:\n{1}"
            logging.warning(
                msg.format("_read_states_sph, sph_data",
                           trb_msg))
        finally:
            var_index += n_particles * n_variables

        logging.debug("_read_states_sph end at var_index {}".format(var_index))

        return var_index

    def _read_states_airbags(self, state_data: np.ndarray, var_index: int, array_dict: dict) -> int:
        ''' Read the airbag state data

        Parameters
        ----------
        state_data : np.ndarray
            array with entire state data
        var_index : int
            variable index in the state data array
        array_dict : `dict`
            dictionary to store the loaded arrays in

        Returns
        -------
        var_index : int
            updated variable index after reading the section
        '''

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

        # Warning
        # Im not sure if this is right ...
        n_total_vars = \
            n_airbags * n_state_airbag_vars\
            + n_particles * n_particle_vars

        try:
            # types
            # nlist = ngeom + nvar + nstgeom
            airbag_var_types = self.arrays[arraytype.airbag_variable_types]
            # geom_var_types = airbag_var_types[:n_airbag_geom_vars]
            particle_var_types = airbag_var_types[
                n_airbag_geom_vars:n_airbag_geom_vars + n_particle_vars]
            airbag_state_var_types = \
                airbag_var_types[n_airbag_geom_vars + n_particle_vars:]

            # required for dynamic reading
            def get_dtype(type_flag):
                return self.itype if type_flag == 1 else self.ftype

            # extract airbag data
            airbag_state_data = state_data[:, var_index:var_index + n_total_vars]

            # airbag data
            airbag_data = airbag_state_data[:, :n_airbags * n_state_airbag_vars]\
                .reshape((n_states, n_airbags, n_state_airbag_vars))
            airbag_state_offset = n_airbags * n_state_airbag_vars

            # particle data
            particle_data = \
                airbag_state_data[:,
                                  airbag_state_offset:
                                  airbag_state_offset + n_particles * n_particle_vars]\
                .reshape((n_states, n_particles, n_particle_vars))

            # save sh...

            # airbag active particles
            if n_state_airbag_vars >= 1:
                try:
                    array_dict[arraytype.airbag_n_active_particles] = \
                        airbag_data[:, :, 0]\
                        .view(get_dtype(airbag_state_var_types[0]))
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in {0} was caught:\n{1}"
                    logging.warning(
                        msg.format("_read_states_airbags, airbag_n_active_particles",
                                   trb_msg))

            # airbag bag volumne
            if n_state_airbag_vars >= 2:
                try:
                    array_dict[arraytype.airbag_bag_volume] = \
                        airbag_data[:, :, 1]\
                        .view(get_dtype(airbag_state_var_types[1]))
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in {0} was caught:\n{1}"
                    logging.warning(
                        msg.format("_read_states_airbags, airbag_volume",
                                   trb_msg))

            # particle gas id
            if n_particle_vars >= 1:
                try:
                    array_dict[arraytype.airbag_particle_gas_id] = \
                        particle_data[:, :, 0]\
                        .view(get_dtype(particle_var_types[0]))
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in {0} was caught:\n{1}"
                    logging.warning(
                        msg.format("_read_states_airbags, particle_gas_id",
                                   trb_msg))

            # particle chamber id
            if n_particle_vars >= 2:
                try:
                    array_dict[arraytype.airbag_particle_chamber_id] = \
                        particle_data[:, :, 1]\
                        .view(get_dtype(particle_var_types[1]))
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in {0} was caught:\n{1}"
                    logging.warning(
                        msg.format("_read_states_airbags, particle_chamber_id",
                                   trb_msg))

            # particle leakage
            if n_particle_vars >= 3:
                try:
                    array_dict[arraytype.airbag_particle_leakage] = \
                        particle_data[:, :, 2]\
                        .view(get_dtype(particle_var_types[2]))
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in {0} was caught:\n{1}"
                    logging.warning(
                        msg.format("_read_states_airbags, particle_leakage",
                                   trb_msg))

            # particle mass
            if n_particle_vars >= 4:
                try:
                    array_dict[arraytype.airbag_particle_mass] = \
                        particle_data[:, :, 3]\
                        .view(get_dtype(particle_var_types[3]))
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in {0} was caught:\n{1}"
                    logging.warning(
                        msg.format("_read_states_airbags, particle_mass",
                                   trb_msg))

            # particle radius
            if n_particle_vars >= 5:
                try:
                    array_dict[arraytype.airbag_particle_radius] = \
                        particle_data[:, :, 4]\
                        .view(get_dtype(particle_var_types[4]))
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in {0} was caught:\n{1}"
                    logging.warning(
                        msg.format("_read_states_airbags, particle_radius",
                                   trb_msg))

            # particle spin energy
            if n_particle_vars >= 6:
                try:
                    array_dict[arraytype.airbag_particle_spin_energy] = \
                        particle_data[:, :, 5]\
                        .view(get_dtype(particle_var_types[5]))
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in {0} was caught:\n{1}"
                    logging.warning(
                        msg.format("_read_states_airbags, particle_spin_energy",
                                   trb_msg))

            # particle translational energy
            if n_particle_vars >= 7:
                try:
                    array_dict[arraytype.airbag_particle_translation_energy] = \
                        particle_data[:, :, 6]\
                        .view(get_dtype(particle_var_types[6]))
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in {0} was caught:\n{1}"
                    logging.warning(
                        msg.format("_read_states_airbags, particle_transl_energy",
                                   trb_msg))

            # particle segment distance
            if n_particle_vars >= 8:
                try:
                    array_dict[arraytype.airbag_particle_nearest_segment_distance] = \
                        particle_data[:, :, 7]\
                        .view(get_dtype(particle_var_types[7]))
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in {0} was caught:\n{1}"
                    logging.warning(
                        msg.format("_read_states_airbags, particle_segment_distance",
                                   trb_msg))

            # particle position
            if n_particle_vars >= 11:
                try:
                    array_dict[arraytype.airbag_particle_position] = \
                        particle_data[:, :, 8:11]\
                        .view(get_dtype(particle_var_types[8]))
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in {0} was caught:\n{1}"
                    logging.warning(
                        msg.format("_read_states_airbags, particle_position",
                                   trb_msg))

            # particle velocity
            if n_particle_vars >= 14:
                try:
                    array_dict[arraytype.airbag_particle_velocity] = \
                        particle_data[:, :, 11:14]\
                        .view(get_dtype(particle_var_types[11]))
                except Exception:
                    trb_msg = traceback.format_exc()
                    msg = "A failure in {0} was caught:\n{1}"
                    logging.warning(
                        msg.format("_read_states_airbags, particle_velocity",
                                   trb_msg))

        except Exception:
            trb_msg = traceback.format_exc()
            msg = "A failure in {0} was caught:\n{1}"
            logging.warning(
                msg.format("_read_states_airbags, particle_data",
                           trb_msg))
        finally:
            var_index += n_total_vars

        logging.debug(
            "_read_states_airbags end at var_index {}".format(var_index))

        return var_index

    def _read_states_road_surfaces(self, state_data: np.ndarray, var_index: int, array_dict: dict) -> int:
        ''' Read the road surfaces state data for whoever wants this ...

        Parameters
        ----------
        state_data : np.ndarray
            array with entire state data
        var_index : int
            variable index in the state data array
        array_dict : `dict`
            dictionary to store the loaded arrays in

        Returns
        -------
        var_index : int
            updated variable index after reading the section
        '''

        if not self.header["has_rigid_road_surface"]:
            return var_index

        logging.debug(
            "_read_states_road_surfaces start at var_index {}".format(var_index))

        n_states = state_data.shape[0]
        n_roads = self.header["rigid_road"]["nsurf"]

        try:
            # read road data
            road_data = state_data[:, var_index:var_index + 6 * n_roads]\
                .reshape((n_states, n_roads, 2, 3))

            # DISPLACEMENT
            try:
                array_dict[arraytype.rigid_road_displacement] = \
                    road_data[:, :, 0, :]
            except Exception:
                trb_msg = traceback.format_exc()
                msg = "A failure in {0} was caught:\n{1}"
                logging.warning(
                    msg.format("_read_states_road_surfaces, road_displacement",
                               trb_msg))

            # VELOCITY
            try:
                array_dict[arraytype.rigid_road_velocity] = \
                    road_data[:, :, 1, :]
            except Exception:
                trb_msg = traceback.format_exc()
                msg = "A failure in {0} was caught:\n{1}"
                logging.warning(
                    msg.format("_read_states_road_surfaces, road_velocity",
                               trb_msg))

        except Exception:
            trb_msg = traceback.format_exc()
            msg = "A failure in {0} was caught:\n{1}"
            logging.warning(
                msg.format("_read_states_road_surfaces, road_data",
                           trb_msg))
        finally:
            var_index += 6 * n_roads

        logging.debug(
            "_read_states_road_surfaces end at var_index {}".format(var_index))

        return var_index

    def _read_states_rigid_body_motion(self, state_data: np.ndarray, var_index: int, array_dict: dict) -> int:
        ''' Read the road surfaces state data for whoever want this ...

        Parameters
        ----------
        state_data : np.ndarray
            array with entire state data
        var_index : int
            variable index in the state data array
        array_dict : `dict`
            dictionary to store the loaded arrays in

        Returns
        -------
        var_index : int
            updated variable index after reading the section
        '''

        if not self.header["has_rigid_body_data"]:
            return var_index

        logging.debug(
            "_read_states_rigid_body_motion start at var_index {}".format(var_index))

        n_states = state_data.shape[0]
        n_rigids = self.header["nrigid"]
        n_rigid_vars = 12 if self.header["has_reduced_rigid_body_data"] else 24

        try:
            # do the thing
            rigid_body_data = state_data[:, var_index:var_index + n_rigids * n_rigid_vars]\
                .reshape((n_states, n_rigids, n_rigid_vars))

            # let the party begin
            # rigid coordinates
            try:
                array_dict[arraytype.rigid_body_coordinates] = \
                    rigid_body_data[:, :, :3]
            except Exception:
                trb_msg = traceback.format_exc()
                msg = "A failure in {0} was caught:\n{1}"
                logging.warning(
                    msg.format("_read_states_rigid_body_motion, coordinates",
                               trb_msg))
            finally:
                i_var = 3

            # rotation matrix
            try:
                array_dict[arraytype.rigid_body_rotation_matrix] = \
                    rigid_body_data[:, :, i_var:i_var + 9]
            except Exception:
                trb_msg = traceback.format_exc()
                msg = "A failure in {0} was caught:\n{1}"
                logging.warning(
                    msg.format("_read_states_rigid_body_motion, rot_matrix",
                               trb_msg))
            finally:
                i_var += 9

            if self.header["has_reduced_rigid_body_data"]:
                return var_index

            # velocity pewpew
            try:
                array_dict[arraytype.rigid_body_velocity] = \
                    rigid_body_data[:, :, i_var:i_var + 3]
            except Exception:
                trb_msg = traceback.format_exc()
                msg = "A failure in {0} was caught:\n{1}"
                logging.warning(
                    msg.format("_read_states_rigid_body_motion, velocity",
                               trb_msg))
            finally:
                i_var += 3

            # rotational velocity
            try:
                array_dict[arraytype.rigid_body_rot_velocity] = \
                    rigid_body_data[:, :, i_var:i_var + 3]
            except Exception:
                trb_msg = traceback.format_exc()
                msg = "A failure in {0} was caught:\n{1}"
                logging.warning(
                    msg.format("_read_states_rigid_body_motion, rot_velocity",
                               trb_msg))
            finally:
                i_var += 3

            # acceleration
            try:
                array_dict[arraytype.rigid_body_acceleration] = \
                    rigid_body_data[:, :, i_var:i_var + 3]
            except Exception:
                trb_msg = traceback.format_exc()
                msg = "A failure in {0} was caught:\n{1}"
                logging.warning(
                    msg.format("_read_states_rigid_body_motion, acceleration",
                               trb_msg))
            finally:
                i_var += 3

            # rotational acceleration
            try:
                array_dict[arraytype.rigid_body_rot_acceleration] = \
                    rigid_body_data[:, :, i_var:i_var + 3]
            except Exception:
                trb_msg = traceback.format_exc()
                msg = "A failure in {0} was caught:\n{1}"
                logging.warning(
                    msg.format("_read_states_rigid_body_motion, rot_acceleration",
                               trb_msg))
            finally:
                i_var += 3

        except Exception:
            trb_msg = traceback.format_exc()
            msg = "A failure in {0} was caught:\n{1}"
            logging.warning(
                msg.format("_read_states_rigid_body_motion, rigid_body_data",
                           trb_msg))

        finally:
            var_index += n_rigids * n_rigid_vars

        logging.debug(
            "_read_states_rigid_body_motion end at var_index {}".format(var_index))

        return var_index

    def _collect_file_infos(self, size_per_state: int):
        ''' This routine collects the memory and file info for the d3plot files

        Parameters
        ----------
        size_per_state : int
            size of every state to be read

        Returns
        -------
        memory_infos : `dict`
            containing infos: `start`, `length`, `offset`, `filepath`
            and `n_states`.

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
            return []

        base_filepath = self.bb.filepath_[0]

        # bugfix
        # If you encounter these int casts more often here this is why:
        # Some ints around here are numpy.int32 which can overflow
        # (sometimes there is a warning ... sometimes not ...)
        # we cast to python ints in order to prevent overflow.
        size_per_state = int(size_per_state)

        # Info:
        #
        # We need to determine here how many states are in every file
        # without really loading the file itself. For big files this is
        # simply filesize // state_size.
        # For files though with a smaller filesize this may cause issues
        # e.g.
        # filesize 2048 bytes (minimum filesize from dyna)
        # geom_size 200 bytes
        # state_size 200 bytes
        # File contains:
        # -> 1 state * state_size + geom_size = 400 bytes
        # Wrong State Estimation:
        # -> (filesize - geom_size) // state_size = 9 states
        #
        # To avoid this wrong number of states when reading small files
        # we need to search the end mark (here nonzero byte) from the rear
        # of the file.
        # This though needs the file to be loaded into memory. To make this
        # very light, we simply memorymap a small fraction of the file starting
        # from the rear until we have our nonzero byte. Since the end mark
        # is usually in the first block loaded, there should not be any performance
        # concerns, even with bigger files.

        # query for state files
        filepaths = D3plot._find_dyna_result_files(base_filepath)

        # compute state data in first file
        # search therefore the first non-zero byte from the rear
        last_nonzero_byte_index = self.bb.size
        mview_inv_arr = np.asarray(self.bb.memoryview[::-1])
        BLOCK_SIZE = 2048
        for start in range(0, self.bb.size, BLOCK_SIZE):
            nz_indexes, = np.nonzero(mview_inv_arr[start:start + BLOCK_SIZE])
            if len(nz_indexes):
                last_nonzero_byte_index = self.bb.size - (start + nz_indexes[0])
                break
        n_states_beyond_geom = (
            last_nonzero_byte_index - self.geometry_section_size) // size_per_state

        # memory required later
        memory_infos = [{
            "start": self.geometry_section_size,
            "length": n_states_beyond_geom * size_per_state,
            "offset": 0,
            "filepath": base_filepath,
            "n_states": n_states_beyond_geom
        }]

        # compute amount of state data in every further file
        for filepath in filepaths:
            filesize = os.path.getsize(filepath)
            last_nonzero_byte_index = -1

            import mmap
            n_blocks = filesize // mmap.ALLOCATIONGRANULARITY
            rest_size = filesize % mmap.ALLOCATIONGRANULARITY
            block_length = mmap.ALLOCATIONGRANULARITY
            with open(filepath, "rb") as fp:

                # search last rest block (page-aligned)
                # page-aligned means the start must be
                # a multiple of mmap.ALLOCATIONGRANULARITY
                # otherwise we get an error on linux
                if rest_size:
                    start = n_blocks * block_length
                    mview = memoryview(mmap.mmap(fp.fileno(),
                                                 offset=start,
                                                 length=rest_size,
                                                 access=mmap.ACCESS_READ).read())
                    nz_indexes, = np.nonzero(mview[::-1])
                    if len(nz_indexes):
                        last_nonzero_byte_index = start + rest_size - nz_indexes[0]

                # search in blocks from the rear
                if last_nonzero_byte_index == -1:
                    for i_block in range(n_blocks - 1, 0, -1):
                        start = block_length * i_block
                        mview = memoryview(mmap.mmap(fp.fileno(),
                                                     offset=start,
                                                     length=block_length,
                                                     access=mmap.ACCESS_READ).read())
                        nz_indexes, = np.nonzero(mview[::-1])
                        if len(nz_indexes):
                            index = block_length - nz_indexes[0]
                            last_nonzero_byte_index = start + index
                            break

            if last_nonzero_byte_index == -1:
                msg = "The file {0} seems to be missing it's endmark."
                raise RuntimeError(msg.format(filepath))

            n_states_in_file = last_nonzero_byte_index // size_per_state
            memory_infos.append({
                "start": 0,
                "length": size_per_state * (n_states_in_file),
                "offset": memory_infos[-1]["length"],
                "filepath": filepath,
                "n_states": n_states_in_file
            })

        return memory_infos

    @staticmethod
    def _read_file_from_memory_info(memory_infos):
        ''' Read files from a single or multiple memory infos

        Parameters
        ----------
        memory_infos : `dict` or `list` of `dict`
            memory infos for loading a file (see `D3plot._collect_file_infos`)

        Returns
        -------
        bb_states : BinaryBuffer
            New binary buffer with all states perfectly linear in memory
        n_states : int
            Number of states to be expected

        Notes
        -----
            This routine in contrast to `D3plot._read_state_bytebuffer` is used
            to load only a fraction of files into memory.
        '''

        # single file case
        if isinstance(memory_infos, dict):
            memory_infos = [memory_infos]

        # allocate memory
        # bugfix: casting to int prevents int32 overflow for large files
        memory_required = 0
        for mem in memory_infos:
            memory_required += int(mem["length"])
        mview = memoryview(bytearray(memory_required))

        # transfer memory for other files
        n_states = 0
        total_offset = 0
        for minfo in memory_infos:
            start = minfo["start"]
            length = minfo["length"]
            # offset = minfo["offset"]
            filepath = minfo["filepath"]

            logging.debug("opening: {0}".format(filepath))

            with open(filepath, "br") as fp:
                fp.seek(start)
                fp.readinto(mview[total_offset:total_offset + length])

            total_offset += length
            n_states += minfo["n_states"]

        # save
        bb_states = BinaryBuffer()
        bb_states.memoryview = mview

        return bb_states, n_states

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

        memory_infos = self._collect_file_infos(size_per_state)

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
                fp.seek(offset)
                fp.readinto(mview[total_offset:total_offset + length])

            total_offset += length
            n_states += minfo["n_states"]

        # save
        bb_states = BinaryBuffer()
        bb_states.memoryview = mview
        return bb_states, n_states

    def _read_words(self, words_to_read: dict, storage_dict: dict = None):
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

        if storage_dict is None:
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
                        data[0], data[1] * data[2])
                except UnicodeDecodeError as err:
                    storage_dict[name] = ""

            else:
                raise RuntimeError(
                    "Encountered unknown dtype {} during reading.".format(str(data[1])))

        return storage_dict

    @staticmethod
    def _find_dyna_result_files(filepath: str):
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

        pattern = "({path})[0-9]+$".format(path=file_basename)
        reg = re.compile(pattern)

        filepaths = [os.path.join(file_dir, path) for path
                     in os.listdir(file_dir)
                     if os.path.isfile(os.path.join(file_dir, path)) and
                     reg.match(path)]

        # alphasort files to handle d3plots with more than 100 files
        # e.g. d3plot01, d3plot02, ..., d3plot100
        def convert(text):
            return int(text) if text.isdigit() else text.lower()
        number_pattern = '([0-9]+)'

        def alphanum_key(key):
            return [convert(c) for c in re.split(number_pattern, key)]

        return sorted(filepaths, key=alphanum_key)

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

        raise RuntimeError("Unknown file type '{0}'.".format(value))

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
        assert(arraytype.node_displacement in self.arrays)

        # shell nodes
        shell_node_indexes = self.arrays[arraytype.element_shell_node_indexes]

        # get node displacement
        node_xyz = self.arrays[arraytype.node_displacement][i_timestep, :, :]

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

    def _get_zero_byte_padding(self,
                               n_bytes_written: int,
                               block_size_bytes: int):
        ''' Compute the zero byte-padding at the end of files

        Parameters
        ----------
        n_bytes_written : int
            number of bytes already written to file
        block_size_bytes : int
            byte block size of the file

        Returns
        -------
        zero_bytes : bytes
            zero-byte padding ready to be written to the file
        '''

        if block_size_bytes > 0:
            remaining_bytes = n_bytes_written % block_size_bytes
            n_bytes_to_fill = block_size_bytes - remaining_bytes if remaining_bytes != 0 else 0
            return b'\x00' * n_bytes_to_fill

    def compare(self, d3plot2, array_eps: Union[float, None] = None):
        ''' Compare two d3plots and print the info

        Parameters
        ----------
        d3plot2 : D3plot
            second d3plot
        array_eps : float or None
            tolerance for arrays, None by default

        Returns
        -------
        hdr_differences : dict
            differences in the header
        array_differences : dict
            difference between arrays (number of non-matching elements)
        '''
        assert(isinstance(d3plot2, D3plot))
        d3plot1 = self

        def _keys_not_set(key_list: List[str]) -> bool:
            for key in key_list:
                if d3plot1.header[key] != 0\
                   or d3plot2.header[key] != 0:
                    return False
            return True

        hdr_differences = {}

        for key, value in d3plot1.header.items():

            if key not in d3plot2.header:
                hdr_differences[key] = "missing"
                continue

            value2 = d3plot2.header[key]

            # d3parts write results for rigid shells.
            # when rewriting as d3plot we simply
            # don't write the part_material_types
            # array which is the same as having no
            # rigid shells.
            d3plot1_is_d3part = d3plot1.header["filetype"] == 5 \
                if "filetype" in d3plot1.header \
                else False
            d3plot2_is_d3part = d3plot2.header["filetype"] == 5 \
                if "filetype" in d3plot2.header \
                else False
            if key == "mattyp" and (
                    d3plot1_is_d3part or d3plot2_is_d3part):
                continue

            # it doesn't matter to check for shell
            # flags if there are no elements
            # output for shells at all
            if key.startswith("ioshl") \
               and _keys_not_set(["nel4", "nelth"]):
                continue

            # Variable Flags
            #
            # variable counts will not be compared if
            # no elements are present. The write process
            # usually corrects this to 0 again.
            if key == "nv3d" \
               and _keys_not_set(["nel8"]):
                continue

            if key == "nv2d" \
               and _keys_not_set(["nel4"]):
                continue

            if key == "nv1d" \
               and _keys_not_set(["nel2"]):
                continue

            if key == "nv3dt" \
               and _keys_not_set(["nelth"]):
                continue

            if key == "neipb" \
               and _keys_not_set(["nel2"]):
                continue

            if key == "neiph" \
               and _keys_not_set(["nel8"]):
                continue

            if key == "neips" \
               and _keys_not_set(["nel4", "nelth"]):
                continue

            # Material Comparison
            #
            # materials are assumed to be the same when their unique
            # material indexes match. Dyna sometimes writes more
            # materials than unique indexes, this is corrected during
            # the write process

            def _unique_array_equal(array_type: str) -> bool:
                array1 = d3plot1.arrays[array_type]
                array2 = d3plot2.arrays[array_type]
                array1 = np.unique(array1)
                array2 = np.unique(array2)
                if np.array_equal(array1, array2):
                    return True
                return False

            if key == "nummat2" \
               and _unique_array_equal(arraytype.element_beam_part_indexes):
                continue

            if key == "nummat4" \
               and _unique_array_equal(arraytype.element_shell_part_indexes):
                continue

            if key == "nummat8" \
               and _unique_array_equal(arraytype.element_solid_part_indexes):
                continue

            if key == "nummatt" \
               and _unique_array_equal(arraytype.element_tshell_part_indexes):
                continue

            # release version
            #
            # some release versions are not valid and are returned as
            # empty bytestring. Writing this again will be padded with
            # spaces by default, thus causing differences
            if key == "release_version" \
                    and not d3plot1.header[key].strip() \
                    and not d3plot2.header[key].strip():
                continue

            # numbering header
            #
            # the numbering header is a subdict in the header.
            # There is also a clause to escape differences in
            # numrbs if nmmat is the same. This is done due to
            # a hack which makes non-existing ghost materials
            # in the d3plot a rigid body when rewriting.
            if key == "numbering_header":
                for subkey in value.keys():
                    subvalue = value[subkey]
                    subvalue2 = value2[subkey]

                    if subkey == "numrbs" \
                       and d3plot1.header["nmmat"] \
                       == d3plot2.header["nmmat"]:
                        continue

                    if subvalue != subvalue2:
                        print(subkey, subvalue, subvalue2)
                continue

            # general comparison
            #
            # this is for the comparison of any left value
            if value != value2:
                hdr_differences[key] = (value, value2)

        # ARRAY COMPARISON
        array_differences = {}

        for name, array1 in d3plot1.arrays.items():
            array2 = d3plot2.arrays[name] \
                if name in d3plot2.arrays else "array is missing"

            # d3parts write results for rigid shells.
            # when rewriting as d3plot we simply
            # don't write the part_material_types
            # array which is the same as having no
            # rigid shells.
            d3plot1_is_d3part = d3plot1.header["filetype"] == 5 \
                if "filetype" in d3plot1.header \
                else False
            d3plot2_is_d3part = d3plot2.header["filetype"] == 5 \
                if "filetype" in d3plot1.header \
                else False
            if name == "part_material_type" and (
                    d3plot1_is_d3part or d3plot2_is_d3part):
                continue

            # we have an array to compare
            if isinstance(array2, np.ndarray):
                comparison = False

                # compare arrays
                if isinstance(array1, np.ndarray):
                    if array1.shape != array2.shape:
                        comparison = "shape mismatch {0} != {1}"\
                            .format(array1.shape, array2.shape)
                    else:
                        # comparison = (array1 != array2).sum()

                        if array_eps is not None and np.issubdtype(array1.dtype, np.number) \
                           and np.issubdtype(array2.dtype, np.number):
                            comparison = (np.abs(array1 - array2) > array_eps).sum()
                        else:
                            comparison = (array1 != array2).sum()

                else:
                    comparison = array1 != array2

                # print
                if comparison:
                    array_differences[name] = comparison

            # missing flag was set
            elif isinstance(array2, str):
                array_differences[name] = array2

        return hdr_differences, array_differences

    def get_part_filter(self,
                        filter_type: FilterType,
                        part_ids: Iterable[int],
                        for_state_array: bool = True) -> np.ndarray:
        """ Get a part filter for different entities

        Parameters
        ----------
        filter_type: `lasso.dyna.FilterType` or `str`
            the array type to filter (beam, shell, solid, tshell)
        part_ids: `Iterable[int]`
            part ids to filter out
        for_state_array: `bool`
            if the filter is meant for a state array. Makes a difference
            for shells if rigid bodies are in the model (mattyp == 20)

        Returns
        -------
        mask: `np.ndarray`
            mask usable on arrays to filter results

        Examples
        --------
            >>> from lasso.dyna import D3plot, ArrayType, FilterType
            >>> d3plot = D3plot("path/to/d3plot")
            >>> part_ids = [13, 14]
            >>> mask = d3plot.get_part_filter(FilterType.shell)
            >>> shell_stress = d3plot.arrays[ArrayType.element_shell_stress]
            >>> shell_stress.shape
            (34, 7463, 3, 6)
            >>> # select only parts from part_ids
            >>> shell_stress_parts = shell_stress[:, mask]
        """

        # we need part ids first
        if ArrayType.part_ids in self.arrays:
            d3plot_part_ids = self.arrays[ArrayType.part_ids]
        elif ArrayType.part_titles_ids in self.arrays:
            d3plot_part_ids = self.arrays[ArrayType.part_titles_ids]
        else:
            msg = "D3plot does neither contain '{0}' nor '{1}'"
            raise RuntimeError(msg.format(ArrayType.part_ids, ArrayType.part_titles_ids))

        # if we filter parts we can stop here
        if filter_type == FilterType.PART:
            return np.isin(d3plot_part_ids, part_ids)

        # get part indexes from part ids
        part_indexes = np.argwhere(np.isin(d3plot_part_ids, part_ids)).flatten()

        # associate part indexes with entities
        if filter_type == FilterType.BEAM:
            entity_part_indexes = self.arrays[ArrayType.element_beam_part_indexes]
        elif filter_type == FilterType.SHELL:
            entity_part_indexes = self.arrays[ArrayType.element_shell_part_indexes]

            # shells may contain "rigid body shell elements"
            # for these shells no state data is output and thus
            # the state arrays have a reduced element count
            if for_state_array and "numrbe" in self.header and self.header["numrbe"] != 0:
                mat_types = self.arrays[ArrayType.part_material_type]
                mat_type_filter = mat_types[entity_part_indexes] != 20
                entity_part_indexes = entity_part_indexes[mat_type_filter]

        elif filter_type == FilterType.TSHELL:
            entity_part_indexes = self.arrays[ArrayType.element_tshell_part_indexes]
        elif filter_type == FilterType.SOLID:
            entity_part_indexes = self.arrays[ArrayType.element_solid_part_indexes]
        else:
            msg = "Invalid filter_type '{0}'. Use lasso.dyna.FilterType."
            raise ValueError(msg.format(filter_type))

        mask = np.isin(entity_part_indexes, part_indexes)
        return mask

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
