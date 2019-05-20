
import os
import sys
import numpy as np


class BinaryBuffer:
    '''This class is used to handle binary data
    '''

    def __init__(self, filepath=None):
        '''Buffer used to read binary files

        Parameters
        ----------
        filepath : str
            path to a binary file

        Returns
        -------
        instance : BinaryBuffer
        '''
        self.filepath_ = None
        self.sizes_ = []
        self.load(filepath)

    @property
    def memoryview(self):
        '''Get the underlying memoryview of the binary buffer

        Returns
        -------
        mv_ : memoryview
            memoryview used to store the data
        '''
        return self.mv_

    @memoryview.setter
    def memoryview(self, new_mv):
        '''Set the memoryview of the binary buffer manually

        Parameters
        ----------
        new_mv : memoryview
            memoryview used to store the bytes
        '''
        assert(isinstance(new_mv, memoryview))
        self.mv_ = new_mv
        self.sizes_ = [len(self.mv_)]

    def get_slice(self, binary_buffer, start, end=None, step=1, copy=False):
        '''Get a slice of the binary buffer

        Parameters
        ----------
        binary_buffer : BinaryBuffer
            buffer to get a slice from
        start : int
            start position in bytes
        end : int
            end position
        copy : bool
            whether the data shall be copied or not

        Returns
        -------
        new_buffer : BinaryBuffer
            the slice as a new buffer
        '''

        assert(isinstance(binary_buffer, BinaryBuffer))
        assert(start < len(self))
        assert(end == None or end < len(self))

        end = len(self) if end == None else end

        new_binary_buffer = BinaryBuffer()
        new_binary_buffer.set_memoryview(self.mv_[start:end:step])

        return new_binary_buffer

    def __len__(self):
        '''Get the length of the byte buffer

        Returns
        -------
        len : int
        '''
        return len(self.mv_)

    @property
    def size(self):
        '''Get the size of the byte buffer

        Returns
        -------
        size : int
            size of buffer in bytes
        '''
        return len(self.mv_)

    @size.setter
    def size(self, size):
        '''Set the length of the byte buffer

        Parameters
        ----------
        size : int
            new size of the buffer
        '''

        if len(self.mv_) > size:
            self.mv_ = self.mv_[:size]
        elif len(self.mv_) < size:
            buffer = bytearray(self.mv_) + bytearray(b'0'*(size-len(self.mv_)))
            self.mv_ = memoryview(buffer)

    def read_number(self, start, dtype):
        '''Read a number from the buffer

        Parameters
        ----------
        start : int
            at which byte to start reading
        dtype : np.dtype
            type of the number to read

        Returns
        -------
        number : np.dtype
            number with the type specified
        '''
        return np.frombuffer(self.mv_,
                             dtype=dtype,
                             count=1,
                             offset=start)[0]

    def write_number(self, start, value, dtype):
        '''Write a number to the buffer

        Parameters
        ----------
        start : int
            at which byte to start writing
        value : np.dtype
            value to write
        dtype : np.dtype
            type of the number to write
        '''

        wrapper = np.frombuffer(self.mv_[start:], dtype=dtype)
        wrapper[0] = value

    def read_ndarray(self, start, length, step, dtype):
        '''Read a numpy array from the buffer

        Parameters
        ----------
        start : int
            at which byte to start reading
        len : int
            length in bytes to read
        step : int
            byte step size (how many bytes to skip)
        dtype : np.dtype
            type of the number to read

        Returns
        -------
        array : np.andrray
        '''

        return np.frombuffer(self.mv_[start:start+length:step],
                             dtype=dtype)

    def write_ndarray(self, array, start, step):
        '''Write a numpy array to the buffer

        Parameters
        ----------
        array : np.ndarray
            array to save to the file
        start : int
            start in bytes
        step : int
            byte step size (how many bytes to skip)
        '''

        wrapper = np.frombuffer(self.mv_[start::step],
                                dtype=array.dtype)

        np.copyto(wrapper[:array.size], array, casting='no')

    def read_text(self, start, length, step=1, encoding='ascii'):
        '''Read text from the binary buffer

        Parameters
        ----------
        start : int
            start in bytes
        length : int
            length in bytes to read
        step : int
            byte step size
        encoding : str
            encoding used
        '''
        return self.mv_[start:start+length:step].tobytes().decode(encoding)

    def save(self, filepath=None):
        '''Save the binary buffer to a file

        Parameters
        ----------
        filepath : str
            path where to save the data

        Notes
        -----
            Overwrites to original file if no filepath
            is specified.
        '''

        filepath = filepath if filepath else self.filepath_

        if not filepath:
            return

        with open(filepath, "wb") as fp:
            fp.write(self.mv_)

        self.filepath_ = filepath

    def load(self, filepath=None):
        '''load a file

        Parameters
        ----------
        filepath : str
            path to the file to load

        Notes
        -----
            If not filepath is specified, then the opened file is simply
            reloaded.
        '''

        filepath = filepath if filepath else self.filepath_

        if not filepath:
            return

        # convert to a list if only a single file is given
        if isinstance(filepath, str):
            filepath = [filepath]

        # get size of all files
        sizes = [os.path.getsize(path) for path in filepath]
        memorysize = sum(sizes)

        # allocate memory
        buffer = memoryview(bytearray(b'0'*memorysize))

        # read files and concatenate them
        sizes_tmp = [0] + sizes
        for i_path, path in enumerate(filepath):
            with open(path, "br") as fp:
                fp.readinto(buffer[sizes_tmp[i_path]:])

        self.filepath_ = filepath
        self.sizes_ = sizes
        self.mv_ = buffer

    def append(self, binary_buffer):
        '''Append another binary buffer to this one

        Parameters
        ----------
        binary_buffer : BinaryBuffer
            buffer to append
        '''

        assert(isinstance(binary_buffer, BinaryBuffer))

        self.mv_ = memoryview(bytearray(self.mv_) +
                              bytearray(binary_buffer.mv_))
        self.sizes_.append(len(binary_buffer))
