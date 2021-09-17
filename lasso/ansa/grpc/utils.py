

import io
import os
import sys
import pickle
import inspect
import importlib

sys.path.append(os.path.dirname(__file__))
import AnsaGRPC_pb2
import AnsaGRPC_pb2_grpc


def pickle_object(obj):
    ''' Pickle a python object as bytes

    Parameters
    ----------
    obj : `object`
        anything you want

    Returns
    -------
    arg_as_bytes : `bytes`
        pickled object
    '''
    fake_file = io.BytesIO()
    pickle.dump(obj, fake_file)
    arg_as_bytes = fake_file.getvalue()
    return arg_as_bytes


class Entity:
    ''' This class is an entity placeholder 

    Notes
    -----
        This entity placehodler replaces original
        ansa entities when being transferred to a 
        remote host. Since the ANSA library is not
        installed remotely, we need to emulate an 
        entity.
    '''

    def __init__(self, id, ansa_type):
        ''' Initialize an entity placeholder

        Parameters
        ----------
        id : `int`
            entity id
        ansa_type : `str`
            ansa entity type as string
        '''

        self._id = id
        self._ansa_type = ansa_type

    def assign_props(self, obj):
        ''' Assign properties of another entity to this one

        Parameters
        ----------
        obj : `object`
            anything that has member properties
        '''

        for prop_name, prop_value in inspect.getmembers(obj):
            if prop_name and not callable(prop_value):
                setattr(self, prop_name, prop_value)

    @property
    def id(self):
        return self._id

    @property
    def ansa_type(self):
        return self._ansa_type

    def __repr__(self):
        ''' Get the string representation of the entity

        Returns
        -------
        repr : `str`
            string representation
        '''
        return "<Entity: {0} type: {1} id:{2}>".format(hex(id(self)), self.ansa_type, self.id)

    '''
    * ansa_type(deck: int) -> str
    * card_fields(deck) -> List[str]
    * is_usable() -> bool
    * get_entity_values(deck, fields: List[str] ) -> dict or None
    * set_entity_values(deck, fields) -> int
    '''
