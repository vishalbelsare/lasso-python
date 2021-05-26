
from lasso.utils.ConsoleColoring import ConsoleColoring
from lasso.logging import str_error, str_warn, str_info
import pickle
import os
import sys
import time
import logging
from concurrent import futures

import ansa

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

# messages
_msg_site_packages_dir_not_set = ("Environment variable '{0}' was not specified. "
                                  "Assuming all required packages where installed"
                                  " already somewhere else.")
_msg_port_taken = "Port {} is already in use."
_msg_stopping_server = "Stopping GRPC server"
_msg_invalid_argument_index = ("A negative function argument index of '{0}' is not allowed."
                               " Either set a name or a valid index.")
_msg_some_weird_error = "Encountered error '{0}'. Shutting down service."
_msg_import_error = '''

    ImportError: {0}

    Please install the package for ANSA as follows:

    (1) Create and activate a conda environment with python 3.3
        - 'conda create -n py33 python=3.3'
        - 'activate py33'

    (2) Install required packages
        - 'python -m pip install grpcio enum34 protobuf'

    (3) Run the server with the option '--python33-path'
        - 'python -m lasso.ansa.grpc.server --python33-path path/to/anaconda/envs/py33'

        Or set the environment variable 'ANSA_GRPC_SITE_PACKAGES_PATH'
        - csh : 'setenv ANSA_GRPC_SITE_PACKAGES_PATH path/to/anaconda/envs/py33'
        - bash: 'export ANSA_GRPC_SITE_PACKAGES_PATH="path/to/anaconda/envs/py33"'
        - ps  : '$env:ANSA_GRPC_SITE_PACKAGES_PATH = "path/to/anaconda/envs/py33"'
        - cmd : 'setx ANSA_GRPC_SITE_PACKAGES_PATH "path/to/anaconda/envs/py33"'

    (4) Enjoy life ♥
'''

# this is a utility for the command line usage
try:
    sys.path.append(os.environ["ANSA_GRPC_SITE_PACKAGES_PATH"])
except KeyError:
    print(str_warn(_msg_site_packages_dir_not_set.format(
        "ANSA_GRPC_SITE_PACKAGES_PATH")))

try:
    import AnsaGRPC_pb2
    import AnsaGRPC_pb2_grpc
    import grpc
    from utils import Entity, pickle_object
except ImportError as err:
    raise RuntimeError(str_error(_msg_import_error.format(str(err))))


def print_header():

    header = '''
    ANSA Remote Scripting Server by {0}
    ------------------------------------------
    '''.format(ConsoleColoring.blue("LASSO GmbH", light=True))

    print(header)


def _serialize(obj):
    ''' Serializes an arbitrary object for transfer

    Parameters
    ----------
    obj : `obj`
        object to be serialized for transfer

    Returns
    -------
    pickled_object : `lasso.ansa.rpc.PickledObject`
        protobuf serialized message

    Notes
    -----
        Converts any ansa entities to remote placeholders.
    '''

    # first convert ansa entites to fake entities
    if isinstance(obj, ansa.base.Entity):
        obj = _serialize_ansa_entity(obj)
    elif isinstance(obj, list):
        obj = [_serialize_ansa_entity(entry) if isinstance(entry, ansa.base.Entity)
               else entry for entry in obj]
    elif isinstance(obj, tuple):
        obj = tuple(_serialize_ansa_entity(entry) if isinstance(entry, ansa.base.Entity)
                    else entry for entry in obj)
    elif isinstance(obj, dict):
        obj = {
            _serialize_ansa_entity(key) if isinstance(key, Entity) else key:
            _serialize_ansa_entity(value) if isinstance(
                value, Entity) else value
            for key, value in obj.items()
        }

    # then we pickle everything
    return AnsaGRPC_pb2.PickledObject(
        data=pickle_object(obj))


def _serialize_ansa_entity(ansa_entity):
    ''' Replaces an ansa entity by a placeholder entity

    Parameters
    ----------
    ansa_entity : `ansa.base.Entity`
        ansa entity

    Returns
    -------
    entity : `lasso.ansa.Entity`
        entity placeholder instance

    Notes
    -----
        The placeholder has all properties of
        the original instance.
    '''

    assert(isinstance(ansa_entity, ansa.base.Entity))

    # create entity
    entity = Entity(
        id=ansa_entity._id,
        ansa_type=ansa_entity.ansa_type(ansa.base.CurrentDeck())
    )

    # we take all properties with us
    entity.assign_props(ansa_entity)

    # transfer the thing
    return entity


def _deserialize_entity(entity):
    ''' Convert a placeholder entity to an entity

    Parameters
    ----------
    entity : `lasso.ansa.rpc.Entity`
        entity instance to be replaced

    Returns
    -------
    ansa_entity : `ansa.base.Entity`
        ansa entity
    '''

    assert(isinstance(entity, Entity))

    return ansa.base.GetEntity(
        ansa.base.CurrentDeck(),
        entity.ansa_type,
        entity.id
    )


def _convert_any_ansa_entities(obj):
    ''' Converts any ansa entities contained in whatever it is

    Parameters
    ----------
    obj : `object`
        any object to check or convert

    Returns
    -------
    ret : `object`
        object with converted entities

    Notes
    -----
        Useful for converting stuff for lists of entities etc.
    '''

    if isinstance(obj, Entity):
        return _deserialize_entity(obj)
    elif isinstance(obj, list):
        return [_deserialize_entity(entry) if isinstance(entry, Entity)
                else entry for entry in obj]
    elif isinstance(obj, tuple):
        return tuple(_deserialize_entity(entry) if isinstance(entry, Entity)
                     else entry for entry in obj)
    elif isinstance(obj, dict):
        return {
            _deserialize_entity(key) if isinstance(key, Entity) else key:
            _deserialize_entity(value) if isinstance(
                value, Entity) else value
            for key, value in obj.items()
        }
    else:
        return obj


class LassoAnsaDriverServicer(AnsaGRPC_pb2_grpc.LassoAnsaDriverServicer):
    ''' Implementation of the server
    '''

    def __init__(self):
        ''' Create a server executing tasks
        '''
        logging.info(str_info("LassoAnsaDriverServicer.__init__"))
        self.please_shutdown = False

    def _deserialize_args(self, pb_function_arguments):
        ''' Deserialize function arguments

        Parameters
        ----------
        pb_function_arguments : `list` of `lasso.ansa.rpc.FunctionArgument`
            function arguments to iterate over

        Returns
        -------
        args : `list`
            argument list
        kwargs : `dict`
            kwargs dictionary for function
        '''

        # these guys or ladies get returned
        args = []
        kwargs = {}

        for function_argument in pb_function_arguments:

            # get argument info
            argument_index = function_argument.index
            argument_name = function_argument.name
            argument_value = pickle.loads(function_argument.value.data)

            # convert entities back to ansa entities
            argument_value = _convert_any_ansa_entities(argument_value)

            # I belong to KWARGS
            if len(argument_name) != 0:
                kwargs[argument_name] = argument_value

            # I belong to ARGS
            else:

                # sorry, these guys are not allowed
                if argument_index < 0:
                    raise RuntimeError(
                        str_error(_msg_invalid_argument_index.format(argument_index)))

                # extend args if required
                if len(args) <= argument_index:
                    args.extend([None] * (argument_index - len(args) + 1))

                args[argument_index] = argument_value

        return args, kwargs

    def _run_function(self, function_name, args, kwargs):
        ''' This function actually runs a module function

        Parameters
        ----------
        function_name : `str`
            name of the function to execute with full module path
        args : `list`
            argument list
        kwargs : `dict`
            dictionary of named args

        Returns
        -------
        return_value : `object`
            whatever came out of the function

        Notes
        -----
            For security reasons this function tries to import
            the ansa function from the modules and thus does
            not use `eval` or `exec` to run any code.
            This can be broken quite easily but at least it is
            better than nothing.
        '''

        assert(isinstance(args, list))
        assert(isinstance(kwargs, dict))

        # seperate module path from function name
        module_name, function_name = function_name.rsplit('.', 1)

        # import module
        my_module = __import__(module_name, globals(),
                               locals(), (function_name, ), 0)

        # get function from module
        my_function = getattr(my_module, function_name)

        # run function
        return my_function(*args, **kwargs)

    def RunAnsaFunction(self, request: AnsaGRPC_pb2.AnsaFunction, context):
        ''' Implementation of protobuf interface function
        '''

        # get function name
        function_name = request.name
        logging.info("-" * 60)
        logging.info("function: {0}".format(function_name))

        # deserialize function arguments
        args, kwargs = self._deserialize_args(request.args)

        logging.info("args    : {0}".format(args))
        logging.info("kwargs  : {0}".format(kwargs))

        # run the thing
        return_value = self._run_function(function_name, args, kwargs)
        logging.info("return  : {0}".format(return_value))

        # serialize return
        return_anything = _serialize(return_value)

        return return_anything

    def Shutdown(self, request: AnsaGRPC_pb2.Empty, context):
        ''' Shutdown the server
        '''
        self.please_shutdown = True
        return AnsaGRPC_pb2.Empty()


def serve(port, interactive, enable_logging):
    ''' Run the grpc server
    '''

    print_header()

    # set logging
    if enable_logging:
        logging.basicConfig(level=logging.INFO)

    fmt_settings = "{0:14}: {1}"
    logging.info(str_info(fmt_settings.format("port", port)))
    logging.info(str_info(fmt_settings.format("interactive", interactive)))
    logging.info(str_info(fmt_settings.format(
        "enable_logging", enable_logging)))

    # grpc server options
    # We increase the transfer limit from 4MB to 1GB here
    # This is seriously bad since big stuff should be streamed
    # but I'm not getting paid for this.
    gigabyte = 1024 ** 3
    options = [
        ('grpc.max_send_message_length', gigabyte),
        ('grpc.max_receive_message_length', gigabyte)
    ]

    # run server
    # Note: Since ANSA is not threadsafe we allow only 1 worker.
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=1),
        options=options
    )

    # register our driver
    driver_servicer = LassoAnsaDriverServicer()
    AnsaGRPC_pb2_grpc.add_LassoAnsaDriverServicer_to_server(
        driver_servicer, server)
    port = server.add_insecure_port('[::]:{}'.format(port))

    # check if port was fine
    if port == 0:
        logging.error(_msg_port_taken.format(port))
        raise RuntimeError(str_error(_msg_port_taken.format(port)))

    # finally start server
    server.start()

    # let main process wait or make ANSA interactively accessible
    # while the threadpool process handles incoming commands
    try:
        if interactive:
            if enable_logging:
                print()
            import code
            code.interact(local=locals())
        else:
            while True:
                # time.sleep(60 * 60 * 24)
                time.sleep(1)
                if driver_servicer.please_shutdown:
                    raise KeyboardInterrupt()

    except KeyboardInterrupt:
        logging.info(_msg_stopping_server)
        server.stop(0)
    except Exception as err:
        logging.error(str_error(_msg_some_weird_error.format(str(err))))
        server.stop(0)
