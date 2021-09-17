
import os
import sys
if not hasattr(sys, 'argv'):
    sys.argv = ['']
import grpc
import pickle
import argparse
import typing

from lasso.logging import str_info
from lasso.utils.ConsoleColoring import ConsoleColoring
from . import AnsaGRPC_pb2
from . import AnsaGRPC_pb2_grpc
from .settings import GRPC_DEFAULT_PORT
from .utils import pickle_object

# messages
_msg_using_address = "Using address '{0}'."


def serialize_args(args, kwargs) -> typing.List[AnsaGRPC_pb2.PickledObject]:
    ''' Convert arguments for messaging

    Parameters
    -----------
    args : `list`
        arguments
    kwargs : `dict`
        dictionary of named args

    Returns
    -------
    serialied_args : `list` of `lasso.ansa.grpc.PickledObject`
        arguments serialized with protobuf interface
    '''

    serialized_args = []

    # convert args
    for i_arg, arg in enumerate(args):

        pickled_object = AnsaGRPC_pb2.PickledObject(
            data=pickle_object(arg))

        function_argument = AnsaGRPC_pb2.FunctionArgument(
            index=i_arg, name="", value=pickled_object)

        serialized_args.append(function_argument)

    # convert kwargs
    for arg_name, arg_value in kwargs.items():

        pickled_object = AnsaGRPC_pb2.PickledObject(
            data=pickle_object(arg_value))

        function_argument = AnsaGRPC_pb2.FunctionArgument(
            index=-1, name=arg_name, value=pickled_object)

        serialized_args.append(function_argument)

    return serialized_args


class AnsaClient(object):
    ''' An ANSA Client manages the communication to the remote ANSA
    '''

    def __init__(self, stub: AnsaGRPC_pb2_grpc.LassoAnsaDriverStub):
        ''' Create an AnsaClient for remote scripting

        Parameters
        ----------
        stub : `AnsaGRPC_pb2_grpc.LassoAnsaDriverStub`
            grpc stub from connection

        Examples
        --------

            Please see the docs or the script:
            'lasso/ansa/grpc/example_client.py'
        '''

        assert(isinstance(stub, AnsaGRPC_pb2_grpc.LassoAnsaDriverStub))

        self.stub = stub

    def run(self, function_name, *args, **kwargs) -> object:
        ''' Run an arbitrary ANSA function

        Parameters
        ----------
        function_name : `str`
            full module path of the ansa function to call as string
        *args : `object`
            arguments passed to function in ANSA
        **kwargs : `object`
            named arguments passed to function in ANSA

        Returns
        -------
        anything : `object`
            whatever ANSA returns

        Examples
        --------
            >>> entity = client.run("ansa.base.CreateEntity", 1, "POINT")
            >>> client.run("ansa.base.SetEntityCardValues",
            >>>     deck=1,
            >>>     entity=entity,
            >>>     fields={
            >>>         "X": 4
            >>>     })

        '''

        # Build message
        ansa_function = AnsaGRPC_pb2.AnsaFunction(
            name=function_name,
            args=serialize_args(args, kwargs)
        )

        # send function and get return value
        pickled_object = self.stub.RunAnsaFunction(ansa_function)

        # deserialize message and return
        return pickle.loads(pickled_object.data)

    def shutdown(self):
        ''' shut down the remote server
        '''

        empty = AnsaGRPC_pb2.Empty()
        self.stub.Shutdown(empty)

    def _import_modules(self):

        # file which contains all module information
        _filepath = os.path.join(
            os.path.dirname(__file__), "ansa_scraped.json")


def get_grpc_connection_options() -> list:
    ''' Get the options for the grpc connection
    '''

    gigabyte = 1024 ** 3

    return [
        ('grpc.max_send_message_length', gigabyte),
        ('grpc.max_receive_message_length', gigabyte)
    ]


def connect_interactive(address: str):
    ''' Run a client connection on a specific port

    Parameters
    ----------
    address . `str`
        ip address to connect to
    '''

    # logging
    print(str_info(_msg_using_address.format(address)))
    print()

    # get options for rpc bridge
    options = get_grpc_connection_options()

    # open channel and do the thign
    with grpc.insecure_channel(address, options) as channel:
        stub = AnsaGRPC_pb2_grpc.LassoAnsaDriverStub(channel)

        client = AnsaClient(stub)

        import code
        code.interact(local=locals())


def print_header():

    header = '''
    ANSA Remote Scripting Client by {0}
    ------------------------------------------
    '''.format(ConsoleColoring.blue("LASSO GmbH", light=True))

    print(header)


def parse_args() -> argparse.Namespace:
    ''' Parses arguments when being run from the command line

    Returns
    -------
    parser : `argparse.Namespace`
        parsed command line arguments
    '''

    print_header()

    default_address = "localhost:" + str(GRPC_DEFAULT_PORT)

    parser = argparse.ArgumentParser(
        description='GRPC Server for ANSA Remote Scripting from LASSO GmbH.')
    parser.add_argument('--address',
                        type=str,
                        default=default_address,
                        help='Address of machine running an ANSA GRPC server.')

    return parser.parse_args()


if __name__ == '__main__':

    # fetch info
    parser = parse_args()

    # run interactive shell
    connect_interactive(parser.address)
