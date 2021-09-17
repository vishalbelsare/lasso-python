
import os
import sys
import ansa
import json
import signal
import inspect
import logging
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from lasso.logging import str_error, str_warn, str_info
from lasso.utils.ConsoleColoring import ConsoleColoring
from settings import (
    FUNCTION_NAME,
    DEFAULT_REST_SERVER_PORT,
    SERVER_NAME,
    ENTITY_ID,
    ENTITY_ANSA_TYPE,
)
from server_html import SERVER_HTML_TEMPLATE
from materialize_min_js import (
    WEB_MATERIALIZE_MIN_JS,
    WEB_MATERIALIZE_MIN_CSS,
)

# messages
_msg_site_packages_dir_not_set = "Environment variable '{0}' was not specified. Assuming all required packages where installed already somewhere else."
_msg_import_error = '''

    ImportError: {0}

    Please install the package for ANSA as follows:

    (1) Create and activate a conda environment with python 3.3
        - 'conda create -n py33 python=3.3'
        - Win: 'activate py33' or Linux: 'source activate py33'

    (2) Install required packages
        - 'conda install flask'

    (3) Run the server with the option '--python33-path'
        - 'python -m lasso.ansa.rest.server --python33-path path/to/anaconda/envs/py33'

        Or set the environment variable 'ANSA_REST_SITE_PACKAGES_PATH'
        - csh : 'setenv ANSA_REST_SITE_PACKAGES_PATH path/to/anaconda/envs/py33'
        - bash: 'export ANSA_REST_SITE_PACKAGES_PATH="path/to/anaconda/envs/py33"'
        - ps  : '$env:ANSA_REST_SITE_PACKAGES_PATH = "path/to/anaconda/envs/py33"'
        - cmd : 'setx ANSA_REST_SITE_PACKAGES_PATH "path/to/anaconda/envs/py33"'

    (4) Enjoy life ♥
'''
_msg_missing_json_data = '''Missing json data.

A json object with three entries is expected:
 - {function_name}: str (required, ansa function name e.g. ansa.base.GetEntity)
 - args: list (arguments to the function)
 - kwargs: obj (named arguments for the function)
'''.format(function_name=FUNCTION_NAME)
_msg_missing_function_name = "message json data requires the attribute '{0}'".format(
    FUNCTION_NAME)
_msg_wrong_type = "{0} must be of type {1}"
_msg_shutdown = "Shutting down ANSA server."
_msg_only_method_post = "Only method post is supported for this address."
_msg_missing_entitiy_id = "An entity of ansa_type '{0}' is missing an id."
_msg_entity_not_found = "An ANSA entity with id '{0}' and type {1} can not be found."

# set the path for the site packages folder
try:
    sys.path.append(os.environ["ANSA_REST_SITE_PACKAGES_PATH"])
except KeyError as err:
    print(str_warn(_msg_site_packages_dir_not_set.format(
        "ANSA_REST_SITE_PACKAGES_PATH")))

# try importing flask here
# if the environment is set up correctly, the package should be
# found in sys.path
try:
    import flask
    from flask import (
        Flask,
        json,
        jsonify,
        request,
    )
except ImportError as err:
    raise RuntimeError(str_error(_msg_import_error.format(str(err))))


class AnsaJsonEncoder(json.JSONEncoder):
    ''' Custom JSON encoder for ANSA 
    '''

    def default(self, obj: object):
        ''' encoder function
        '''

        if isinstance(obj, ansa.base.Entity):

            ansa_dict = {
                prop_name: prop_value
                for prop_name, prop_value in inspect.getmembers(obj)
                if prop_name and not callable(prop_value)
            }
            ansa_dict["id"] = obj._id
            ansa_dict["ansa_type"] = obj.ansa_type(ansa.base.CurrentDeck())

            return ansa_dict

        elif isinstance(obj, bytes):
            return obj.decode("utf-8")

        return json.JSONEncoder.default(self, obj)


def _dict_is_ansa_entity(obj: dict):
    ''' Checks if a dictionary is an ansa entity

    Parameters
    ----------
    obj: `dict`
        object to check

    Returns
    -------
    is_ansa_entity: `bool`
    '''

    if isinstance(obj, dict) and ENTITY_ANSA_TYPE in obj:
        return True
    else:
        return False


def _deserialize_ansa_entity(dict_entity: dict):
    ''' deserializes an ansa entity

    Parameters
    ----------
    dict_entity: `dict`
        deserializes an ansa entity from a dict

    Returns
    -------
    entity: `ansa.base.Entity`

    Raises
    ------
    ValueError: if entity was not found 
    '''

    # we assume that this is given
    ansa_type = dict_entity[ENTITY_ANSA_TYPE]

    # check if an id is there
    if ENTITY_ID not in dict_entity:
        raise ValueError(_msg_missing_entitiy_id.format(
            dict_entity[ENTITY_ANSA_TYPE]))

    entity_id = dict_entity[ENTITY_ID]

    # in case you wonder, yes all of this is expensive ...
    entity = ansa.base.GetEntity(
        ansa.base.CurrentDeck(),
        ansa_type,
        entity_id
    )

    if entity == None:
        raise ValueError(_msg_entity_not_found.format(entity_id, ansa_type))

    return entity


def _deserialize_obj_rest(obj: object):
    ''' Deserializes an object from the REST API such as Lists or ANSA entities

    Parameters
    ----------
    obj: `object`
        object to deserialize

    Returns
    -------
    obj: `object`
        deserialized object 

    '''

    # DICT
    if isinstance(obj, dict):

        # sir, we have an entity
        if _dict_is_ansa_entity(obj):
            return _deserialize_ansa_entity(obj)
        # just another dict
        else:
            return {
                _deserialize_obj_rest(key): _deserialize_obj_rest(value)
                for key, value in obj.items()
            }
    # LIST
    elif isinstance(obj, (list, tuple)):
        return [_deserialize_obj_rest(value) for value in obj]
    # all fine
    else:
        return obj


class AnsaFunction:
    ''' Utility class for managing ansa function data
    '''

    def __init__(self, function_name: str, args: list, kwargs: dict):
        ''' Initialize an ansa function

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
        ansa_function : `AnsaFunction`
            ansa function wrapper

        '''

        # set stuff
        self.function_name = function_name
        self.args = args
        self.kwargs = kwargs
        self.result = None

    def run(self):
        ''' Run the ansa function

        Returns
        -------
        result : `object`
            whatever came out of the function
        '''

        # seperate module path from function name
        module_name, function_name = self.function_name.rsplit('.', 1)

        # import module
        my_module = __import__(module_name, globals(),
                               locals(), (function_name, ), 0)

        # get function from module
        my_function = getattr(my_module, function_name)

        # run function
        self.result = my_function(*self.args, **self.kwargs)

        return self.result


def parse_json_rest(json_data: dict):
    ''' Parse JSON data originating from REST

    Parameters
    ----------
    json_data : `dict`
        json data as dict

    Returns
    -------
    ansa_function : `AnsaFunction`
        ansa function wrapper object

    Raises:
    -------
    ValueError: if anything was missing or wrong
    '''

    if json_data == None:
        raise ValueError(_msg_missing_json_data)

    if not FUNCTION_NAME in json_data:
        raise ValueError(_msg_missing_function_name)

    # parse args
    args = json_data["args"] if "args" in json_data else []
    if not isinstance(args, (list, tuple)):
        raise ValueError(_msg_wrong_type.format("args", "list"))

    # parse lwargs
    kwargs = json_data["kwargs"] if "kwargs" in json_data else {}
    if not isinstance(kwargs, dict):
        raise ValueError(_msg_wrong_type.format("kwargs", "dict"))

    # find and deserialize ansa entities in json data
    args = _deserialize_obj_rest(args)
    kwargs = _deserialize_obj_rest(kwargs)

    return AnsaFunction(
        function_name=json_data[FUNCTION_NAME],
        args=args,
        kwargs=kwargs,
    )


def print_header():

    header = '''
    ANSA Remote Scripting Server by {0}
    ------------------------------------------
    '''.format(ConsoleColoring.blue("LASSO GmbH", light=True))

    print(header)

##############################################
#                   FLASK
##############################################


# initiate flask app
app = Flask(SERVER_NAME)
app.json_encoder = AnsaJsonEncoder


@app.route("/shutdown")
def handle_terminate():
    ''' This function terminates the REST server remotely
    '''

    # htis was a little too rude, even for me
    # os.kill(os.getpid(), signal.SIGTERM)
    # return "Shutting down server"

    func = request.environ.get('werkzeug.server.shutdown')
    func()
    return _msg_shutdown


@app.route("/")
def infoPage():
    ''' Display base info page
    '''
    try:
        return SERVER_HTML_TEMPLATE.format(
            address=request.base_url,
            address_run=request.base_url + "run/",
            address_shutdown=request.base_url + "shutdown",
            function_name=FUNCTION_NAME,
            args="args",
            kwargs="kwargs",
            materialize_css=WEB_MATERIALIZE_MIN_CSS,
            materialize_js=WEB_MATERIALIZE_MIN_JS,
        )
    except Exception as err:
        return str(err)


@app.route('/run/', methods=['POST'])
def runAnsaFunction():
    ''' Runs an ansa function from REST
    '''

    try:
        if request.method == 'POST':

            # json request
            json_data = request.get_json()
            # form request
            if not json_data:
                function_name = request.form.get(FUNCTION_NAME, None)
                if function_name == None:
                    raise ValueError(_msg_missing_function_name)

                args = request.form.get("args", "[]")
                if not args:
                    args = []
                print(args, type(args))
                args = flask.json.loads(args)
                print(args, type(args))

                kwargs = request.form.get("kwargs")
                if not kwargs:
                    kwargs = "{}"
                print(kwargs, type(kwargs))
                kwargs = flask.json.loads(kwargs)
                print(kwargs, type(kwargs))

                json_data = {
                    FUNCTION_NAME: request.form[FUNCTION_NAME],
                    "args": args,
                    "kwargs": kwargs,
                }
                print(json_data)

            # parse arguments
            ansa_function = parse_json_rest(json_data)

            # run function
            ansa_function.run()

            # return the thing
            return jsonify({
                "success": True,
                "payload": ansa_function.result,
            })

        else:
            return _msg_only_method_post

    except Exception as err:
        return jsonify({
            "success": False,
            "payload": str(err),
        })

def serve(port: int = DEFAULT_REST_SERVER_PORT):
    ''' Initializes the flask REST service
    '''
    print_header()
    app.run(port=port)
