# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: AnsaGRPC.proto

from google.protobuf import symbol_database as _symbol_database
from google.protobuf import reflection as _reflection
from google.protobuf import message as _message
from google.protobuf import descriptor as _descriptor
import sys
_b = sys.version_info[0] < 3 and (lambda x: x) or (lambda x: x.encode('latin1'))
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


DESCRIPTOR = _descriptor.FileDescriptor(
    name='AnsaGRPC.proto',
    package='lasso',
    syntax='proto3',
    serialized_options=None,
    serialized_pb=_b(
        '\n\x0e\x41nsaGRPC.proto\x12\x05lasso\"\x07\n\x05\x45mpty\"\x1d\n\rPickledObject\x12\x0c\n\x04\x64\x61ta\x18\x01 \x01(\x0c\"T\n\x10\x46unctionArgument\x12\r\n\x05index\x18\x01 \x01(\x05\x12\x0c\n\x04name\x18\x02 \x01(\t\x12#\n\x05value\x18\x03 \x01(\x0b\x32\x14.lasso.PickledObject\"C\n\x0c\x41nsaFunction\x12\x0c\n\x04name\x18\x01 \x01(\t\x12%\n\x04\x61rgs\x18\x02 \x03(\x0b\x32\x17.lasso.FunctionArgument2{\n\x0fLassoAnsaDriver\x12>\n\x0fRunAnsaFunction\x12\x13.lasso.AnsaFunction\x1a\x14.lasso.PickledObject\"\x00\x12(\n\x08Shutdown\x12\x0c.lasso.Empty\x1a\x0c.lasso.Empty\"\x00\x62\x06proto3')
)


_EMPTY = _descriptor.Descriptor(
    name='Empty',
    full_name='lasso.Empty',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    fields=[
    ],
    extensions=[
    ],
    nested_types=[],
    enum_types=[
    ],
    serialized_options=None,
    is_extendable=False,
    syntax='proto3',
    extension_ranges=[],
    oneofs=[
    ],
    serialized_start=25,
    serialized_end=32,
)


_PICKLEDOBJECT = _descriptor.Descriptor(
    name='PickledObject',
    full_name='lasso.PickledObject',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    fields=[
        _descriptor.FieldDescriptor(
            name='data', full_name='lasso.PickledObject.data', index=0,
            number=1, type=12, cpp_type=9, label=1,
            has_default_value=False, default_value=_b(""),
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR),
    ],
    extensions=[
    ],
    nested_types=[],
    enum_types=[
    ],
    serialized_options=None,
    is_extendable=False,
    syntax='proto3',
    extension_ranges=[],
    oneofs=[
    ],
    serialized_start=34,
    serialized_end=63,
)


_FUNCTIONARGUMENT = _descriptor.Descriptor(
    name='FunctionArgument',
    full_name='lasso.FunctionArgument',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    fields=[
        _descriptor.FieldDescriptor(
            name='index', full_name='lasso.FunctionArgument.index', index=0,
            number=1, type=5, cpp_type=1, label=1,
            has_default_value=False, default_value=0,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR),
        _descriptor.FieldDescriptor(
            name='name', full_name='lasso.FunctionArgument.name', index=1,
            number=2, type=9, cpp_type=9, label=1,
            has_default_value=False, default_value=_b("").decode('utf-8'),
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR),
        _descriptor.FieldDescriptor(
            name='value', full_name='lasso.FunctionArgument.value', index=2,
            number=3, type=11, cpp_type=10, label=1,
            has_default_value=False, default_value=None,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR),
    ],
    extensions=[
    ],
    nested_types=[],
    enum_types=[
    ],
    serialized_options=None,
    is_extendable=False,
    syntax='proto3',
    extension_ranges=[],
    oneofs=[
    ],
    serialized_start=65,
    serialized_end=149,
)


_ANSAFUNCTION = _descriptor.Descriptor(
    name='AnsaFunction',
    full_name='lasso.AnsaFunction',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    fields=[
        _descriptor.FieldDescriptor(
            name='name', full_name='lasso.AnsaFunction.name', index=0,
            number=1, type=9, cpp_type=9, label=1,
            has_default_value=False, default_value=_b("").decode('utf-8'),
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR),
        _descriptor.FieldDescriptor(
            name='args', full_name='lasso.AnsaFunction.args', index=1,
            number=2, type=11, cpp_type=10, label=3,
            has_default_value=False, default_value=[],
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR),
    ],
    extensions=[
    ],
    nested_types=[],
    enum_types=[
    ],
    serialized_options=None,
    is_extendable=False,
    syntax='proto3',
    extension_ranges=[],
    oneofs=[
    ],
    serialized_start=151,
    serialized_end=218,
)

_FUNCTIONARGUMENT.fields_by_name['value'].message_type = _PICKLEDOBJECT
_ANSAFUNCTION.fields_by_name['args'].message_type = _FUNCTIONARGUMENT
DESCRIPTOR.message_types_by_name['Empty'] = _EMPTY
DESCRIPTOR.message_types_by_name['PickledObject'] = _PICKLEDOBJECT
DESCRIPTOR.message_types_by_name['FunctionArgument'] = _FUNCTIONARGUMENT
DESCRIPTOR.message_types_by_name['AnsaFunction'] = _ANSAFUNCTION
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Empty = _reflection.GeneratedProtocolMessageType('Empty', (_message.Message,), dict(
    DESCRIPTOR=_EMPTY,
    __module__='AnsaGRPC_pb2'
    # @@protoc_insertion_point(class_scope:lasso.Empty)
))
_sym_db.RegisterMessage(Empty)

PickledObject = _reflection.GeneratedProtocolMessageType('PickledObject', (_message.Message,), dict(
    DESCRIPTOR=_PICKLEDOBJECT,
    __module__='AnsaGRPC_pb2'
    # @@protoc_insertion_point(class_scope:lasso.PickledObject)
))
_sym_db.RegisterMessage(PickledObject)

FunctionArgument = _reflection.GeneratedProtocolMessageType('FunctionArgument', (_message.Message,), dict(
    DESCRIPTOR=_FUNCTIONARGUMENT,
    __module__='AnsaGRPC_pb2'
    # @@protoc_insertion_point(class_scope:lasso.FunctionArgument)
))
_sym_db.RegisterMessage(FunctionArgument)

AnsaFunction = _reflection.GeneratedProtocolMessageType('AnsaFunction', (_message.Message,), dict(
    DESCRIPTOR=_ANSAFUNCTION,
    __module__='AnsaGRPC_pb2'
    # @@protoc_insertion_point(class_scope:lasso.AnsaFunction)
))
_sym_db.RegisterMessage(AnsaFunction)


_LASSOANSADRIVER = _descriptor.ServiceDescriptor(
    name='LassoAnsaDriver',
    full_name='lasso.LassoAnsaDriver',
    file=DESCRIPTOR,
    index=0,
    serialized_options=None,
    serialized_start=220,
    serialized_end=343,
    methods=[
        _descriptor.MethodDescriptor(
            name='RunAnsaFunction',
            full_name='lasso.LassoAnsaDriver.RunAnsaFunction',
            index=0,
            containing_service=None,
            input_type=_ANSAFUNCTION,
            output_type=_PICKLEDOBJECT,
            serialized_options=None,
        ),
        _descriptor.MethodDescriptor(
            name='Shutdown',
            full_name='lasso.LassoAnsaDriver.Shutdown',
            index=1,
            containing_service=None,
            input_type=_EMPTY,
            output_type=_EMPTY,
            serialized_options=None,
        ),
    ])
_sym_db.RegisterServiceDescriptor(_LASSOANSADRIVER)

DESCRIPTOR.services_by_name['LassoAnsaDriver'] = _LASSOANSADRIVER

# @@protoc_insertion_point(module_scope)
