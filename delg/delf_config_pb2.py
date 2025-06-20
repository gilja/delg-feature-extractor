"""
delf_config_pb2
===============

This is an auto-generated module created by the Protocol Buffers compiler from
the `delf/protos/delf_config.proto` file in the original research GitHub repository.
It contains Python classes that represent the DELG model configuration messages
used by this package.

Do not edit this file manually. Any changes to the configuration schema should
be made in the original `.proto` file and then recompiled using the Protocol Buffers
compiler (protoc).

Notes:
------
Author: Duje Giljanović (giljanovic.duje@gmail.com)
License: Apache License 2.0 (same as the official DELG implementation)

This package uses the DELG model originally developed by Google Research and published
in the paper "Unifying Deep Local and Global Features for Image Search" authored by Bingyi Cao,
Andre Araujo, and Jack Sim.

If you use this Python package in your research or any other publication, please cite both this
package and the original DELG paper as follows:

@software{delg,
    title = {delg: A Python Package for Dockerized DELG Implementation},
    author = {Duje Giljanović},
    year = {2025},
    url = {https://github.com/gilja/delg-feature-extractor}
}

@article{cao2020delg,
    title = {Unifying Deep Local and Global Features for Image Search},
    author = {Bingyi Cao and Andre Araujo and Jack Sim},
    journal = {arXiv preprint arXiv:2001.05027},
    year = {2020}
}
"""

# Generated by the protocol buffer compiler.  DO NOT EDIT!

import sys
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database

_b = sys.version_info[0] < 3 and (lambda x: x) or (lambda x: x.encode("latin1"))
_sym_db = _symbol_database.Default()


DESCRIPTOR = _descriptor.FileDescriptor(
    name="delf/protos/delf_config.proto",
    package="delf.protos",
    syntax="proto2",
    serialized_pb=_b(
        '\n\x1d\x64\x65lf/protos/delf_config.proto\x12\x0b\x64\x65lf.protos"\x91\x01\n\x11\x44\x65lfPcaParameters\x12\x11\n\tmean_path\x18\x01 \x01(\t\x12\x1e\n\x16projection_matrix_path\x18\x02 \x01(\t\x12\x0f\n\x07pca_dim\x18\x03 \x01(\x05\x12\x1c\n\ruse_whitening\x18\x04 \x01(\x08:\x05\x66\x61lse\x12\x1a\n\x12pca_variances_path\x18\x05 \x01(\t"\xfc\x01\n\x16\x44\x65lfLocalFeatureConfig\x12\x15\n\x07use_pca\x18\x01 \x01(\x08:\x04true\x12\x14\n\nlayer_name\x18\x02 \x01(\t:\x00\x12\x18\n\riou_threshold\x18\x03 \x01(\x02:\x01\x31\x12\x1d\n\x0fmax_feature_num\x18\x04 \x01(\x05:\x04\x31\x30\x30\x30\x12\x1c\n\x0fscore_threshold\x18\x05 \x01(\x02:\x03\x31\x30\x30\x12\x36\n\x0epca_parameters\x18\x06 \x01(\x0b\x32\x1e.delf.protos.DelfPcaParameters\x12&\n\x17use_resized_coordinates\x18\x07 \x01(\x08:\x05\x66\x61lse"\x82\x01\n\x17\x44\x65lfGlobalFeatureConfig\x12\x15\n\x07use_pca\x18\x01 \x01(\x08:\x04true\x12\x36\n\x0epca_parameters\x18\x02 \x01(\x0b\x32\x1e.delf.protos.DelfPcaParameters\x12\x18\n\x10image_scales_ind\x18\x03 \x03(\x05"\xf8\x02\n\nDelfConfig\x12 \n\x12use_local_features\x18\x07 \x01(\x08:\x04true\x12>\n\x11\x64\x65lf_local_config\x18\x03 \x01(\x0b\x32#.delf.protos.DelfLocalFeatureConfig\x12"\n\x13use_global_features\x18\x08 \x01(\x08:\x05\x66\x61lse\x12@\n\x12\x64\x65lf_global_config\x18\t \x01(\x0b\x32$.delf.protos.DelfGlobalFeatureConfig\x12\x12\n\nmodel_path\x18\x01 \x01(\t\x12\x1e\n\x0fis_tf2_exported\x18\n \x01(\x08:\x05\x66\x61lse\x12\x14\n\x0cimage_scales\x18\x02 \x03(\x02\x12\x1a\n\x0emax_image_size\x18\x04 \x01(\x05:\x02-1\x12\x1a\n\x0emin_image_size\x18\x05 \x01(\x05:\x02-1\x12 \n\x11use_square_images\x18\x06 \x01(\x08:\x05\x66\x61lse'
    ),
)


_DELFPCAPARAMETERS = _descriptor.Descriptor(
    name="DelfPcaParameters",
    full_name="delf.protos.DelfPcaParameters",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    fields=[
        _descriptor.FieldDescriptor(
            name="mean_path",
            full_name="delf.protos.DelfPcaParameters.mean_path",
            index=0,
            number=1,
            type=9,
            cpp_type=9,
            label=1,
            has_default_value=False,
            default_value=_b("").decode("utf-8"),
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            options=None,
        ),
        _descriptor.FieldDescriptor(
            name="projection_matrix_path",
            full_name="delf.protos.DelfPcaParameters.projection_matrix_path",
            index=1,
            number=2,
            type=9,
            cpp_type=9,
            label=1,
            has_default_value=False,
            default_value=_b("").decode("utf-8"),
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            options=None,
        ),
        _descriptor.FieldDescriptor(
            name="pca_dim",
            full_name="delf.protos.DelfPcaParameters.pca_dim",
            index=2,
            number=3,
            type=5,
            cpp_type=1,
            label=1,
            has_default_value=False,
            default_value=0,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            options=None,
        ),
        _descriptor.FieldDescriptor(
            name="use_whitening",
            full_name="delf.protos.DelfPcaParameters.use_whitening",
            index=3,
            number=4,
            type=8,
            cpp_type=7,
            label=1,
            has_default_value=True,
            default_value=False,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            options=None,
        ),
        _descriptor.FieldDescriptor(
            name="pca_variances_path",
            full_name="delf.protos.DelfPcaParameters.pca_variances_path",
            index=4,
            number=5,
            type=9,
            cpp_type=9,
            label=1,
            has_default_value=False,
            default_value=_b("").decode("utf-8"),
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            options=None,
        ),
    ],
    extensions=[],
    nested_types=[],
    enum_types=[],
    options=None,
    is_extendable=False,
    syntax="proto2",
    extension_ranges=[],
    oneofs=[],
    serialized_start=47,
    serialized_end=192,
)


_DELFLOCALFEATURECONFIG = _descriptor.Descriptor(
    name="DelfLocalFeatureConfig",
    full_name="delf.protos.DelfLocalFeatureConfig",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    fields=[
        _descriptor.FieldDescriptor(
            name="use_pca",
            full_name="delf.protos.DelfLocalFeatureConfig.use_pca",
            index=0,
            number=1,
            type=8,
            cpp_type=7,
            label=1,
            has_default_value=True,
            default_value=True,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            options=None,
        ),
        _descriptor.FieldDescriptor(
            name="layer_name",
            full_name="delf.protos.DelfLocalFeatureConfig.layer_name",
            index=1,
            number=2,
            type=9,
            cpp_type=9,
            label=1,
            has_default_value=True,
            default_value=_b("").decode("utf-8"),
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            options=None,
        ),
        _descriptor.FieldDescriptor(
            name="iou_threshold",
            full_name="delf.protos.DelfLocalFeatureConfig.iou_threshold",
            index=2,
            number=3,
            type=2,
            cpp_type=6,
            label=1,
            has_default_value=True,
            default_value=float(1),
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            options=None,
        ),
        _descriptor.FieldDescriptor(
            name="max_feature_num",
            full_name="delf.protos.DelfLocalFeatureConfig.max_feature_num",
            index=3,
            number=4,
            type=5,
            cpp_type=1,
            label=1,
            has_default_value=True,
            default_value=1000,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            options=None,
        ),
        _descriptor.FieldDescriptor(
            name="score_threshold",
            full_name="delf.protos.DelfLocalFeatureConfig.score_threshold",
            index=4,
            number=5,
            type=2,
            cpp_type=6,
            label=1,
            has_default_value=True,
            default_value=float(100),
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            options=None,
        ),
        _descriptor.FieldDescriptor(
            name="pca_parameters",
            full_name="delf.protos.DelfLocalFeatureConfig.pca_parameters",
            index=5,
            number=6,
            type=11,
            cpp_type=10,
            label=1,
            has_default_value=False,
            default_value=None,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            options=None,
        ),
        _descriptor.FieldDescriptor(
            name="use_resized_coordinates",
            full_name="delf.protos.DelfLocalFeatureConfig.use_resized_coordinates",
            index=6,
            number=7,
            type=8,
            cpp_type=7,
            label=1,
            has_default_value=True,
            default_value=False,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            options=None,
        ),
    ],
    extensions=[],
    nested_types=[],
    enum_types=[],
    options=None,
    is_extendable=False,
    syntax="proto2",
    extension_ranges=[],
    oneofs=[],
    serialized_start=195,
    serialized_end=447,
)

_DELFGLOBALFEATURECONFIG = _descriptor.Descriptor(
    name="DelfGlobalFeatureConfig",
    full_name="delf.protos.DelfGlobalFeatureConfig",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    fields=[
        _descriptor.FieldDescriptor(
            name="use_pca",
            full_name="delf.protos.DelfGlobalFeatureConfig.use_pca",
            index=0,
            number=1,
            type=8,
            cpp_type=7,
            label=1,
            has_default_value=True,
            default_value=True,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            options=None,
        ),
        _descriptor.FieldDescriptor(
            name="pca_parameters",
            full_name="delf.protos.DelfGlobalFeatureConfig.pca_parameters",
            index=1,
            number=2,
            type=11,
            cpp_type=10,
            label=1,
            has_default_value=False,
            default_value=None,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            options=None,
        ),
        _descriptor.FieldDescriptor(
            name="image_scales_ind",
            full_name="delf.protos.DelfGlobalFeatureConfig.image_scales_ind",
            index=2,
            number=3,
            type=5,
            cpp_type=1,
            label=3,
            has_default_value=False,
            default_value=[],
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            options=None,
        ),
    ],
    extensions=[],
    nested_types=[],
    enum_types=[],
    options=None,
    is_extendable=False,
    syntax="proto2",
    extension_ranges=[],
    oneofs=[],
    serialized_start=450,
    serialized_end=580,
)

_DELFCONFIG = _descriptor.Descriptor(
    name="DelfConfig",
    full_name="delf.protos.DelfConfig",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    fields=[
        _descriptor.FieldDescriptor(
            name="use_local_features",
            full_name="delf.protos.DelfConfig.use_local_features",
            index=0,
            number=7,
            type=8,
            cpp_type=7,
            label=1,
            has_default_value=True,
            default_value=True,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            options=None,
        ),
        _descriptor.FieldDescriptor(
            name="delf_local_config",
            full_name="delf.protos.DelfConfig.delf_local_config",
            index=1,
            number=3,
            type=11,
            cpp_type=10,
            label=1,
            has_default_value=False,
            default_value=None,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            options=None,
        ),
        _descriptor.FieldDescriptor(
            name="use_global_features",
            full_name="delf.protos.DelfConfig.use_global_features",
            index=2,
            number=8,
            type=8,
            cpp_type=7,
            label=1,
            has_default_value=True,
            default_value=False,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            options=None,
        ),
        _descriptor.FieldDescriptor(
            name="delf_global_config",
            full_name="delf.protos.DelfConfig.delf_global_config",
            index=3,
            number=9,
            type=11,
            cpp_type=10,
            label=1,
            has_default_value=False,
            default_value=None,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            options=None,
        ),
        _descriptor.FieldDescriptor(
            name="model_path",
            full_name="delf.protos.DelfConfig.model_path",
            index=4,
            number=1,
            type=9,
            cpp_type=9,
            label=1,
            has_default_value=False,
            default_value=_b("").decode("utf-8"),
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            options=None,
        ),
        _descriptor.FieldDescriptor(
            name="is_tf2_exported",
            full_name="delf.protos.DelfConfig.is_tf2_exported",
            index=5,
            number=10,
            type=8,
            cpp_type=7,
            label=1,
            has_default_value=True,
            default_value=False,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            options=None,
        ),
        _descriptor.FieldDescriptor(
            name="image_scales",
            full_name="delf.protos.DelfConfig.image_scales",
            index=6,
            number=2,
            type=2,
            cpp_type=6,
            label=3,
            has_default_value=False,
            default_value=[],
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            options=None,
        ),
        _descriptor.FieldDescriptor(
            name="max_image_size",
            full_name="delf.protos.DelfConfig.max_image_size",
            index=7,
            number=4,
            type=5,
            cpp_type=1,
            label=1,
            has_default_value=True,
            default_value=-1,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            options=None,
        ),
        _descriptor.FieldDescriptor(
            name="min_image_size",
            full_name="delf.protos.DelfConfig.min_image_size",
            index=8,
            number=5,
            type=5,
            cpp_type=1,
            label=1,
            has_default_value=True,
            default_value=-1,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            options=None,
        ),
        _descriptor.FieldDescriptor(
            name="use_square_images",
            full_name="delf.protos.DelfConfig.use_square_images",
            index=9,
            number=6,
            type=8,
            cpp_type=7,
            label=1,
            has_default_value=True,
            default_value=False,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            options=None,
        ),
    ],
    extensions=[],
    nested_types=[],
    enum_types=[],
    options=None,
    is_extendable=False,
    syntax="proto2",
    extension_ranges=[],
    oneofs=[],
    serialized_start=583,
    serialized_end=959,
)

_DELFLOCALFEATURECONFIG.fields_by_name["pca_parameters"].message_type = (
    _DELFPCAPARAMETERS
)
_DELFGLOBALFEATURECONFIG.fields_by_name["pca_parameters"].message_type = (
    _DELFPCAPARAMETERS
)
_DELFCONFIG.fields_by_name["delf_local_config"].message_type = _DELFLOCALFEATURECONFIG
_DELFCONFIG.fields_by_name["delf_global_config"].message_type = _DELFGLOBALFEATURECONFIG
DESCRIPTOR.message_types_by_name["DelfPcaParameters"] = _DELFPCAPARAMETERS
DESCRIPTOR.message_types_by_name["DelfLocalFeatureConfig"] = _DELFLOCALFEATURECONFIG
DESCRIPTOR.message_types_by_name["DelfGlobalFeatureConfig"] = _DELFGLOBALFEATURECONFIG
DESCRIPTOR.message_types_by_name["DelfConfig"] = _DELFCONFIG
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

DelfPcaParameters = _reflection.GeneratedProtocolMessageType(
    "DelfPcaParameters",
    (_message.Message,),
    dict(
        DESCRIPTOR=_DELFPCAPARAMETERS,
        __module__="delf.protos.delf_config_pb2",
    ),
)
_sym_db.RegisterMessage(DelfPcaParameters)

DelfLocalFeatureConfig = _reflection.GeneratedProtocolMessageType(
    "DelfLocalFeatureConfig",
    (_message.Message,),
    dict(
        DESCRIPTOR=_DELFLOCALFEATURECONFIG,
        __module__="delf.protos.delf_config_pb2",
    ),
)
_sym_db.RegisterMessage(DelfLocalFeatureConfig)

DelfGlobalFeatureConfig = _reflection.GeneratedProtocolMessageType(
    "DelfGlobalFeatureConfig",
    (_message.Message,),
    dict(
        DESCRIPTOR=_DELFGLOBALFEATURECONFIG,
        __module__="delf.protos.delf_config_pb2",
    ),
)
_sym_db.RegisterMessage(DelfGlobalFeatureConfig)

DelfConfig = _reflection.GeneratedProtocolMessageType(
    "DelfConfig",
    (_message.Message,),
    dict(
        DESCRIPTOR=_DELFCONFIG,
        __module__="delf.protos.delf_config_pb2",
    ),
)
_sym_db.RegisterMessage(DelfConfig)
