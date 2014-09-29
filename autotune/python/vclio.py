import sys
import os
import utils

def append_include(data, path):
    include_name = '#include "' + path +'"\n'
    already_included = data.find(include_name)
    if already_included == -1:
        insert_index = data.index('\n', data.index('#define')) + 1
        return data[:insert_index] + '\n' + include_name + data[insert_index:]
    return data

def generate_viennacl_headers(viennacl_root, device, datatype, operation, additional_parameters, parameters):
    builtin_database_dir = os.path.join(viennacl_root, "device_specific", "builtin_database")
    if not os.path.isdir(builtin_database_dir):
        raise EnvironmentError('ViennaCL root path is incorrect. Cannot access ' + builtin_database_dir + '!\n'
                                'Your version of ViennaCL may be too old and/or corrupted.')

    function_name_dict = { vcl.float32: 'add_4B',
                           vcl.float64: 'add_8B' }

    additional_parameters_dict = {'N':  "char_to_type<'N'>",
                                  'T':  "char_to_type<'T'>"}

    #Create the device-specific headers
    cpp_device_name = utils.sanitize_string(device.name)
    function_name = function_name_dict[datatype]
    operation = operation.replace('-','_')

    cpp_class_name = operation + '_template'
    header_name = cpp_device_name + ".hpp"
    function_declaration = 'inline void ' + function_name + '(' + ', '.join(['database_type<' + cpp_class_name + '::parameters_type> & db'] + \
                                                                          [additional_parameters_dict[x] for x in additional_parameters]) + ')'

    device_type_prefix = utils.DEVICE_TYPE_PREFIX[device.type]
    vendor_prefix = utils.VENDOR_PREFIX[device.vendor_id]
    architecture_family = vcl.opencl.architecture_family(device.vendor_id, device.name)

    header_hierarchy = ["devices", device_type_prefix, vendor_prefix, architecture_family]
    header_directory = os.path.join(builtin_database_dir, *header_hierarchy)
    header_path = os.path.join(header_directory, header_name)

    if not os.path.exists(header_directory):
        os.makedirs(header_directory)

    if os.path.exists(header_path):
        with open (header_path, "r") as myfile:
            data=myfile.read()
    else:
        data = ''

    if not data:
        ifndef_suffix = ('_'.join(header_hierarchy) + '_hpp_').upper()
        data =  ('#ifndef VIENNACL_DEVICE_SPECIFIC_BUILTIN_DATABASE_' + ifndef_suffix + '\n'
            '#define VIENNACL_DEVICE_SPECIFIC_BUILTIN_DATABASE_' + ifndef_suffix + '\n'
            '\n'
            '#include "viennacl/device_specific/forwards.h"\n'
            '#include "viennacl/device_specific/builtin_database/common.hpp"\n'
            '\n'
            'namespace viennacl{\n'
            'namespace device_specific{\n'
            'namespace builtin_database{\n'
            'namespace devices{\n'
            'namespace '  + device_type_prefix + '{\n'
            'namespace '  + vendor_prefix + '{\n'
            'namespace '  + architecture_family + '{\n'
            'namespace '  + cpp_device_name + '{\n'
            '\n'
            '}\n'
            '}\n'
            '}\n'
            '}\n'
            '}\n'
            '}\n'
            '}\n'
            '}\n'
            '#endif\n'
            '')

    data = append_include(data, 'viennacl/device_specific/templates/' + cpp_class_name + '.hpp')

    add_to_database_arguments = [vendor_prefix + '_id', utils.DEVICE_TYPE_CL_NAME[device.type], 'ocl::'+architecture_family,
                  '"' + device.name + '"',  cpp_class_name + '::parameters' + str(parameters)]
    core = '  db.' + function_name + '(' + ', '.join(add_to_database_arguments) + ');'

    already_declared = data.find(function_declaration)
    if already_declared==-1:
        substr = 'namespace '  + cpp_device_name + '{\n'
        insert_index = data.index(substr) + len(substr)
        data = data[:insert_index] + '\n' + function_declaration + '\n{\n' + core + '\n}\n' + data[insert_index:]
    else:
        i1 = data.find('{', already_declared)
        if data[i1-1]=='\n':
            i1 = i1 - 1
        i2 = data.find('}', already_declared) + 1
        data = data[:i1]  + '\n{\n' + core + '\n}' + data[i2:]

    #Write the header file
    with open(header_path, "w+") as myfile:
        myfile.write(data)

    #Updates the global ViennaCL headers
    with open(os.path.join(builtin_database_dir, operation + '.hpp'), 'r+') as operation_header:
        data = operation_header.read()
        data = append_include(data, os.path.relpath(header_path, os.path.join(viennacl_root, os.pardir)))

        scope_name = '_'.join(('init', operation) + additional_parameters)
        scope = data.index(scope_name)
        function_call = '  ' + '::'.join(header_hierarchy + [cpp_device_name, function_name]) + '(' + ', '.join(['result'] + [additional_parameters_dict[k] + '()' for k in additional_parameters]) + ')'
        if function_call not in data:
            insert_index = data.rindex('\n', 0, data.index('return result', scope))
            data = data[:insert_index] + function_call + ';\n' + data[insert_index:]

        operation_header.seek(0)
        operation_header.truncate()
        operation_header.write(data)
