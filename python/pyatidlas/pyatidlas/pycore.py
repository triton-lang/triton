from . import _atidlas as _atd
import pyviennacl.pycore as vcl

FetchingPolicy = _atd.fetching_policy_type

class TemplateBase(object):

    Parameters = _atd.template_base.parameters_type

    def __init__(self):
        pass

    @property
    def parameters(self):
        return self._vcl_template.parameters()
        
    def lmem_usage(self, statements):
        return self._vcl_template.lmem_usage(statements.vcl_tuple)
        
    def registers_usage(self, statements):
        return self._vcl_template.registers_usage(statements.vcl_tuple)
        
    def check(self, statement):
        vcl_statements = vcl.StatementsTuple(statement).vcl_tuple
        vcl_context = statement.result.context.vcl_sub_context
        return self._vcl_template.check_invalid(vcl_statements, vcl_context.current_device)

    def execute(self, statement, force_compilation=False):
        vcl_statements = vcl.StatementsTuple(statement).vcl_tuple
        vcl_context = statement.result.context.vcl_sub_context
        _atd.execute(self._vcl_template, vcl_statements, vcl_context, force_compilation)
        return statement.result


class VectorAxpyTemplate(TemplateBase):

    Parameters = _atd.vector_axpy_template.parameters_type

    def __init__(self, parameters):
        super(VectorAxpyTemplate, self).__init__()
        self._vcl_template = _atd.vector_axpy_template(parameters)


class MatrixAxpyTemplate(TemplateBase):

    Parameters = _atd.matrix_axpy_template.parameters_type

    def __init__(self, parameters):
        super(MatrixAxpyTemplate, self).__init__()
        self._vcl_template = _atd.matrix_axpy_template(parameters)


class ReductionTemplate(TemplateBase):

    Parameters = _atd.reduction_template.parameters_type

    def __init__(self, parameters):
        super(ReductionTemplate, self).__init__()
        self._vcl_template = _atd.reduction_template(parameters)

class RowWiseReductionTemplate(TemplateBase):

    Parameters = _atd.row_wise_reduction_template.parameters_type

    def __init__(self, parameters):
        super(RowWiseReductionTemplate, self).__init__()
        self._vcl_template = _atd.row_wise_reduction_template(parameters)


class MatrixProductTemplate(TemplateBase):

    Parameters = _atd.matrix_product_template.parameters_type

    def __init__(self, parameters, A_trans, B_trans):
        super(MatrixProductTemplate, self).__init__();
        self._A_trans = A_trans
        self._B_trans = B_trans
        self._vcl_template = _atd.matrix_product_template(parameters, A_trans,  B_trans)

    @property
    def A_trans(self):
        return self._A_trans

    @property
    def B_trans(self):
        return self._B_trans
