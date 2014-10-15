import abc, logging
from . import _viennacl as _v
from .pycore import Node, Statement

class OrderType(object):
    def __init__(*args):
        raise TypeError("This class is not supposed to be instantiated")

class SequentialOrder(OrderType):
    vcl_order = _v.statements_tuple_order_type.SEQUENTIAL

class IndependentOrder(OrderType):
    vcl_order = _v.statements_tuple_order_type.INDEPENDENT


class StatementsTuple(object):
    vcl_statements_tuple = None

    def __init__(self, statements, order = SequentialOrder):
        if not isinstance(statements, list):
            statements = [statements]
        def to_vcl_statement(s):
            if isinstance(s, Node):
                return Statement(s).vcl_statement
            else:
                return s.vcl_statement
        vcl_statements = list(map(to_vcl_statement, statements))
        self.order = order
        self.vcl_tuple = _v.statements_tuple(vcl_statements, order.vcl_order)

FetchingPolicy = _v.fetching_policy_type

class TemplateBase(object):

    Parameters = _v.template_base.parameters_type

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
        vcl_statement = statement.vcl_statement;
        vcl_context = statement.result.context.vcl_sub_context;
        return vcl_statement.check_template(self._vcl_template, vcl_context);

    def execute(self, statement, force_compilation=False):
        vcl_statement = statement.vcl_statement;
        vcl_context = statement.result.context.vcl_sub_context;
        vcl_statement.execute_template(self._vcl_template, vcl_context, force_compilation);
        return statement.result;


class VectorAxpyTemplate(TemplateBase):

    Parameters = _v.vector_axpy_template.parameters_type

    def __init__(self, parameters):
        super(VectorAxpyTemplate, self).__init__()
        self._vcl_template = _v.vector_axpy_template(parameters)


class MatrixAxpyTemplate(TemplateBase):

    Parameters = _v.matrix_axpy_template.parameters_type

    def __init__(self, parameters):
        super(MatrixAxpyTemplate, self).__init__()
        self._vcl_template = _v.matrix_axpy_template(parameters)


class ReductionTemplate(TemplateBase):

    Parameters = _v.reduction_template.parameters_type

    def __init__(self, parameters):
        super(ReductionTemplate, self).__init__()
        self._vcl_template = _v.reduction_template(parameters)

class RowWiseReductionTemplate(TemplateBase):

    Parameters = _v.row_wise_reduction_template.parameters_type

    def __init__(self, parameters):
        super(RowWiseReductionTemplate, self).__init__()
        self._vcl_template = _v.row_wise_reduction_template(parameters)


class MatrixProductTemplate(TemplateBase):

    Parameters = _v.matrix_product_template.parameters_type

    def __init__(self, parameters, A_trans, B_trans):
        super(MatrixProductTemplate, self).__init__();
        self._A_trans = A_trans
        self._B_trans = B_trans
        self._vcl_template = _v.matrix_product_template(parameters, A_trans,  B_trans)

    @property
    def A_trans(self):
        return self._A_trans

    @property
    def B_trans(self):
        return self._B_trans
