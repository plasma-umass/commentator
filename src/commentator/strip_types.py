import ast_comments as ast

class TypeStripper(ast.NodeTransformer):
    
    def visit_Assign(self, node):
        node.type_comment = None
        return node
    
    def visit_Name(self, node):
        node.annotation = None
        return node

    def visit_For(self, node):
        node.type_comment = None
        return node

    def visit_With(self, node):
        node.type_comment = None
        return node

    def visit_AsyncFunctionDef(self, node):
        return self.process_function(node)
    
    def visit_FunctionDef(self, node):
        return self.process_function(node)

    def visit_arguments(self, node):
        # Remove the type annotations from the function arguments
        for arg in node.args:
            arg.annotation = None
        if node.vararg:
            node.vararg.annotation = None
        if node.kwarg:
            node.kwarg.annotation = None
        return node
    
    def process_function(self, node):
        node.returns = None
        node.type_comment = None
        self.generic_visit(node)
        return node
    
    def visit_arg(self, node):
        node.type_comment = None
        return node
    
    def visit_AnnAssign(self, node):
        node.annotation = None
        return node

def strip_types(node):
    return ast.fix_missing_locations(TypeStripper().visit(node))
