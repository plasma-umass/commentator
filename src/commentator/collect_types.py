import ast_comments as ast

class TypeCollector(ast.NodeVisitor):
    def __init__(self):
        self.types = set()

    def visit_FunctionDef(self, node):
        self._collect_types_from_annotations(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        self._collect_types_from_annotations(node)
        self.generic_visit(node)

    def visit_AnnAssign(self, node):
        self.types.add(ast.unparse(node.annotation).strip())
        self.generic_visit(node)

    #def visit_arguments(self, node):
    #    # Remove the type annotations from the function arguments
    #    for arg in node.args:
    #        self._collect_types_from_annotations(arg)
    #    if node.vararg:
    #        self._collect_types_from_annotations(node.vararg)
    #    if node.kwarg:
    #        self._collect_types_from_annotations(node.kwarg)
    #    self.generic_visit(node)

    def _collect_types_from_annotations(self, node):
        for arg in node.args.args:
            if arg.annotation:
                self.types.add(ast.unparse(arg.annotation).strip())

        if node.returns:
            self.types.add(ast.unparse(node.returns).strip())

    def get_types(self):
        return list(self.types)

def collect_types(node):
    tc = TypeCollector()
    tc.visit(node)
    return tc.get_types()
