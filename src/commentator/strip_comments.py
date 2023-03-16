import ast_comments as ast

class StripComments(ast.NodeTransformer):
    def visit_Comment(self, node):
        return None
    def visit_FunctionDef(self, node):
        return self.process_function(node)
    def visit_AsyncFunctionDef(self, node):
        return self.process_function(node)
    def visit_ClassDef(self, node):
        return self.process_function(node)
        
    def process_function(self, node):
        # Remove the docstring if it exists
        if isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Str):
            node.body.pop(0)
        self.generic_visit(node)
        return node
        
def strip_comments(node):
    sc = StripComments()
    node = sc.visit(node)
    return ast.unparse(ast.fix_missing_locations(node))

