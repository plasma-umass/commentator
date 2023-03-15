import ast_comments as ast

class StripImports(ast.NodeTransformer):
    def visit_Import(self, node):
        return None

    def visit_ImportFrom(self, node):
        return None
    
def strip_imports(node):
    sc = StripImports()
    node = sc.visit(node)
    return ast.unparse(ast.fix_missing_locations(node))

