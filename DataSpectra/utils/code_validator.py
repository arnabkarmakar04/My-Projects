import ast

class CodeValidator(ast.NodeVisitor):
    def __init__(self, allowed_modules=None, allowed_functions=None, allowed_attributes=None):
        self.allowed_modules = set(allowed_modules) if allowed_modules else set()
        self.allowed_functions = set(allowed_functions) if allowed_functions else set()
        self.allowed_attributes = set(allowed_attributes) if allowed_attributes else set()
        self.is_safe = True
        self.error_message = ""

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id not in self.allowed_functions:
            self.is_safe = False
            self.error_message = f"Error: Function '{node.func.id}' is not allowed."
            return
        self.generic_visit(node)

    def visit_Attribute(self, node):
        if node.attr not in self.allowed_attributes:
            self.is_safe = False
            self.error_message = f"Error: Attribute '{node.attr}' is not allowed."
            return
        self.generic_visit(node)

    def visit_Name(self, node):
        # Added 'result' to the list of allowed names.
        if node.id not in self.allowed_modules and node.id not in ['df', 'pd', 'px', 'np', 'result']:
            self.is_safe = False
            self.error_message = f"Error: The use of the name '{node.id}' is not allowed."
        self.generic_visit(node)

    @staticmethod
    def validate(code):
        try:
            # Changed mode from 'eval' to 'exec' to handle multi-line code
            tree = ast.parse(code, mode='exec')
            
            allowed_modules = {'pd', 'px', 'np'}
            allowed_functions = {
                'print', 'list', 'dict', 'set', 'tuple', 'str', 'int', 'float', 'len', 'range', 'isinstance',
                'abs', 'round', 'sum', 'min', 'max', 'sorted'
            }
            allowed_attributes = {
                # Data exploration and properties
                'mean', 'sum', 'median', 'min', 'max', 'std', 'var', 'count', 'size', 'shape', 'columns', 'index', 'values',
                'head', 'tail', 'info', 'describe', 'value_counts', 'sort_values', 'sort_index', 'reset_index', 'set_index',
                'groupby', 'agg', 'apply', 'loc', 'iloc', 'corr', 'nunique', 'unique', 'duplicated', 'drop_duplicates',
                'dtypes', 'select_dtypes', 'memory_usage', 'nsmallest', 'nlargest', 'abs', 'round', 'clip', 'isin',
                'pivot_table', 'melt', 'pct_change', 'diff', 'rank', 'cumsum', 'cumprod', 'cummin', 'cummax',
                
                # Data cleaning
                'isna', 'isnull', 'notna', 'notnull', 'dropna', 'fillna', 'astype', 'copy', 'rename', 'drop', 
                'merge', 'join', 'concat', 'equals', 'assign', 'mode', 'replace', 'interpolate', 'bfill', 'ffill',
                'str', 'lower', 'upper', 'strip', 'lstrip', 'rstrip', 'contains', 'startswith', 'endswith',
                'dt', 'year', 'month', 'day', 'hour', 'minute', 'second', 'dayofweek', 'day_name', 'month_name',
                
                # Plotting
                'bar', 'scatter', 'line', 'histogram', 'box', 'pie', 'imshow', 'show', 'sunburst', 'heatmap',
                'area', 'violin', 'hexbin', 'kdeplot', 'pairplot'
            }

            validator = CodeValidator(allowed_modules, allowed_functions, allowed_attributes)
            validator.visit(tree)
            return validator.is_safe, validator.error_message
        except (SyntaxError, TypeError) as e:
            return False, f"Error parsing code: {e}"
