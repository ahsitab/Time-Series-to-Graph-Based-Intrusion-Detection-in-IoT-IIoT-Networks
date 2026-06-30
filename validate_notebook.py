import json, ast, sys

nb = json.load(open('aiml505_research_notebook.ipynb', encoding='utf-8'))
errors = []
code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']

for i, cell in enumerate(code_cells):
    src = ''.join(cell['source'])
    try:
        ast.parse(src)
    except SyntaxError as e:
        errors.append(f'Code cell {i}: {e}')

if errors:
    for e in errors:
        print('ERROR:', e)
    sys.exit(1)
else:
    print(f'All {len(code_cells)} code cells parsed successfully — no syntax errors.')
    print(f'Total cells: {len(nb["cells"])}  (MD: {sum(1 for c in nb["cells"] if c["cell_type"]=="markdown")}, Code: {len(code_cells)})')
    print('Notebook JSON: VALID')
