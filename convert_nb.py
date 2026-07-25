import json

with open('final.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

with open('final.py', 'w', encoding='utf-8') as f:
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            f.write(''.join(cell['source']) + '\n\n')
