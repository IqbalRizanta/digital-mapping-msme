import json
import re

def summarize_diagram(path, out_f):
    out_f.write("=" * 60 + "\n")
    out_f.write(f"FILE: {path}\n")
    out_f.write("=" * 60 + "\n")
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for p_idx, page in enumerate(data.get('pages', [])):
        page_name = page.get('name', f'Page-{p_idx}')
        out_f.write(f"\n--- PAGE: {page_name} ---\n")
        for c in page.get('cells', []):
            label = c.get('label') or c.get('value') or ''
            html = c.get('html') or ''
            txt = label if label else html
            if txt:
                clean = re.sub(r'<[^>]+>', ' ', txt)
                clean = ' '.join(clean.split())
                c_id = c.get('id', '')
                c_type = c.get('type', '')
                parent = c.get('parent', '')
                if clean:
                    out_f.write(f"[{c_id}] ({c_type}, parent={parent}) -> {clean}\n")

if __name__ == '__main__':
    with open('diagram_summary.txt', 'w', encoding='utf-8') as out_f:
        summarize_diagram('diagram-architecture/Diagram-Architecture-Kelompok-3.json', out_f)
        summarize_diagram('diagram-architecture/Flowchart-Mandiri.json', out_f)
    print("Done writing diagram_summary.txt")
