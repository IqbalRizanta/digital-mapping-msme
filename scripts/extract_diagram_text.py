import json
import re
import html
from pathlib import Path

def clean_html(text):
    if not text:
        return ""
    text = html.unescape(text)
    text = re.sub(r'<br\s*/?>', '\n', text)
    text = re.sub(r'</p>', '\n', text)
    text = re.sub(r'<[^>]+>', ' ', text)
    lines = [re.sub(r'\s+', ' ', l).strip() for l in text.split('\n')]
    return '\n'.join([l for l in lines if l])

def extract_cells(cells):
    nodes = []
    edges = []
    layers = []
    for c in cells:
        ctype = c.get("type", "")
        cid = c.get("id", "")
        lbl = c.get("label", "") or c.get("value", "") or ""
        clean_lbl = clean_html(lbl)
        parent = c.get("parent", "")
        src = c.get("source", "")
        tgt = c.get("target", "")
        
        if ctype == "layer":
            layers.append({"id": cid, "label": clean_lbl})
        elif ctype == "edge" or "source" in c or "target" in c:
            edges.append({"id": cid, "label": clean_lbl, "source": src, "target": tgt})
        else:
            nodes.append({"id": cid, "label": clean_lbl, "parent": parent, "type": ctype})
    return layers, nodes, edges

def process_file(file_path):
    out = []
    out.append(f"# Analysis of {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for p in data.get("pages", []):
        p_name = p.get("name")
        p_id = p.get("id")
        cells = p.get("cells", [])
        layers, nodes, edges = extract_cells(cells)
        out.append(f"\n## PAGE: {p_name} (ID: {p_id}) - {len(layers)} layers, {len(nodes)} nodes, {len(edges)} edges\n")
        out.append("### NODES & CONTAINERS:")
        for n in nodes:
            lines = n['label'].split('\n')
            formatted_val = lines[0]
            if len(lines) > 1:
                formatted_val += " | " + " / ".join(lines[1:])
            out.append(f"- [{n['id']}] (parent: {n['parent']}) : {formatted_val}")
        
        out.append("\n### EDGES / CONNECTIONS:")
        for e in edges:
            lbl_str = f" : {e['label']}" if e['label'] else ""
            out.append(f"- ({e['source']} -> {e['target']}){lbl_str}")
    return "\n".join(out)


if __name__ == "__main__":
    report1 = process_file("diagram-architecture/flowchart-mandiri.json")
    with open("reports/flowchart_mandiri_parsed.md", "w", encoding="utf-8") as f:
        f.write(report1)
        
    report2 = process_file("diagram-architecture/H7D.5-3_diagram-architecture-2026 (1).json")
    with open("reports/architecture_kelompok_parsed.md", "w", encoding="utf-8") as f:
        f.write(report2)
        
    print("Parsed reports successfully written to reports/!")
