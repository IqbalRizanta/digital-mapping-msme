"""
Runner script to execute Jupyter notebooks headlessly,
capture text outputs and generated plots into the notebook,
save the outputs to Google Drive (G:\My Drive\Iqbal-*) and local reports/,
and update the executed .ipynb file.
"""

import sys
import os
import json
import io
import base64
import shutil
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from IPython.core.interactiveshell import InteractiveShell

def execute_notebook(notebook_path: Path):
    print(f"=== Menjalankan Notebook: {notebook_path.name} ===")
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    shell = InteractiveShell.instance()
    
    # Track execution count
    exec_count = 1
    
    # Store figures captured during cell execution
    captured_figs = []
    
    # Monkey patch plt.show to capture figure for notebook embedding
    orig_show = plt.show
    def custom_show(*args, **kwargs):
        fig = plt.gcf()
        if fig and fig.get_axes():
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            img_b64 = base64.b64encode(buf.read()).decode('utf-8')
            captured_figs.append(img_b64)
        orig_show(*args, **kwargs)
        plt.close('all')

    plt.show = custom_show

    # Make display available in shell namespace
    def custom_display(*args):
        for arg in args:
            if hasattr(arg, 'to_string'):
                print(arg.to_string())
            else:
                print(str(arg))

    shell.user_ns['display'] = custom_display

    cells = nb.get('cells', [])
    for idx, cell in enumerate(cells):
        if cell.get('cell_type') != 'code':
            continue
        
        code = "".join(cell.get('source', []))
        if not code.strip():
            continue

        print(f"-> Menjalankan Cell [{exec_count}] (Index {idx})...")
        cell['execution_count'] = exec_count
        captured_figs.clear()

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()

        with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
            result = shell.run_cell(code)

        stdout_text = stdout_buf.getvalue()
        stderr_text = stderr_buf.getvalue()

        outputs = []
        if stdout_text:
            print(f"   [stdout]: {stdout_text[:120].strip()}...")
            outputs.append({
                "output_type": "stream",
                "name": "stdout",
                "text": stdout_text.splitlines(keepends=True)
            })

        if stderr_text:
            print(f"   [stderr]: {stderr_text[:120].strip()}...")
            outputs.append({
                "output_type": "stream",
                "name": "stderr",
                "text": stderr_text.splitlines(keepends=True)
            })

        for img_b64 in captured_figs:
            outputs.append({
                "output_type": "display_data",
                "data": {
                    "image/png": img_b64,
                    "text/plain": ["<Figure size ...>"]
                },
                "metadata": {}
            })

        if not result.success:
            err_msg = str(result.error_in_exec) if result.error_in_exec else "Execution error"
            print(f"   [ERROR]: {err_msg}")
            outputs.append({
                "output_type": "error",
                "ename": type(result.error_in_exec).__name__ if result.error_in_exec else "Error",
                "evalue": str(result.error_in_exec) if result.error_in_exec else "Unknown",
                "traceback": [err_msg]
            })

        cell['outputs'] = outputs
        exec_count += 1

        if not result.success:
            print(f"-> [GAGAL] Penghentian eksekusi pada Cell {idx} karena error.")
            break

    # Save executed notebook locally
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"-> [SUKSES] Notebook berhasil dieksekusi dan disimpan: {notebook_path}")

    # Copy executed notebook to structured Google Drive (BDA-2026-Iqbal)
    from model.storage_config import (
        GDRIVE_MODIFIKASI_DIR,
        GDRIVE_PROJECT_ROOT,
        GDRIVE_REPEAT_DIR,
        GDRIVE_REPORTS_BDA_DIR,
        GDRIVE_REPORTS_EDA_DIR,
        GDRIVE_REPORTS_EVAL_DIR,
    )

    gdrive_report_dir = None
    if "eda" in notebook_path.name.lower():
        gdrive_report_dir = GDRIVE_REPORTS_EDA_DIR
    elif "model" in notebook_path.name.lower():
        gdrive_report_dir = GDRIVE_REPORTS_EVAL_DIR
    elif "bda" in notebook_path.name.lower():
        gdrive_report_dir = GDRIVE_REPORTS_BDA_DIR

    if GDRIVE_PROJECT_ROOT.exists():
        dest_dir = GDRIVE_REPEAT_DIR if "repeat" in notebook_path.name.lower() else GDRIVE_MODIFIKASI_DIR
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_nb = dest_dir / notebook_path.name
        shutil.copy2(notebook_path, dest_nb)
        print(f"-> [TERSIMPAN DI GDRIVE] Salinan notebook ter-upload ke: {dest_nb}")

    # Also mirror files from GDrive report folder to local reports/figures for local redundancy
    local_reports = Path("docs/reports/figures")
    local_reports.mkdir(parents=True, exist_ok=True)
    if gdrive_report_dir and gdrive_report_dir.exists():
        for png_file in gdrive_report_dir.glob("*.png"):
            shutil.copy2(png_file, local_reports / png_file.name)
        print(f"-> [MIRROR LOKAL] Seluruh grafik tersinkronisasi ke folder lokal {local_reports}")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        target_path = Path(sys.argv[1])
        execute_notebook(target_path)
    else:
        execute_notebook(Path("model/notebooks/eda_3_09040624091.ipynb"))
