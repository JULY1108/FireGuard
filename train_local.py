"""Run the author's FireGuard notebook on Windows; data and results stay in outputs/training."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
BASE = ROOT / 'outputs' / 'training'
NOTEBOOK = ROOT / 'FireGuard_Pipeline.ipynb'
STEPS = {6: 'download', 8: 'extract', 10: 'merge_labels', 12: 'deduplicate',
         14: 'group_split', 16: 'audit', 18: 'augmentation', 20: 'smoke_test',
         22: 'baseline', 24: 'ablation', 26: 'train', 28: 'evaluate',
         30: 'confidence_threshold', 32: 'export', 34: 'model_card'}


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def safe_rmtree(path, **kwargs):
    target = Path(path).resolve()
    if target == BASE.resolve() or not target.is_relative_to(BASE.resolve()):
        raise ValueError(f'Refusing cleanup outside generated training directories: {target}')
    shutil.rmtree(target, **kwargs)


def extract_archive(archive, destination):
    import zipfile
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(destination) or (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(f'Unsafe ZIP member: {member.filename}')
        z.extractall(destination)


def replace_once(source, old, new):
    assert source.count(old) == 1, f'Notebook changed; review adaptation: {old[:90]}'
    return source.replace(old, new, 1)


def valid_box(box, allow_outside=False):
    cls, x, y, w, h = box
    basic = cls in (0, 1) and all(math.isfinite(v) for v in (x, y, w, h)) and 0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1
    return basic and (allow_outside or (min(x-w/2, y-h/2) >= -1e-6 and max(x+w/2, y+h/2) <= 1+1e-6))


def clip_box(box):
    assert valid_box(box, allow_outside=True), f'Invalid box: {box}'
    cls, x, y, w, h = box
    x1, y1 = max(0., x-w/2), max(0., y-h/2)
    x2, y2 = min(1., x+w/2), min(1., y+h/2)
    return (cls, (x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1)


def audit_data(directory):
    """Check every prepared image/label pair; fingerprint labels and image sizes."""
    import collections
    import yaml
    directory = Path(directory)
    cfg = yaml.safe_load((directory / 'fire.yaml').read_text(encoding='utf-8'))
    assert cfg['names'] == {0: 'fire', 1: 'smoke'} and cfg['nc'] == 2
    assert Path(cfg['path']).resolve() == directory.resolve()
    digest, stats = hashlib.sha256(), {}
    for split in ('train', 'val', 'test'):
        assert cfg[split] == f'images/{split}'
        images = sorted((directory / 'images' / split).iterdir())
        labels = {p.stem: p for p in (directory / 'labels' / split).glob('*.txt')}
        assert {p.stem for p in images} == labels.keys(), f'Missing/unpaired labels in {split}'
        stats[split] = collections.Counter(img=len(images), **{'if': 0, 'is': 0, 'bf': 0, 'bs': 0, 'neg': 0})
        for ip in images:
            lp = labels[ip.stem]
            raw = lp.read_bytes()
            digest.update(f'{split}/{ip.name}:{ip.stat().st_size}\n'.encode('utf-8') + raw + b'\0')
            boxes = [tuple(map(float, line.split())) for line in raw.decode('utf-8').splitlines() if line.strip()]
            assert all(len(b) == 5 and valid_box(b) for b in boxes), f'Invalid label: {lp}'
            classes = {b[0] for b in boxes}
            stats[split]['if'] += 0 in classes
            stats[split]['is'] += 1 in classes
            stats[split]['neg'] += not boxes
            for b in boxes:
                stats[split]['bf' if b[0] == 0 else 'bs'] += 1
        assert stats[split]['bf'] and stats[split]['bs'], f'Empty class in {split}'
    return {'splits': stats, 'label_sha256': digest.hexdigest()}


def copy_atomic(source, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + '.copying')
    shutil.copy2(source, tmp)
    tmp.replace(destination)


def restore_stage(local, backup, epochs, imgsz, data):
    """Prefer the most advanced valid checkpoint; finalized stages cannot resume."""
    import torch
    local, backup = Path(local), Path(backup)
    choices, failures = [], []
    for directory in (local, backup):
        last = directory / 'weights' / 'last.pt'
        if not last.exists():
            continue
        try:
            checkpoint = torch.load(last, map_location='cpu', weights_only=False)
            cfg = checkpoint['train_args']
            assert cfg['epochs'] == epochs and cfg['imgsz'] == imgsz, 'Stage configuration changed'
            assert Path(cfg['data']).resolve() == Path(data).resolve(), 'Stage dataset changed'
            model = checkpoint.get('ema') or checkpoint.get('model')
            assert model.names == {0: 'fire', 1: 'smoke'}, 'Wrong checkpoint classes'
            epoch = checkpoint.get('epoch', -1)
            history = checkpoint.get('train_results', {}).get('epoch', [])
            completed_epochs = epoch + 1 if epoch >= 0 else (int(history[-1]) if history else 0)
            assert completed_epochs > 0, 'Checkpoint contains no training history'
            finished = epoch == -1 or completed_epochs >= epochs
            assert finished or checkpoint.get('optimizer') is not None, 'Optimizer state missing'
            choices.append((completed_epochs, finished, directory == local, directory))
            del checkpoint, model
        except Exception as error:
            failures.append(f'{last}: {error}')
    if not choices:
        if failures:
            raise RuntimeError('No valid stage checkpoint; refusing to overwrite weights. ' + '; '.join(failures))
        return False, False
    for failure in failures:
        print(f'Ignored invalid checkpoint: {failure}', flush=True)
    _, finished, _, selected = max(choices, key=lambda item: item[:3])
    if selected != local:
        for relative in ('weights/last.pt', 'weights/best.pt', 'results.csv', 'args.yaml'):
            if (selected / relative).exists():
                copy_atomic(selected / relative, local / relative)
    if finished:
        best = local / 'weights' / 'best.pt'
        assert best.is_file(), f'Completed stage has no best.pt: {local}'
        checkpoint = torch.load(best, map_location='cpu', weights_only=False)
        assert (checkpoint.get('ema') or checkpoint.get('model')).names == {0: 'fire', 1: 'smoke'}
        for relative in ('weights/last.pt', 'weights/best.pt', 'results.csv', 'args.yaml'):
            if (local / relative).exists():
                copy_atomic(local / relative, backup / relative)
    return not finished, finished


def local_source(index, cell):
    source = ''.join(cell['source'])
    source = '\n'.join(line for line in source.splitlines() if not line.lstrip().startswith(('!', '%')))
    source = source.replace('/content', BASE.as_posix()).replace('shutil.rmtree(', 'safe_rmtree(')
    if index == 6:
        source = replace_once(source,
            "if os.path.isdir(f'{DF_OUT}/train/images') and os.listdir(f'{DF_OUT}/train/images'):",
            "if os.path.exists(f'{DF_OUT}/.dfire_complete'):")
        source = replace_once(source, "    return f'✅ D-Fire — {n:,}",
            "    Path(f'{DF_OUT}/.dfire_complete').write_text(str(n), encoding='utf-8')\n    return f'✅ D-Fire — {n:,}")
    if index == 8:
        source = replace_once(source, 'if os.path.isdir(out) and os.listdir(out):',
                              "if os.path.exists(f'{out}/.unpacked'):")
        source = replace_once(source, 'with zipfile.ZipFile(src) as z: z.extractall(out)',
            "extract_archive(src, out)\n    Path(f'{out}/.unpacked').write_text('complete', encoding='utf-8')")
        source += "\nDFIRE_DIR = DF_OUT if USE_DFIRE else None\n"
        source += "assert FASDD_DIR and DFIRE_DIR, 'Both author datasets must be present'\n"
    if index == 14:
        source = replace_once(source, "with open(YAML, 'w') as f:", "with open(YAML, 'w', encoding='utf-8') as f:")
    if index == 20:
        source = replace_once(source, "with open(f'{SM}/fire.yaml','w') as f:",
                              "with open(f'{SM}/fire.yaml','w', encoding='utf-8') as f:")
    if index == 12:
        # ponytail: keep the author's O(n²) bucket scan; indexed Hamming search if runtime dominates.
        source = replace_once(source, '                    seen.add(pk)\n', '')
        source = replace_once(source, "                    if d <= DEDUP_DISTANCE:",
            "                    if d <= GROUP_DISTANCE: seen.add(pk)\n                    if d <= DEDUP_DISTANCE:")
    if index == 22:
        source = replace_once(source, "'fire' in n.lower() or 'smoke' in n.lower()", "n.lower() in ('fire', 'smoke')")
        source = source.replace('❌ هیچ‌کدام → دقتش روی این مسئله صفر است', '未包含 fire/smoke；未测量检测 mAP')
    if index == 26:
        source = replace_once(source, "def save_state(): json.dump(state, open(STATE_F,'w'))",
                              'def save_state(): save_json(STATE_F, state)')
        source = replace_once(source, 'threading.Thread(target=work, daemon=True).start()', 'work()')
        start = source.index("    dbest = f'{DRIVE_RUN}/{name}/weights/best.pt'")
        end = source.index("    weights = f'{local}/weights/last.pt' if resume", start)
        source = source[:start] + """    resume, finished = restore_stage(local, f'{DRIVE_RUN}/{name}', eps, imgsz, YAML)
    if finished:
        if name not in state['done']: state['done'].append(name)
        save_state()
        prev_best = f'{local}/weights/best.pt'
        print(f'Completed stage recovered: {name}')
        continue
    assert name not in state['done'], f'Completed stage files are missing/incomplete: {name}'

""" + source[end:]
        source = source.replace('shutil.copy(', 'copy_atomic(')
    if index == 28:
        source = replace_once(source,
            'print(f"\\nمدل خام YOLO26 روی این مسئله: 0.0000  (کلاس fire/smoke نداشت)")',
            "print('COCO 基础模型未包含 fire/smoke；本流程只测量其速度，没有检测 mAP 基线。')")
    if index == 30:
        source = source.replace('/images/test/', '/images/val/').replace('/labels/test/', '/labels/val/')
        source = source.replace('آزمون:', '验证集阈值校准:').replace('تشخیص', 'Positive image trigger rate')
        source = replace_once(source, 'SAMPLE = 2000', 'SAMPLE = max(len(neg), len(pos))\nassert neg and pos')
        source += """
CALIBRATION = dict(split='val', negatives=len(cn), positives=len(cp),
                   recommended_conf=RECOMMENDED_CONF, target_far=TARGET_FAR,
                   far=float((cn >= th).mean()), positive_trigger_rate=float((cp >= th).mean()),
                   target_met=bool(ok), metric='any detection per image; no class/IoU match')
save_json(Path(DRIVE_RUN) / 'threshold_calibration.json', CALIBRATION)
"""
    if index == 34:
        source = replace_once(source, "    'recommended_conf': RECOMMENDED_CONF,",
            "    'recommended_conf': RECOMMENDED_CONF,\n    'threshold_calibration': CALIBRATION,")
        source = replace_once(source,
            "json.dump(card, open(f'{FINAL}/model_card.json','w'), indent=2, ensure_ascii=False)",
            """import csv
card['train']['epochs_planned'] = card['train'].pop('epochs')
card['train']['epochs_completed_per_stage'] = {}
for stage, *_ in stages:
    with open(f'{DRIVE_RUN}/{stage}/results.csv', encoding='utf-8', newline='') as training_csv:
        completed = list(csv.DictReader(training_csv))
    card['train']['epochs_completed_per_stage'][stage] = len(completed)
card['data']['traps_handled'] = ['D-Fire class remapping', 'recognized groups kept within split',
                               'invalid labels isolated', 'boundary boxes clipped', 'within-stage resume']
card['data']['grouping_completeness_guaranteed'] = False
save_json(f'{FINAL}/model_card.json', card)""")
    return source + '\n'


def run(args):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    BASE.mkdir(parents=True, exist_ok=True)
    notebook = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
    fingerprint = hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest()
    sources = {i: local_source(i, notebook['cells'][i]) for i in STEPS}
    for i, source in sources.items():
        compile(source, f'notebook_cell_{i}', 'exec')
    assert 'DFIRE_DIR = DF_OUT' in sources[8]
    assert 'optimizer=\'auto\'' in sources[18]
    assert "('s2_hi', IMGSZ_HI, EPOCHS-E1, None)" in sources[26]
    if args.check:
        import contextlib
        import io
        import re
        results = []
        for code in [''.join(notebook['cells'][12]['source']), sources[12]]:
            code = re.sub(r'def dhash64\(p\):.*?(?=KEY_PARENT =)',
                          'def dhash64(p): return int(p)\n\n', code, flags=re.S)
            hashes = [0, 0, 1, 7, 15, 255, 65535, (1 << 64) - 1]
            test_ns = dict(DEDUP=True, WORKERS=1, DEDUP_DISTANCE=3, GROUP_DISTANCE=12,
                records=[dict(src=h, lab=[(0,.5,.5,.2,.2)] if i % 2 else [],
                              source='fasdd' if i % 2 else 'dfire', group=str(i)) for i,h in enumerate(hashes)])
            with contextlib.redirect_stdout(io.StringIO()):
                exec(code, test_ns)
            results.append((test_ns['records'], test_ns['NEAR'],
                            {k: test_ns['kfind'](k) for k in test_ns['KEY_PARENT']}))
        assert results[0] == results[1], 'Memory adaptation changed author dedup/group results'
        assert valid_box((0, .5, .5, .2, .2))
        assert not valid_box((1, .5, .5, 0, 0))
        assert not valid_box((1, float('nan'), .5, .2, .2))
        print('OK: 15 cells compile; D-Fire/two-stage config, dedup equivalence and invalid-box guards pass.')
        return

    for key, folder in [('HF_HOME', 'hf_cache'), ('YOLO_CONFIG_DIR', 'yolo_config'),
                        ('MPLCONFIGDIR', 'mpl_config'), ('TORCH_HOME', 'torch_cache'),
                        ('PIP_CACHE_DIR', 'pip_cache'), ('TEMP', 'tmp'), ('TMP', 'tmp')]:
        path = BASE / folder
        path.mkdir(exist_ok=True)
        os.environ[key] = str(path)
    os.environ['HF_HUB_DISABLE_XET'] = '1'
    os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '120'
    os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
    os.environ['MPLBACKEND'] = 'Agg'
    os.chdir(BASE)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import torch
    import ultralytics
    from ultralytics.cfg import get_cfg
    assert torch.cuda.is_available(), 'CUDA GPU is required for this full training run'
    assert get_cfg(overrides={'augmentations': []}).augmentations == [], 'Ultralytics lacks the author CCTV augmentation hook'

    ns = dict(__name__='__main__', Path=Path, safe_rmtree=safe_rmtree, extract_archive=extract_archive, save_json=save_json,
              copy_atomic=copy_atomic, restore_stage=restore_stage,
              MODEL=str(BASE / 'models' / 'yolo26s.pt'), RUN='fireguard_s_local_v1',
              EPOCHS=120, PROGRESSIVE=True, IMGSZ_HI=640, IMGSZ_LO=512, STAGE1_FRAC=0.60,
              BATCH=-1, WORKERS=min(8, max(2, (os.cpu_count() or 2) - 1)),
              CCTV_AUG=True, RUN_SMOKETEST=True, RUN_ABLATION=args.ablation,
              USE_FASDD=True, USE_DFIRE=True, DEDUP=True, DEDUP_DISTANCE=3,
              GROUP_DISTANCE=12, SYNC_EVERY=5, DRIVE_DATA=(BASE / 'archives').as_posix(),
              DRIVE_RUNS=(BASE / 'checkpoints').as_posix(), WORK=(BASE / 'work').as_posix(),
              DATA_DIR=(BASE / 'fireguard_data').as_posix(), RUNS_LOCAL=(BASE / 'runs').as_posix())
    ns['DRIVE_RUN'] = ns['DRIVE_RUNS'] + '/' + ns['RUN']
    ns['YAML'] = ns['DATA_DIR'] + '/fire.yaml'
    configuration = ns.copy()
    for key in ['DRIVE_DATA', 'DRIVE_RUNS', 'DRIVE_RUN', 'WORK', 'DATA_DIR', 'RUNS_LOCAL']:
        Path(ns[key]).mkdir(parents=True, exist_ok=True)
    (BASE / 'models').mkdir(exist_ok=True)
    save_json(BASE / 'run_config.json', {k: v for k, v in ns.items() if isinstance(v, (str, int, float, bool))} |
              {'notebook_sha256': fingerprint, 'torch': torch.__version__, 'ultralytics': ultralytics.__version__,
               'adapter_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'gpu': torch.cuda.get_device_name(0), 'training_workers': 0})
    print(f'GPU: {torch.cuda.get_device_name(0)}; torch {torch.__version__}; ultralytics {ultralytics.__version__}', flush=True)
    print('Author configuration: YOLO26s, all layers, 72 epochs @512 + 48 epochs @640; AutoBatch.', flush=True)

    figure_count = 0
    def save_figures():
        nonlocal figure_count
        for number in plt.get_fignums():
            figure_count += 1
            path = BASE / 'reports' / f'figure_{figure_count:02d}.png'
            path.parent.mkdir(exist_ok=True)
            plt.figure(number).savefig(path, dpi=140, bbox_inches='tight')
        plt.close('all')
    plt.show = save_figures

    def step(index):
        phase = STEPS[index]
        save_json(BASE / 'progress.json', {'status': 'running', 'phase': phase, 'pid': os.getpid(), 'time': time.time()})
        print(f'\n=== [{index}] {phase} ===', flush=True)
        source_file = BASE / 'sources' / f'cell_{index:02d}_{phase}.py'
        source_file.parent.mkdir(exist_ok=True)
        source_file.write_text(sources[index], encoding='utf-8')
        exec(compile(sources[index], str(source_file), 'exec'), ns)

    prepared = BASE / 'prepare_summary.json'
    if prepared.exists():
        summary = json.loads(prepared.read_text(encoding='utf-8'))
        assert summary['notebook_sha256'] == fingerprint, 'Notebook changed since data preparation'
        audit = audit_data(ns['DATA_DIR'])
        assert audit['splits'] == summary['splits'], 'Prepared split/class counts changed'
        assert audit['label_sha256'] == summary['label_sha256'], 'Prepared labels/images changed; review data before training'
        print('Reusing completed, audited preparation.', flush=True)
    else:
        cached = all((BASE / p).exists() for p in ['work/dfire/.dfire_complete', 'work/fasdd/.unpacked'])
        required_gib = 15 if cached else 25
        if shutil.disk_usage(BASE).free < required_gib * 1024**3:
            raise RuntimeError(f'Need {required_gib} GiB free for the remaining author data preparation.')
        for index in [6, 8, 10]:
            step(index)
        counts = {name: sum(r['source'] == name for r in ns['records']) for name in ['fasdd', 'dfire']}
        assert counts['fasdd'] > 90000 and counts['dfire'] > 20000, f'Incomplete source datasets: {counts}'
        quarantine = [r for r in ns['records'] if not all(valid_box(b, allow_outside=True) for b in r['lab'])]
        save_json(BASE / 'label_quarantine.json', quarantine)
        ns['records'] = [r for r in ns['records'] if all(valid_box(b, allow_outside=True) for b in r['lab'])]
        corrected = []
        for r in ns['records']:
            for i, box in enumerate(r['lab']):
                if not valid_box(box):
                    fixed = clip_box(box)
                    corrected.append({'source': r['src'], 'box': box, 'fixed': fixed})
                    r['lab'][i] = fixed
        save_json(BASE / 'label_boundary_fixes.json', corrected)
        print(f'Quarantined {len(quarantine)} images with invalid boxes; raw files preserved.', flush=True)
        for index in [12, 14, 16]:
            step(index)
        assert ns['strad'] == 0, 'An event group leaked between splits'
        for split in ['train', 'val', 'test']:
            assert ns['st'][split]['img'] and ns['st'][split]['bf'] and ns['st'][split]['bs'], f'Empty split/class: {split}'
        assert not ns['issues'], f'Label audit failed: {ns["issues"]}'
        audit = audit_data(ns['DATA_DIR'])
        save_json(prepared, {'notebook_sha256': fingerprint, 'source_images': counts,
                  'after_dedup': len(ns['records']), 'splits': ns['st'], 'audit_issues': ns['issues'],
                  'group_leaks': ns['strad'], 'quarantined_images': len(quarantine),
                  'label_sha256': audit['label_sha256'], 'clipped_boxes': len(corrected)})

    # Release the author's candidate hash sets before allocating training batches.
    ns = configuration
    import gc
    gc.collect()
    # Windows notebook-defined transforms are not importable by spawned DataLoader workers.
    # Keep the author's preprocessing thread count above; use the reliable Windows loader below.
    ns['WORKERS'] = 0
    step(18)
    assert ns['CCTV_AUGS'], 'Author CCTV augmentation must not silently fall back'
    from ultralytics import YOLO
    model = YOLO(ns['MODEL'])
    assert len(model.names) == 80, 'Starting weights must be the COCO pretrained model'
    del model
    import albumentations
    expected_smoke = {'notebook_sha256': fingerprint, 'data_label_sha256': audit['label_sha256'],
                      'ultralytics': ultralytics.__version__, 'albumentations': albumentations.__version__}
    smoke_marker = BASE / 'smoke_test_complete.json'
    stamp = json.loads(smoke_marker.read_text(encoding='utf-8')) if smoke_marker.exists() else {}
    smoke_ready = all(stamp.get(k) == v for k, v in expected_smoke.items())
    smoke_run = BASE / 'runs' / '_smoketest'
    smoke_ready = smoke_ready and all((smoke_run / p).is_file() for p in ('weights/best.pt', 'weights/last.pt', 'results.csv'))
    if smoke_ready:
        checked = YOLO(str(smoke_run / 'weights' / 'best.pt'))
        assert checked.names == {0: 'fire', 1: 'smoke'}, 'Saved smoke-test model has incorrect classes'
        del checked
    else:
        if smoke_run.exists():
            history = BASE / 'smoke_history' / time.strftime('%Y%m%d_%H%M%S')
            history.parent.mkdir(exist_ok=True)
            shutil.copytree(smoke_run, history)
            print(f'Previous smoke-test models preserved: {history}', flush=True)
        step(20)
        save_json(smoke_marker, expected_smoke | {'classes': ['fire', 'smoke']})
    step(22)
    step(24)
    if args.phase == 'prepare':
        save_json(BASE / 'progress.json', {'status': 'prepared', 'phase': 'ready_to_train', 'pid': os.getpid(), 'time': time.time()})
        return
    for index in [26, 28, 30, 32, 34]:
        step(index)
    save_json(BASE / 'progress.json', {'status': 'complete', 'phase': 'model_card',
              'model': ns['FINAL'] + '/best.pt', 'pid': os.getpid(), 'time': time.time()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['prepare', 'all'], default='all')
    parser.add_argument('--ablation', action='store_true', help='Enable the author optional 15-epoch A/B experiment')
    parser.add_argument('--check', action='store_true', help='Compile adapted notebook cells without downloads or training')
    arguments = parser.parse_args()
    try:
        run(arguments)
    except Exception as error:
        save_json(BASE / 'progress.json', {'status': 'failed', 'error': str(error), 'pid': os.getpid(), 'time': time.time()})
        traceback.print_exc()
        sys.exit(1)
