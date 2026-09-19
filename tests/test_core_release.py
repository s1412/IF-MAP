"""Offline release checks; no workflow execution or paid requests."""

import ast
import contextlib
import importlib
import io
import json
import os
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

from src.paths import PROJECT_ROOT, display_path, project_path
from src.privacy import public_metadata, redact_text


@contextlib.contextmanager
def outside_project():
    previous = Path.cwd()
    with tempfile.TemporaryDirectory() as directory:
        try:
            os.chdir(directory)
            yield Path(directory)
        finally:
            os.chdir(previous)


class CoreReleaseTests(unittest.TestCase):
    def test_relative_paths_ignore_cwd(self):
        with outside_project():
            self.assertEqual(project_path('configs'), PROJECT_ROOT / 'configs')
            self.assertEqual(project_path('../qwen3-8b').resolve(),
                             PROJECT_ROOT.parent / 'qwen3-8b')
            self.assertEqual(display_path('src/score.py'), 'IF-MAP/src/score.py')

    def test_absolute_overrides_remain_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(project_path(directory), Path(directory))

    def test_workflow_loading_outside_project(self):
        from src.workflow_loader import load_workflow
        with outside_project():
            workflow = load_workflow('configs/workflows/who_when_hc.yaml')
        self.assertEqual(workflow.entry_agent, 'Orchestrator')

    def test_imports_do_not_start_experiments(self):
        with patch.dict(os.environ, {}, clear=True), \
                patch('socket.socket.connect', side_effect=AssertionError('network disabled')):
            entry = importlib.import_module('run')
            importlib.import_module('src.rq2_runner')
            importlib.import_module('src.textgrad_opt')
        self.assertEqual(entry.RUNTIME['diversity_sampling_temperature'], 0.3)
        self.assertEqual(entry.EXPERIMENT['rq2']['datasets'],
                         ['HumanEval', 'MBPP', 'NQ', 'DROP', 'HotpotQA', 'MATH'])

    def test_template_credentials_are_not_loaded(self):
        import run
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / '.env').write_text('OPENROUTER_API_KEY=YOUR_OPENROUTER_API_KEY\n')
            with patch.object(run, 'PROJECT_ROOT', root), patch.dict(os.environ, {}, clear=True):
                run._bootstrap_env()
                self.assertNotIn('OPENROUTER_API_KEY', os.environ)

    def test_exported_key_takes_precedence(self):
        import run
        secret = 'sk-' + 'synthetic' * 5
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / '.env').write_text('OPENROUTER_API_KEY=local-placeholder\n')
            with patch.object(run, 'PROJECT_ROOT', root), \
                    patch.dict(os.environ, {'OPENROUTER_API_KEY': secret}, clear=True):
                run._bootstrap_env()
                self.assertEqual(os.environ['OPENROUTER_API_KEY'], secret)
                self.assertEqual(os.environ['OPENAI_API_KEY'], secret)

    def test_no_implicit_parent_credential_read(self):
        import run
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(run, 'PROJECT_ROOT', root), \
                    patch.dict(os.environ, {}, clear=True), \
                    patch('builtins.open', side_effect=AssertionError('unexpected credential read')):
                run._bootstrap_env()

    def test_error_redaction(self):
        from src.errors import _format
        secret = 'sk-' + 'synthetic' * 5
        message = _format('ERROR', 'test', secret, {'api_key': secret,
                          'path': str(PROJECT_ROOT / 'private.json')})
        self.assertNotIn(secret, message)
        self.assertNotIn(str(PROJECT_ROOT), message)
        self.assertIn('IF-MAP/private.json', message)

    def test_metadata_does_not_retain_secrets(self):
        value = public_metadata({'api_key': 'example', 'nested': {'password': 'example'}})
        self.assertEqual(value['api_key'], '<redacted>')
        self.assertEqual(value['nested']['password'], '<redacted>')

    def test_run_metadata_omits_machine_identity(self):
        from src.logger import init_run
        secret = 'sk-' + 'synthetic' * 5
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            _, run_dir = init_run('release_test', directory, 42, {'api_key': secret})
            meta = json.loads((Path(run_dir) / 'meta.json').read_text())
            self.assertNotIn('machine', meta)
            self.assertNotIn('git_hash', meta)
            self.assertNotIn(secret, (Path(run_dir) / 'config.yaml').read_text())

    def test_manifest_excludes_private_artifacts(self):
        paths = (PROJECT_ROOT / 'core_files.txt').read_text().splitlines()
        self.assertEqual(len(paths), len(set(paths)))
        excluded = {'aflow', 'baselines', 'autogen', 'results', 'cache',
                    'data', 'analysis', 'scripts_bench', '.git'}
        for name in paths:
            path = Path(name)
            self.assertFalse(path.is_absolute(), name)
            self.assertNotIn('..', path.parts, name)
            self.assertNotIn(path.parts[0], excluded, name)
            self.assertNotEqual(name, '.env')
            self.assertTrue((PROJECT_ROOT / path).is_file(), name)
            self.assertFalse((PROJECT_ROOT / path).is_symlink(), name)

    def test_release_text_and_syntax(self):
        for name in (PROJECT_ROOT / 'core_files.txt').read_text().splitlines():
            text = (PROJECT_ROOT / name).read_text()
            self.assertIsNone(re.search(r'[\u3400-\u9fff]', text), name)
            self.assertIsNone(re.search(r'\bsk-(?:or-v\d+-)?[A-Za-z0-9_-]{16,}\b', text), name)
            if name.endswith('.py'):
                ast.parse(text, filename=name)


if __name__ == '__main__':
    unittest.main()
