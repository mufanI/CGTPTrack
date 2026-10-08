import ast


def test_local_path_generation_handles_windows_paths_and_quotes(tmp_path, monkeypatch):
    from lib.train.admin import environment as train_env
    from lib.test.evaluation import environment as test_env
    for i, module in enumerate((train_env, test_env)):
        folder = tmp_path / str(i)
        folder.mkdir()
        monkeypatch.setattr(module, '__file__', str(folder / 'environment.py'))
        workspace = "C:\\Users\\研究者's folder\\CGTPTrack"
        data = 'C:\\data'
        if i == 0:
            module.create_default_local_file_ITP_train(workspace, data)
        else:
            module.create_default_local_file_ITP_test(workspace, data, 'C:\\output')
        text = (folder / 'local.py').read_text(encoding='utf-8')
        ast.parse(text)
        namespace = {}
        exec(compile(text, str(folder / 'local.py'), 'exec'), namespace)
        settings = namespace['EnvironmentSettings']() if i == 0 else namespace['local_env_settings']()
        assert getattr(settings, 'workspace_dir' if i == 0 else 'prj_dir') == workspace
