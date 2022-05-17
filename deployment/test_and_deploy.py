import sys
import shutil
from time import sleep
from deploy_utils import *

CONDA_ENV_FOR_LIB = 'develop_xklearn_lib'
CONDA_ENV_FOR_LIB_VERSION = '3.8'


def main():
    args = sys.argv

    if '--test-only' in args:
        test_library_on_all_python_versions()
    else:
        full_deployment_script()


def test_library_on_all_python_versions():
    environments = create_conda_environments()

    console_print('Testing library on all available Python versions.')
    for environment in environments:
        test_library_in_environment(environment, True)


def full_deployment_script():
    ensure_script_is_called_from_root()
    ensure_pypi_credential_exists()
    ensure_git_all_changes_committed()
    delete_all_macos_ds_store_files()
    delete_build_folders()
    lib_version = load_library_version()
    environments = create_conda_environments()

    all_passed = True
    python_errors = []
    console_print('Testing library on all available Python versions.')
    for environment in environments:
        passed, version = test_library_in_environment(environment, False)
        if not passed:
            all_passed = False
            python_errors.append(version)

    if all_passed:
        build_library_for_pypi()
        ensure_archive_has_source_only_folders(lib_version)
        deploy_archives_to_pypi()

        delete_build_folders()
    else:
        console_print('The following environment(s) failed unit tests:')
        for python_version in python_errors:
            console_print('    * ' + python_version)
        exit(1)


def ensure_pypi_credential_exists():
    console_print('Checking ~/.pypirc')
    path = os.path.expanduser('~/.pypirc')
    if not os.path.exists(path):
        # Reference to make the file:
        # https://zestreleaser.readthedocs.io/en/latest/uploading.html
        raise Exception('The ~/.pypirc is missing. Please make a file according to https://zestreleaser.readthedocs.io/en/latest/uploading.html')


def delete_build_folders():
    console_print('Deleting build files & cleaning up source folder.')
    folders_to_delete = ['build', 'dist', 'sklearn4x.egg-info']

    for folder in folders_to_delete:
        if os.path.exists('./' + folder):
            console_print(f'    * ./{folder}')
            shutil.rmtree('./' + folder)


def check_if_conda_is_up_to_date(conda_version):
    output = run_command_for_output('conda search conda')[-1]

    info = remove_empty_element(output.split(' '))
    version = info[1]

    if version != conda_version:
        raise Exception('You are not using the latest version of conda, please update it.')


def create_conda_environments():
    console_print('Evaluating conda environment')
    conda_version = get_conda_version()
    console_print('    * Conda version: ' + conda_version)
    check_if_conda_is_up_to_date(conda_version)
    console_print('    * Conda installation path: ' + get_conda_path())
    console_print(f'    * Development conda environment: {create_conda_environment(CONDA_ENV_FOR_LIB, CONDA_ENV_FOR_LIB_VERSION)}')
    available_pythons = get_conda_available_python_versions()
    console_print(f'    * All available python versions: {pretty_format_list(available_pythons)}')
    console_print('    * Preparing all testing environments')

    environments = get_conda_environments()
    test_environments = []
    for version in available_pythons:
        env = f'test_xklearn_lib_on_python_{version.replace(".", "_")}'
        test_environments.append(env)
        if env in environments.keys():
            env_info = environments[env]
            if version == major_minor_version(env_info.python_version):
                console_print(f'        - {env} -> Already exists')
            else:
                raise Exception(f'The environment "{env}" already exists, but the python version is incorrect. Requested: {version}, Found: {env_info.python_version}.')
        else:
            create_conda_environment(env, version)
            console_print(f'        - {env} -> Created')

    return test_environments


def test_library_in_environment(environment, throw_exception):
    all_tests_passed = True
    environment_info = get_conda_environments()[environment]
    console_print(f'    * Testing library on Python {environment_info.python_version}')
    console_print('        - Preparing conda environment')
    install_packages(environment_info)
    console_print('        - Running unit tests')
    unit_tests = run_command_for_output(environment_info.get_python() + ' tests/run_tests.py')

    if 'All Tests Passed!' not in unit_tests:
        if throw_exception:
            raise Exception(f'The unit tests failed on python {environment_info.python_version} [{environment_info.name}]')
        all_tests_passed = False
    else:
        passed = unit_tests[1]
        if passed.endswith(' 0'):
            console_print('        - No test found at all')
            if throw_exception:
                raise Exception(f'No unit tests found on python {environment_info.python_version} [{environment_info.name}]')
            all_tests_passed = False

    integration_root = './tests/integration_tests/'
    integration_tests = []
    list_all_integration_tests(integration_root, integration_tests)
    if len(integration_tests) == 0:
        all_tests_passed = False
    console_print(f'        - Running {len(integration_tests)} integration test scripts')
    for integration_test in integration_tests:
        script = subprocess.run([environment_info.get_python(), integration_test], stdout=subprocess.PIPE)
        if script.returncode != 0:
            if throw_exception:
                raise Exception(f'The integration test "{integration_test}" failed on python {environment_info.python_version} [{environment_info.name}]')
            all_tests_passed = False

    return all_tests_passed, major_minor_version(environment_info.python_version)


def list_all_integration_tests(folder, integration_tests):
    integrations = os.listdir(folder)
    for file in integrations:
        path = folder + file

        if os.path.isdir(path):
            list_all_integration_tests(path + '/', integration_tests)
        else:
            integration_tests.append(path)


def build_library_for_pypi():
    console_print('Building library wheel for pypi.org')
    run_command_for_output('python setup.py sdist bdist_wheel')


def ensure_archive_has_source_only_folders(lib_version):
    console_print('Validating the distribution archive')
    archive_path = f'./dist/sklearn4x-{lib_version}.tar.gz'
    extract_path = f'dist/'
    if not os.path.exists(archive_path):
        raise Exception('Unable to find the distribution archive to upload to pypi.org')

    archive_path = archive_path[2:]
    run_command_for_output(f'tar -xf {archive_path} -C {extract_path}')

    extract_path = f'./dist/sklearn4x-{lib_version}'
    included_files = set(os.listdir(extract_path))
    expected_files = {'setup.py', 'sklearn4x', 'PKG-INFO', 'LICENSE', 'setup.cfg', 'sklearn4x.egg-info', 'README.md'}

    if len(included_files) != len(expected_files):
        raise Exception('The final archive contains files that are for development only.')

    additional = included_files - expected_files
    if len(additional) != 0:
        raise Exception('The final archive contains files that are for development only.')

    shutil.rmtree(extract_path)


def deploy_archives_to_pypi():
    console_print('Deploying the library to pypi.org')
    output = run_command_for_output('twine upload dist/*')

    for i, line in enumerate(output):
        if 'ERROR' in line:
            console_print(line)
            sleep(2)
            raise Exception('An error happened during deploying to pypi.org')
        elif 'View at:' in line:
            console_print('   * Successfully deployed to: ' + output[i + 1])


def load_library_version():
    with open('setup.cfg', 'r') as handle:
        lines = handle.readlines()

    version = lines[2].strip()
    if not version.startswith('version = '):
        raise Exception('Unable to read version from setup.cfg')

    version = version[version.rindex(' '):].strip()
    return version


if __name__ == '__main__':
    main()
