import os
import shutil
import subprocess

CONDA_ENV_FOR_LIB = 'develop_xklearn_lib'
CONDA_ENV_FOR_LIB_VERSION = '3.8'


class CondaEnvironment:
    def __init__(self):
        self.name = ''
        self.path = ''
        self.python_version = ''

    def __repr__(self):
        return f'{self.name} @ {self.python_version} path: {self.path}'

    def load_version(self):
        output = run_command_for_output(self.path + '/bin/python --version')
        self.python_version = output[0]
        self.python_version = self.python_version[self.python_version.rindex(' '):].strip()

    def get_python(self):
        return self.path + '/bin/python'

    def get_pip(self):
        return self.path + '/bin/pip'


def main():
    ensure_script_is_called_from_root()
    delete_all_macos_ds_store_files()
    delete_build_folders()
    lib_version = load_library_version()
    environments = create_conda_environments()

    all_passed = True
    python_errors = []
    console_print('Testing library on all available Python versions.')
    for environment in environments:
        passed, version = test_library_in_environment(environment)
        if not passed:
            all_passed = False
            python_errors.append(version)

    if all_passed:
        build_library_for_pypi()
        ensure_archive_has_source_only_folders(lib_version)
        deploy_archives_to_pypi()

        # delete_build_folders()
    else:
        console_print('The following environment(s) failed unit tests:')
        for python_version in python_errors:
            console_print('    * ' + python_version)
        exit(1)


def delete_build_folders():
    console_print('Deleting build files & cleaning up source folder.')
    folders_to_delete = ['build', 'dist', 'sklearn4x.egg-info']

    for folder in folders_to_delete:
        if os.path.exists('./' + folder):
            console_print(f'    * ./{folder}')
            shutil.rmtree('./' + folder)


def run_command_for_output(command):
    result = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    lines = []
    for line in result.stdout:
        lines.append(bytes.decode(line).strip())

    return lines


def get_conda_version():
    console_lines = run_command_for_output('conda --version')
    if len(console_lines) == 0:
        raise Exception('Looks like conda is not installed on this machine. Please install Conda first.')
    else:
        version = console_lines[0]
        version = version[version.index(' ') + 1:]

    return version


def get_conda_path():
    path = run_command_for_output('conda config --show | grep root_prefix')
    path = path[0]
    path = path[path.index(' ') + 1:] + '/'

    return path


def extract_environment_name_and_path(line):
    line = line.replace('*', '')
    components = line.strip().split(' ')
    return components[0], components[-1]


def get_conda_environments():
    result = {}

    output = run_command_for_output('conda env list')[2:-1]
    for line in output:
        name, path = extract_environment_name_and_path(line)
        environment = CondaEnvironment()
        environment.name = name
        environment.path = path
        environment.load_version()

        result[name] = environment

    return result


def major_minor_version(version):
    result = []
    dot_seen = 0

    for ch in version:
        if ch == '.':
            dot_seen += 1
            if dot_seen == 2:
                break

        result.append(ch)

    return ''.join(result)


def create_conda_environment(name, python_version):
    environments = get_conda_environments()

    if name in environments.keys():
        # environment already exits
        if major_minor_version(python_version) != major_minor_version(environments[name].python_version):
            raise Exception(
                f'The environment "{name}" already exists with a different python version. Requested {python_version} but found {environments[name].python_version}.')
    else:
        command = f'conda create --name {name} python={python_version} -y'
        output = run_command_for_output(command)
        success = False
        for line in output:
            if f'conda activate {name}' in line:
                success = True
                break

        if not success:
            raise Exception(f'The command "{command}" failed to create a new environment')

        environments = get_conda_environments()

    return environments[name]


def remove_empty_element(array):
    result = []

    for element in array:
        if element != '':
            result.append(element)

    return result


def get_conda_available_python_versions():
    result = []
    output = run_command_for_output('conda search python')[3:]

    for line in output:
        info = remove_empty_element(line.split(' '))
        major_version = major_minor_version(info[1])
        if major_version not in result:
            result.append(major_version)

    return list(result)


def pretty_format_list(items):
    result = []
    sep = ''

    for i, item in enumerate(items):
        result.append(sep)
        result.append(item)

        if i == len(items) - 2:
            sep = ' and '
        else:
            sep = ', '

    return ''.join(result)


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


def install_packages(environment_info):
    with open('setup.cfg', 'r') as handle:
        lines = handle.readlines()

    index = lines.index('install_requires =\n')
    lines = lines[index + 1:]
    packages = []
    for line in lines:
        if line.startswith('['):
            break
        package = line.strip()
        if package != '':
            packages.append(package)

    for package in packages:
        # print('        - Installing ' + package)
        run_command_for_output(environment_info.get_pip() + ' install ' + package)


def test_library_in_environment(environment):
    all_tests_passed = True
    environment_info = get_conda_environments()[environment]
    console_print(f'    * Testing library on Python {environment_info.python_version}')
    console_print('        - Preparing conda environment')
    install_packages(environment_info)
    console_print('        - Running unit tests')
    unit_tests = run_command_for_output(environment_info.get_python() + ' tests/run_tests.py')

    if 'All Tests Passed!' not in unit_tests:
        all_tests_passed = False
    else:
        passed = unit_tests[1]
        if passed.endswith(' 0'):
            console_print('        - No test found at all')
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
    extract_path = f'./dist/sklearn4x-{lib_version}/'
    if not os.path.exists(archive_path):
        raise Exception('Unable to find the distribution archive to upload to pypi.org')

    run_command_for_output(f'tar -xf {archive_path} -C {extract_path}')


def deploy_archives_to_pypi():
    pass


def find_all_ds_store_files(folder, ds_stores):
    contents = os.listdir(folder)
    for content in contents:
        path = folder + content
        if os.path.isdir(path):
            find_all_ds_store_files(path + '/', ds_stores)
        else:
            if path.endswith('.DS_Store'):
                ds_stores.append(path)


def delete_all_macos_ds_store_files():
    ds_stores = []
    find_all_ds_store_files('./', ds_stores)

    for file in ds_stores:
        os.remove(file)


def load_library_version():
    with open('setup.cfg', 'r') as handle:
        lines = handle.readlines()

    version = lines[2].strip()
    if not version.startswith('version = '):
        raise Exception('Unable to read version from setup.cfg')

    version = version[version.rindex(' '):].strip()
    return version


def console_print(line):
    print(line, flush=True)


def ensure_script_is_called_from_root():
    folder_content = os.listdir('./')
    is_in_root_folder = False

    if '.git' in folder_content:
        is_in_root_folder = True

    if is_in_root_folder:
        console_print("The script is running from the library's root folder.")
    else:
        raise Exception('The script should be invoked with working directory as root folder.')


if __name__ == '__main__':
    main()
