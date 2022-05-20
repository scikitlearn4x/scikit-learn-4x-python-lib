import os
import subprocess


class CondaEnvironment:
    def __init__(self):
        self.name = ''
        self.path = ''
        self.python_version = ''
        self.major_python_version = ''
        self.scikit_learn_version = ''

    def __repr__(self):
        return f'{self.name} @ {self.python_version} path: {self.path}'

    def load_version(self):
        output = run_command_for_output(self.path + '/bin/python --version')
        self.python_version = output[0]
        self.python_version = self.python_version[self.python_version.rindex(' '):].strip()

        self.major_python_version = major_minor_version(self.python_version)
        self.scikit_learn_version = self.check_package_version('scikit-learn')

    def get_python(self):
        return self.path + '/bin/python'

    def get_pip(self):
        return self.path + '/bin/pip'

    def install_package(self, package_name, version=None):
        version = '' if version is None else '==' + version
        output = run_command_for_output(self.get_pip() + f' install {package_name}{version}', error_to_kill_on=["Installing build dependencies: finished with status 'error'"])

        return self.check_package_version(package_name)

    def check_package_version(self, package_name):
        all_packages = run_command_for_output(self.get_pip() + ' list')

        if not all_packages:
            return False

        result = None

        for line in all_packages:
            if line.startswith(package_name):
                components = remove_empty_element(line.split(' '))
                result = components[1]

        return result


def ensure_git_all_changes_committed():
    output = run_command_for_output('git status')
    if 'nothing to commit, working tree clean' not in output:
        raise Exception('Please commit all changes before running this script.')


def run_command_for_output(command, error_to_kill_on=[]):
    result = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    lines = []
    for line in result.stdout:
        line = bytes.decode(line).strip()

        for error in error_to_kill_on:
            if error in line:
                result.kill()
                result.wait()
                return False
        lines.append(line)

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


def get_conda_available_python_versions():
    result = []
    output = run_command_for_output('conda search python')[3:]

    for line in output:
        info = remove_empty_element(line.split(' '))
        major_version = major_minor_version(info[1])
        if major_version not in result:
            result.append(major_version)

    return list(result)


def remove_empty_element(array):
    result = []

    for element in array:
        if element != '':
            result.append(element)

    return result


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

