import platform
import shutil
import sys

from deploy_utils import *

CONDA_ENV_FOR_MAKING_UNIT_TESTS = 'unit_test_sklearn4x_{python_version}_sk_{sklearn_version}'


def load_classifier_and_configurations():
    result = []

    result.append(add_gaussian_nb())

    return result


def add_gaussian_nb():
    config = get_classifier_config(friendly_name='Gaussian Naive Bayes',
                                   class_name='GaussianNB',
                                   namespace='sklearn.naive_bayes')

    # https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html
    add_classifier_configuration(config=config, name='simplest base case without customization')
    add_classifier_configuration(config=config, name='with explicit prior')
    add_classifier_configuration(config=config, name='with explicit var smoothing')
    add_classifier_configuration(config=config, name='with explicit prior and var smoothing')

    return config


def add_classifier_configuration(config, name, **parameters):
    if name.startswith('test') or name.endswith('test'):
        raise Exception('The test is the case name wil be added by the code.')

    config['configurations'].append({'config_name': name, **parameters})


def get_classifier_config(friendly_name, class_name, namespace):
    return {
        'friendly_name': friendly_name,
        'class_name': class_name,
        'namespace': namespace,
        'configurations': []
    }


def get_all_available_sklearn_versions():
    console_print('Loading all available scikit-learn versions.')
    result = {}
    console = run_command_for_output('conda search scikit-learn')
    console = console[2:]

    versions_list = []
    for line in console:
        components = remove_empty_element(line.split(' '))
        sklearn_version = components[1]
        versions_list.append(sklearn_version)

        if sklearn_version not in result.keys():
            result[sklearn_version] = set()

        python_version = components[2][2:]
        python_version = python_version[0: python_version.index('h')]
        python_version = python_version[0] + '.' + python_version[1:]
        assert python_version.startswith('3')

        result[sklearn_version].add(python_version)

    for sklearn_version in result.keys():
        versions = list(result[sklearn_version])
        versions = sorted(versions, key=lambda version: int(version[2:]))
        result[sklearn_version] = versions

    for version in result.keys():
        console_print(f'    * Version {version} for Python {pretty_format_list(result[version])}')
    console_print(f'    * Total versions: {len(result)} [{versions_list[0]} to {versions_list[-1]}]')

    return result


def prepare_conda_environment(environments, scikit_learn_version, python_version):
    env_name = get_test_environment_name(python_version, scikit_learn_version)

    if env_name not in environments.keys():
        create_conda_environment(env_name, python_version)
        environments[env_name] = get_conda_environments()[env_name]

    environment = environments[env_name]
    installed_version = environment.install_package('scikit-learn', scikit_learn_version)
    return installed_version is not None


def get_test_environment_name(python_version, scikit_learn_version):
    env_name = CONDA_ENV_FOR_MAKING_UNIT_TESTS \
        .replace('{python_version}', python_version) \
        .replace('{sklearn_version}', scikit_learn_version) \
        .replace('.', '_')
    return env_name


def prepare_conda_environment_based_on_sklearn_versions(sklearn_versions):
    all_environments = get_conda_environments()
    properly_setup_environments = []
    failed_environments = []

    console_print('Preparing conda environments to generate unit tests.')
    for scikit_learn_version in sklearn_versions.keys():
        console_print('    * scikit-learn ' + scikit_learn_version)
        for i, python_version in enumerate(sklearn_versions[scikit_learn_version]):
            console_print(f'        - [{i + 1} of {len(sklearn_versions[scikit_learn_version])}] Python {python_version}')
            if prepare_conda_environment(all_environments, scikit_learn_version, python_version):
                properly_setup_environments.append(all_environments[get_test_environment_name(python_version, scikit_learn_version)])
            else:
                failed_environments.append(get_test_environment_name(python_version, scikit_learn_version))

    console_print('')
    console_print(f'    * Properly setup environments: {len(properly_setup_environments)}')
    console_print(f'    * Failed to setup environments: {len(failed_environments)}')
    for failed_env in failed_environments:
        console_print(f'        - {failed_env}')

    return properly_setup_environments, failed_environments


def remove_unsupported_versions(sklearn_versions):
    u = platform.uname()
    if u.system == 'Darwin' and u.machine == 'arm64':
        # On Apple Silicon
        to_remove = []
        for version in sklearn_versions:
            if version.startswith('0'):
                to_remove.append(version)

        for version in to_remove:
            sklearn_versions.pop(version)


def get_argument(arg):
    args = sys.argv

    for i, value in enumerate(args):
        if value == arg:
            return args[i + 1]

    raise Exception(f'Argument {arg} should be provided.')


def check_folder_empty(path):
    if os.path.exists(path):
        if len(os.listdir(path)) > 0:
            shutil.rmtree(path)
            os.mkdir(path)
    else:
        os.mkdir(path)


def create_folder(path):
    os.mkdir(path)
    return path


def build_version_subfolders(folder, properly_setup_environments):
    for env in properly_setup_environments:
        path = folder + env.scikit_learn_version + '/' + env.major_python_version + '/'
        if not os.path.exists(path):
            os.makedirs(path)


def generate_python_code(classifiers_info, config):
    pass


def create_binary_and_test_files(classifiers_info, config, environments):
    python_code = generate_python_code(classifiers_info, config)


def iterate_test_cases_and_create_test_files(properly_setup_environments):
    path = get_argument('--save-path')
    if not path.endswith('/'): path += '/'
    console_print('Start creating test cases.')
    console_print(f'    * Path: {path}')
    check_folder_empty(path)

    scripts = create_folder(path + 'scripts/')
    unit_tests = create_folder(path + 'unit_tests/')
    binaries = create_folder(path + 'binaries/')

    build_version_subfolders(scripts, properly_setup_environments)
    build_version_subfolders(binaries, properly_setup_environments)

    classifiers_info = load_classifier_and_configurations()
    for i, classifier_info in enumerate(classifiers_info):
        console_print(f'    * [{i + 1} of {len(classifiers_info)}] {classifier_info["friendly_name"]}')
        for config in classifier_info["configurations"]:
            console_print(f'        - {config["config_name"]}')
            create_binary_and_test_files(classifiers_info, config, properly_setup_environments)


def main():
    sklearn_versions = get_all_available_sklearn_versions()
    remove_unsupported_versions(sklearn_versions)
    properly_setup_environments, failed_environments = prepare_conda_environment_based_on_sklearn_versions(sklearn_versions)

    iterate_test_cases_and_create_test_files(properly_setup_environments)


if __name__ == '__main__':
    main()
