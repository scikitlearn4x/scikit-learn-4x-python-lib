import platform
import sys

from deploy_utils import *

CONDA_ENV_FOR_MAKING_UNIT_TESTS = 'unit_test_sklearn4x_{python_version}_sk_{sklearn_version}'


def load_classifier_and_configurations():
    result = []

    result.append(add_gaussian_nb())

    return result


def add_gaussian_nb():
    config = get_classifier_config(friendly_name='gaussian naive bayes',
                                   class_name='GaussianNB',
                                   namespace='sklearn.naive_bayes')

    add_classifier_configuration(config=config, name='')

    return config


def add_classifier_configuration(config, name, **parameters):
    config['configurations'] = parameters


def get_classifier_config(friendly_name, class_name, namespace):
    return {
        'friendly_name': friendly_name,
        'class_name': class_name,
        'namespace': namespace,
        'configurations': {}
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

        to_remove = to_remove[1:]
        for version in to_remove:
            sklearn_versions.pop(version)


def get_argument(arg):
    args = sys.argv

    for i, value in enumerate(args):
        if value == arg:
            return args[i + 1]

    raise Exception(f'Argument {arg} should be provided.')


def check_folder_empty(path):
    if len(os.listdir(path)) > 0:
        raise Exception('Output folder should be empty.')


def main():
    sklearn_versions = get_all_available_sklearn_versions()
    remove_unsupported_versions(sklearn_versions)
    properly_setup_environments, failed_environments = prepare_conda_environment_based_on_sklearn_versions(sklearn_versions)

    path = get_argument('--save-path')
    console_print('Start creating test cases.')
    console_print(f'    * Path: {path}')
    check_folder_empty(path)

    classifiers_info = load_classifier_and_configurations()
    for i, classifier_info in enumerate(classifiers_info):
        console_print(f'    * [{i + 1} of {len(classifiers_info)}] ')


if __name__ == '__main__':
    main()
