import platform
from deploy_utils import *

CONDA_ENV_FOR_MAKING_UNIT_TESTS = 'unit_test_sklearn4x_{python_version}_sk_{sklearn_version}'


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
    env_name = CONDA_ENV_FOR_MAKING_UNIT_TESTS\
        .replace('{python_version}', python_version)\
        .replace('{sklearn_version}', scikit_learn_version)\
        .replace('.', '_')

    if env_name not in environments.keys():
        create_conda_environment(env_name, python_version)
        environments[env_name] = get_conda_environments()[env_name]

    environment = environments[env_name]
    environment.install_package('scikit-learn', scikit_learn_version)


def prepare_conda_environment_based_on_sklearn_versions(sklearn_versions):
    result = []
    all_environments = get_conda_environments()

    console_print('Preparing conda environments to generate unit tests.')
    for scikit_learn_version in sklearn_versions.keys():
        console_print('    * scikit-learn ' + scikit_learn_version)
        for i, python_version in enumerate(sklearn_versions[scikit_learn_version]):
            console_print(f'        - [{i + 1} of {len(sklearn_versions[scikit_learn_version])}] Python {python_version}')
            result.append(prepare_conda_environment(all_environments, scikit_learn_version, python_version))

    return result


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



def main():
    sklearn_versions = get_all_available_sklearn_versions()
    remove_unsupported_versions(sklearn_versions)
    conda_environments = prepare_conda_environment_based_on_sklearn_versions(sklearn_versions)


if __name__ == '__main__':
    main()
