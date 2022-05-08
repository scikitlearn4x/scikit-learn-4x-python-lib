import os
import shutil


def delete_build_folders():
    console_print('Deleting build files & cleaning up source folder.')
    shutil.rmtree('./build')
    shutil.rmtree('./dist')
    shutil.rmtree('./sklearn4x.egg-info')


def create_conda_environments():
    result = []

    return result


def test_library_ing_environment(environment):
    pass


def build_library_for_pypi():
    pass


def ensure_archive_has_source_only_folders():
    pass


def deploy_archives_to_pypi():
    pass


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

def main():
    ensure_script_is_called_from_root()
    delete_build_folders()
    environments = create_conda_environments()

    for environment in environments:
        test_library_ing_environment(environment)

    build_library_for_pypi()
    ensure_archive_has_source_only_folders()
    deploy_archives_to_pypi()

    delete_build_folders()


if __name__ == '__main__':
    main()
