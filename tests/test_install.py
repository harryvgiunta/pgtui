"""Exercise the Linux installer without network, system packages or a user's vault."""
from pathlib import Path
import shutil
import subprocess

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "install.sh"
SH = shutil.which("sh")
pytestmark = pytest.mark.skipif(not SH, reason="Linux installer needs a POSIX shell")


def command(directory, name, body):
    path = directory / name
    path.write_bytes(("#!/bin/sh\nset -eu\n" + body + "\n").encode())
    path.chmod(0o755)


@pytest.fixture
def install(tmp_path):
    # Only these test commands are on PATH. Even an accidental sudo, curl or
    # package-manager invocation cannot touch the host. Invoke from outside the repo.
    tools, home, checkout = (tmp_path / name for name in ("tools", "home", "source checkout"))
    for directory in (tools, home, checkout):
        directory.mkdir()
    shutil.copyfile(SCRIPT, checkout / "install.sh")
    command(tools, "uname", "echo Linux")
    command(tools, "dirname", 'exec /usr/bin/dirname "$@"')
    command(tools, "id", "echo 1000")
    command(tools, "sudo", 'echo "$*" >> "$HOME/elevations"; shift; exec "$@"')
    # A stand-in for the locked environment uv would install, not a live TUI.
    bin_dir = home / "packages" / "bin"
    bin_dir.mkdir(parents=True)
    command(bin_dir, "pgtui", '[ "$1" = --help ] && [ ! -f "$HOME/broken-app" ]')
    command(bin_dir, "python", '[ -f "$HOME/native-backend" ] || exit 3')
    command(tools, "uv", '''case " $* " in
        *" --check "*) [ -x .venv/bin/python ] && [ ! -f "$HOME/stale" ] ;;
        *) [ -d .venv ] || /bin/cp -R "$HOME/packages" .venv ;;
    esac''')

    def run(*args):
        return subprocess.run([SH, "-c", 'export HOME="$PWD/home" PATH="$PWD/tools"; '
                               'exec /bin/sh "source checkout/install.sh" "$@"', "sh", *args],
                              cwd=tmp_path, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=15)

    return run


def test_check_does_not_bootstrap_or_repair_environment(tmp_path, install):
    env = tmp_path / "source checkout" / ".venv"
    assert install("--check").returncode != 0
    assert not env.exists()
    setup = install()
    assert setup.returncode == 0, setup.stdout  # no native backend required
    assert env.is_dir()
    assert install("--check").returncode == 0
    (tmp_path / "home" / "stale").touch()
    assert install("--check").returncode != 0
    (tmp_path / "tools" / "uv").unlink()
    command(tmp_path / "tools", "curl", ': > "$HOME/downloaded"')
    assert install("--check").returncode != 0
    assert not (tmp_path / "home" / "downloaded").exists()
    assert not (tmp_path / "home" / "elevations").exists()


@pytest.mark.parametrize("manager", ["dnf", "apt-get"])
def test_requested_packages_do_not_substitute_for_user_session(tmp_path, install, manager):
    tools, home = tmp_path / "tools", tmp_path / "home"
    command(tools, "dbus-run-session", "exit 1")  # installed; must not be started or reinstalled
    command(tools, manager, '''for arg do
        case "$arg" in
            gnome-keyring) /bin/cp "$0" "$HOME/../tools/gnome-keyring-daemon" ;;
            dbus-daemon) exit 2 ;;
        esac
    done''')
    assert install().returncode == 0
    assert install("--check", "--with-keyring").returncode != 0
    assert not (home / "elevations").exists()
    result = install("--with-keyring")
    assert result.returncode != 0 and "unlock" in result.stdout
    assert (tools / "gnome-keyring-daemon").exists()
    elevations = (home / "elevations").read_bytes()
    assert install("--with-keyring").returncode != 0
    assert (home / "elevations").read_bytes() == elevations
    (home / "native-backend").touch()
    (tools / "gnome-keyring-daemon").unlink()  # another native backend also suffices
    assert install("--with-keyring").returncode == 0
    assert (home / "elevations").read_bytes() == elevations


def test_required_startup_failure_is_not_an_optional_keyring_warning(tmp_path, install):
    (tmp_path / "home" / "broken-app").touch()
    assert install("--with-keyring").returncode != 0
    assert not (tmp_path / "home" / "elevations").exists()


def test_unverified_bootstrap_is_not_executed(tmp_path, install):
    tools = tmp_path / "tools"
    (tools / "uv").unlink()
    for utility in ("mktemp", "rm", "sha256sum", "sh"):
        command(tools, utility, f'exec /usr/bin/{utility} "$@"')
    command(tools, "curl", '''for output do :; done
        printf '%s\\n' ': > "$HOME/executed"' > "$output"''')
    result = install()
    assert result.returncode != 0 and "checksum failed" in result.stdout
    assert not (tmp_path / "home" / "executed").exists()
