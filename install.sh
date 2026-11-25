#!/bin/sh
# Set up this checkout as the user who will run PGTUI.
set -eu

die() { printf '%s\n' "$*" >&2; exit 1; }
usage='Usage: sh install.sh [--check] [--with-keyring]
  Default         Install missing uv/Python/project dependencies and check PGTUI.
  --check         Check only; no downloads, package installs or sudo.
  --with-keyring  Require native keyring discovery. In install mode, get missing
                  GNOME Keyring and D-Bus packages with dnf or apt-get if needed.'
check=no
require_keyring=no
for arg do
    case "$arg" in
        --check) check=yes ;;
        --with-keyring) require_keyring=yes ;;
        --help|-h) printf '%s\n' "$usage"; exit 0 ;;
        *) die "$usage" ;;
    esac
done
[ "$(uname -s)" = Linux ] || die 'This installer is for Linux. On other systems use uv sync --locked.'
CDPATH= cd -- "$(dirname -- "$0")"
export UV_PROJECT_ENVIRONMENT="$PWD/.venv"

# uv honors .python-version and uv.lock; no second list of Python dependencies.
user_bin="$HOME/.local/bin"
uv=$(command -v uv) || uv="$user_bin/uv"
if [ ! -x "$uv" ]; then
    [ "$check" = no ] || die 'uv is missing. Run sh install.sh to install it.'
    command -v curl >/dev/null || die 'Bootstrapping uv requires curl. Install curl or uv, then retry.'
    tmp=$(mktemp -d)
    trap 'rm -rf -- "$tmp"' EXIT
    printf '%s\n' "Installing uv in $user_bin (no shell profile changes)."
    curl --disable --proto '=https' --proto-redir '=https' -fsSL \
        https://astral.sh/uv/0.11.13/install.sh -o "$tmp/uv-installer.sh"
    (cd "$tmp" && printf '%s  uv-installer.sh\n' \
        48cd5aca5d5671a3b3d5f61538cc8622e4434af63319115159990d8b0dd02416 | sha256sum -c -) || die 'uv installer checksum failed; not executing it.'
    UV_INSTALL_DIR="$user_bin" UV_NO_MODIFY_PATH=1 sh "$tmp/uv-installer.sh"
fi

if [ "$check" = yes ]; then
    [ -x .venv/bin/python ] || die 'Project environment is missing. Run sh install.sh first.'
    "$uv" sync --locked --check --offline --no-python-downloads || die 'Environment check failed. Run sh install.sh to sync it.'
else
    "$uv" sync --locked
fi
.venv/bin/pgtui --help >/dev/null || die 'PGTUI failed its startup check (pgtui --help).'
printf '%s\n' 'PGTUI startup check passed. Launch from this checkout with .venv/bin/pgtui.'

probe_keyring() {
    # Same allowlist/discovery as Remember. Never read, write or unlock a secret.
    status=0
    .venv/bin/python - <<'PY' || status=$?
from d8r.credentials import CredentialError, _secure_backend

try:
    backend = _secure_backend()
except CredentialError:
    raise SystemExit(3)
print(f"Native keyring detected: {type(backend).__module__}. Remember requires it to be unlocked.")
PY
    case $status in
        0) return 0 ;;
        3) return 1 ;;
        *) die 'Could not run the native keyring check.' ;;
    esac
}

as_root() {
    if [ "$(id -u)" = 0 ]; then "$@"; else
        command -v sudo >/dev/null || die "Ask an administrator to run: $*"
        sudo -- "$@"
    fi
}

if probe_keyring; then exit 0; fi
printf '%s\n' 'No supported native keyring detected in this user session. Uncheck Remember for session-only credentials.'
if [ "$require_keyring" = no ]; then
    printf '%s\n' 'For persistent credentials, run sh install.sh --with-keyring.'
    exit 0
fi

set --
command -v gnome-keyring-daemon >/dev/null || set -- "$@" gnome-keyring
command -v dbus-run-session >/dev/null || set -- "$@" dbus-daemon
if [ "$#" -gt 0 ]; then
    [ "$check" = no ] || die "Missing packages: $*. Run sh install.sh --with-keyring."
    printf '%s\n' "Installing missing system packages: $*"
    if command -v dnf >/dev/null; then
        as_root dnf install -y "$@"
    elif command -v apt-get >/dev/null; then
        as_root apt-get update
        as_root apt-get install -y --no-install-recommends "$@"
    else
        die "Install these packages with your system package manager: $*"
    fi
    if probe_keyring; then exit 0; fi
fi

die 'Native keyring still unavailable. Start a user D-Bus session (dbus-run-session -- sh if needed), unlock GNOME Keyring there, then run sh install.sh --check --with-keyring and PGTUI in that shell. See the headless Linux setup in README.md.'
