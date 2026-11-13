"""Connect native PostgreSQL, Cloudflare D1 or one local Parquet file.

Test owns its connection until Add, a field edit, or cancellation. Only Add
persists credentials and transfers connection ownership to the workspace.
"""
from __future__ import annotations

import asyncio

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Checkbox, Input, Label, Select, Static

from d8r.credentials import CredentialError
from d8r.engine import DataSource, EngineError
from .connection import CONNECTION_TIMEOUT, ConnectionStatus, build_connection, release_source


class AddSourceModal(ModalScreen):
    """An asynchronous connection form; cancellation never remembers credentials."""

    BINDINGS = [Binding("escape", "cancel", "Cancel")]
    CONNECTION_TIMEOUT = CONNECTION_TIMEOUT

    def __init__(self, session, *, profile: dict | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.session = session
        self._initial_profile = dict(profile) if profile else None
        self._postgres_profiles = list(session.postgres_profiles)
        self._d1_profiles = list(session.d1_profiles)
        self._built: tuple[tuple[str, ...], DataSource] | None = None
        self._worker = None
        self._generation = 0
        self._connection_form_closed = False
        self._busy = False
        self._observed_fields: tuple[str, ...] = ()
        self._reconnect_fields: tuple[str, ...] | None = None
        self._credentials_missing = False

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="add-source"):
            yield Label("Connect saved data source" if self._initial_profile else "Add a data source", id="add-source-title")
            yield Static("Connect PostgreSQL, Cloudflare D1 or one local Parquet file. Remember stores credentials "
                         "in your system keyring; uncheck it for this session only. Test and Cancel save nothing. "
                         "Workspace query text is local plaintext.", id="add-source-note", markup=False)
            yield Label("Source type", classes="field-label")
            backend = {"parquet": "parquet", "d1-live": "d1"}.get(
                (self._initial_profile or {}).get("kind"), "postgres")
            yield Select([("PostgreSQL", "postgres"), ("Local Parquet · DuckDB", "parquet"),
                          ("Cloudflare D1", "d1")], value=backend,
                         allow_blank=False, id="source-backend")
            with Vertical(id="parquet-fields"):
                yield Static("Open one existing local .parquet or .pq file without copying or changing it. "
                             "Add remembers its path and display name. Query it with SELECT * FROM data LIMIT 50.", markup=False)
                yield Label("Parquet file path", classes="field-label")
                yield Input(placeholder="existing local .parquet or .pq path", id="parquet-path")
            with Vertical(id="postgres-fields"):
                yield Static("Queries run natively with your role's permissions, including writes and transactions. "
                             "Use a restricted role when appropriate. For remote servers, verify-full checks the "
                             "TLS certificate and hostname.", id="postgres-setup-help", markup=False)
                yield Label("Saved PostgreSQL connection", classes="field-label")
                yield Select([(f"{profile['display']} · {profile['database']}", index)
                              for index, profile in enumerate(self._postgres_profiles)],
                             prompt="New connection", id="postgres-profile")
                for name, label, default in (("host", "Host", "localhost"), ("port", "Port", "5432"),
                                              ("database", "Database", ""), ("user", "User", ""),
                                              ("password", "Password", ""), ("schema", "Schema", "public")):
                    yield Label(label, classes="field-label")
                    yield Input(default, password=name == "password", type="integer" if name == "port" else "text",
                                id=f"postgres-{name}")
                yield Label("TLS / SSL mode", classes="field-label")
                yield Select([(mode, mode) for mode in ("disable", "allow", "prefer", "require", "verify-ca", "verify-full")],
                             value="prefer", allow_blank=False, id="postgres-sslmode")
            with Vertical(id="d1-fields"):
                yield Static("Cloudflare D1 runs native SQLite with your token's permissions. "
                             "Session transactions, temporary tables and LATERAL are not supported. "
                             "Cancellation cannot guarantee remote rollback.", markup=False)
                yield Label("Saved Cloudflare D1 connection", classes="field-label")
                yield Select([(f"{profile['display']} · {profile['database']}", index)
                              for index, profile in enumerate(self._d1_profiles)],
                             prompt="New connection", id="d1-profile")
                yield Label("Cloudflare account ID", classes="field-label")
                yield Input(placeholder="32-character account ID", id="d1-account")
                yield Label("Database name or UUID", classes="field-label")
                yield Input(placeholder="database name or UUID", id="d1-database")
                yield Label("API token", classes="field-label")
                yield Input(password=True, id="d1-token")
            yield Label("Display name", classes="field-label")
            yield Input(placeholder="optional source label", id="display-name")
            yield Checkbox("Remember credentials in system keyring", value=True, id="remember-credentials")
            yield Static(self.session.credential_warning, markup=False)
            yield Static("", id="add-source-message", markup=False)
            with Horizontal(id="add-source-buttons"):
                yield Button("Test connection", id="test")
                yield Button("Connect" if self._initial_profile else "Add", variant="primary", id="add")
                yield Button("Cancel", id="cancel")
        progress = ConnectionStatus(id="source-connection-progress")
        progress.display = False
        yield progress

    def on_mount(self) -> None:
        if self._initial_profile:
            with self.prevent(Input.Changed, Select.Changed):
                if self._initial_profile.get("kind") == "parquet":
                    self.query_one("#parquet-path", Input).value = self._initial_profile["path"]
                    self.query_one("#display-name", Input).value = self._initial_profile.get("display", "")
                elif self._initial_profile.get("kind") == "d1-live":
                    self._fill_d1_profile(self._initial_profile)
                else:
                    self._fill_postgres_profile(self._initial_profile)
            self._reconnect_fields = None if self._credentials_missing else self._fields()
        self._show_backend()
        self._observed_fields = self._fields()
        if self._credentials_missing:
            d1 = self.query_one("#source-backend", Select).value == "d1"
            self._message("Saved credentials are unavailable. Re-enter the token to connect." if d1 else
                          "Saved credentials are unavailable. Re-enter the password to connect.", error=True)
            self.query_one("#d1-token" if d1 else "#postgres-password", Input).focus()

    def _show_backend(self) -> None:
        backend = self.query_one("#source-backend", Select).value
        for name in ("parquet", "postgres", "d1"):
            self.query_one(f"#{name}-fields").display = backend == name
        self.query_one("#remember-credentials").display = backend != "parquet"
        field = {"parquet": "parquet-path", "postgres": "postgres-host", "d1": "d1-account"}[backend]
        self.query_one(f"#{field}", Input).focus()

    def _fields(self) -> tuple[str, ...]:
        # Keep PostgreSQL indexes (including Remember at 10) unchanged.
        fields = (str(self.query_one("#source-backend", Select).value),
                self.query_one("#display-name", Input).value.strip(),
                self.query_one("#parquet-path", Input).value.strip(),
                *(self.query_one(f"#postgres-{name}", Input).value.strip() for name in ("host", "port", "database", "user")),
                self.query_one("#postgres-password", Input).value,
                self.query_one("#postgres-schema", Input).value.strip(),
                str(self.query_one("#postgres-sslmode", Select).value),
                "remember" if self.query_one("#remember-credentials", Checkbox).value else "session")
        if fields[0] == "d1":
            return (fields[0], self.query_one("#display-name", Input).value, *fields[2:],
                    self.query_one("#d1-account", Input).value,
                    self.query_one("#d1-database", Input).value,
                    self.query_one("#d1-token", Input).value)
        return fields

    @staticmethod
    def _postgres_profile(fields: tuple[str, ...]) -> dict:
        return dict(zip(("host", "port", "database", "user", "password", "schema", "sslmode"), fields[3:10]))

    def _fill_postgres_profile(self, profile: dict) -> None:
        self._credentials_missing = bool(profile.get("credentials_missing"))
        defaults = {"host": "localhost", "port": "5432", "schema": "public"}
        for name in ("host", "port", "database", "user", "password", "schema"):
            self.query_one(f"#postgres-{name}", Input).value = str(profile.get(name, defaults.get(name, "")))
        self.query_one("#postgres-sslmode", Select).value = profile.get("sslmode", "prefer")
        self.query_one("#display-name", Input).value = profile.get("display", "")
        self._reconnect_fields = self._fields() if profile and not self._credentials_missing else None
        self._observed_fields = self._fields()

    @staticmethod
    def _d1_profile(fields: tuple[str, ...]) -> dict:
        return {"account_id": fields[11].strip(), "database": fields[12].strip(), "api_token": fields[13]}

    def _fill_d1_profile(self, profile: dict) -> None:
        self._credentials_missing = bool(profile.get("credentials_missing"))
        for field, key in (("account", "account_id"), ("database", "database"), ("token", "api_token")):
            self.query_one(f"#d1-{field}", Input).value = profile.get(key, "")
        self.query_one("#display-name", Input).value = profile.get("display", "")
        self._reconnect_fields = self._fields() if profile and not self._credentials_missing else None
        self._observed_fields = self._fields()

    def _message(self, text: str, error: bool = False) -> None:
        for field in ("postgres-password", "d1-token"):
            secret = self.query_one(f"#{field}", Input).value
            if secret:
                text = text.replace(secret, "[redacted]")
        widget = self.query_one("#add-source-message", Static)
        widget.update(text)
        widget.set_class(error, "error")

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        for name in ("test", "add"):
            self.query_one(f"#{name}", Button).disabled = busy
        self.query_one("#add-source").display = not busy
        progress = self.query_one(ConnectionStatus)
        progress.display = busy
        if busy:
            fields = self._fields()
            target = fields[2] if fields[0] == "parquet" else fields[12] if fields[0] == "d1" else fields[5]
            for secret in (fields[7], fields[13] if fields[0] == "d1" else ""):
                if secret:
                    target = target.replace(secret, "[redacted]")
            progress.start(target or "data source")
            progress.query_one(Button).focus()
        else:
            self.query_one("#add", Button).focus()

    def _invalidate(self) -> None:
        self._generation += 1
        if self._worker is not None:
            self._worker.cancel()
            self._worker = None
        if self._built is not None:
            release_source(self._built[1])
            self._built = None
        if not self._connection_form_closed:
            self._set_busy(False)

    @on(Input.Changed)
    @on(Checkbox.Changed, "#remember-credentials")
    @on(Select.Changed, "#postgres-sslmode")
    def _fields_changed(self) -> None:
        if self._connection_form_closed:
            return
        fields = self._fields()
        if fields != self._observed_fields:
            self._observed_fields = fields
            self._invalidate()
            self._message("Fields changed. Test or Add to connect with these values.")

    @on(Select.Changed, "#source-backend")
    def _backend_selected(self) -> None:
        if not self._connection_form_closed:
            self._show_backend()
            self._fields_changed()

    @on(Select.Changed, "#postgres-profile")
    def _postgres_profile_selected(self, event: Select.Changed) -> None:
        if self._connection_form_closed:
            return
        self._invalidate()
        profile = self._postgres_profiles[event.value] if isinstance(event.value, int) else {}
        with self.prevent(Select.Changed, Input.Changed):
            self._fill_postgres_profile(profile)
        self._message("Saved credentials unavailable; re-enter the password." if self._credentials_missing else
                      "Saved connection loaded. Test or Add to connect." if profile else "")

    @on(Select.Changed, "#d1-profile")
    def _d1_profile_selected(self, event: Select.Changed) -> None:
        if self._connection_form_closed:
            return
        self._invalidate()
        profile = self._d1_profiles[event.value] if isinstance(event.value, int) else {}
        with self.prevent(Select.Changed, Input.Changed):
            self._fill_d1_profile(profile)
        self._message("Saved credentials unavailable; re-enter the token." if self._credentials_missing else
                      "Saved connection loaded. Test or Add to connect." if profile else "")

    def _build(self, fields: tuple[str, ...]) -> DataSource:
        if fields[0] == "parquet":
            return self.session.build_parquet_source(fields[2], fields[1])
        if fields[0] == "d1":
            profile = self._d1_profile(fields)
            if self._credentials_missing and not profile["api_token"]:
                raise CredentialError("Re-enter the missing API token before connecting.")
            return self.session.build_d1_source(**profile, display=fields[1])
        profile = self._postgres_profile(fields)
        if self._credentials_missing and not profile["password"]:
            raise CredentialError("Re-enter the missing password before connecting.")
        if not all(profile[key] for key in ("host", "port", "database", "user", "schema")):
            raise ValueError("Host, port, database, user and schema are required.")
        port = int(profile.pop("port"))
        if not 1 <= port <= 65535:
            raise ValueError("Port must be between 1 and 65535.")
        return self.session.build_postgres_source(**profile, port=port, display=fields[1] or None)

    def _begin(self, add: bool) -> None:
        if self._connection_form_closed or self._busy:
            return
        fields = self._fields()
        if fields != self._observed_fields:
            self._invalidate()
            self._observed_fields = fields
        self._set_busy(True)
        self._message("Connecting… Cancel to return without saving.")
        self._worker = self.run_worker(self._connect(fields, self._generation, add),
                                       name="Source connection", group="source-connect", exit_on_error=False)

    async def _connect(self, fields: tuple[str, ...], generation: int, add: bool) -> None:
        def current() -> bool:
            return not self._connection_form_closed and generation == self._generation and fields == self._fields()
        try:
            if self._built is not None and self._built[0] == fields:
                source = self._built[1]
            else:
                source = await build_connection(lambda: self._build(fields), timeout=self.CONNECTION_TIMEOUT)
                if not current():
                    release_source(source)
                    return
                self._built = (fields, source)
            if add:
                if source.kind == "postgres-live" and fields != self._reconnect_fields:
                    profile = self._postgres_profile(fields)
                    profile["display"] = source.display
                    self.session.remember_postgres(profile, persist=fields[10] == "remember")
                if source.kind == "d1-live" and fields != self._reconnect_fields:
                    profile = self._d1_profile(fields)
                    profile.update(source.d1)
                    profile["display"] = source.display
                    self.session.remember_d1(profile, persist=fields[10] == "remember")
                if source.kind in {"postgres-live", "d1-live"}:
                    self.app.notify("Connected using saved credentials." if fields == self._reconnect_fields else
                                    "Credentials saved in system keyring." if fields[10] == "remember" else
                                    "Connected for this session only; credentials were not saved.", title="Connection credentials")
                self._built = None
                self._connection_form_closed = True
                self.dismiss(source)
            else:
                self._message(f"{source.display} · {source.server_version} · {len(source.datasets)} tables · "
                              + (", ".join(source.datasets) or "no user tables"))
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            if current():
                self._message(f"Connection timed out after {self.CONNECTION_TIMEOUT:g}s. Retry to connect.", error=True)
        except CredentialError as exc:
            if current():
                self._message(f"{exc} Unlock your keyring or uncheck Remember for this session only.", error=True)
        except (EngineError, OSError, RuntimeError, ValueError) as exc:
            if current():
                self._message(str(exc) if fields[0] == "parquet" else self._connection_error(fields), error=True)
        except Exception:
            if current():
                self._message(self._connection_error(fields), error=True)
        finally:
            if generation == self._generation:
                self._worker = None
                if not self._connection_form_closed:
                    self._set_busy(False)

    @staticmethod
    def _connection_error(fields: tuple[str, ...]) -> str:
        if fields[0] == "parquet":
            return "Could not open Parquet. Check the local file path, format and read permissions."
        if fields[0] == "d1":
            return "Could not connect to Cloudflare D1 or save credentials. Check account ID, database name or UUID, API token and access permissions."
        return "Could not connect to PostgreSQL or save credentials. Check host, port, database, user, password, schema, TLS mode and access permissions."

    def action_cancel(self) -> None:
        self._connection_form_closed = True
        self._invalidate()
        self.query_one("#postgres-password", Input).value = ""
        self.query_one("#d1-token", Input).value = ""
        self.dismiss(None)

    def on_unmount(self) -> None:
        self._connection_form_closed = True
        self._invalidate()

    @on(Button.Pressed, ".connection-cancel")
    def _cancel_connection(self, event: Button.Pressed) -> None:
        event.stop()
        self.action_cancel()

    def action_test(self) -> None:
        self._begin(add=False)

    def action_add(self) -> None:
        self._begin(add=True)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "test":
            self.action_test()
        elif event.button.id == "add":
            self.action_add()
        elif event.button.id == "cancel":
            self.action_cancel()
