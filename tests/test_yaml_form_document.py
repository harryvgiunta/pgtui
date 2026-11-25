"""Source-preserving dbt YAML form edits, independent of Textual or the filesystem."""
from dataclasses import FrozenInstanceError

import pytest
import yaml

from d8r.yaml_form import YamlFormDocument, YamlFormError


def test_noop_and_nested_rename_do_not_reformat_dbt_sources():
    source = (
        "---\r\n# Source config\r\nversion: 02  # keep spelling\r\n"
        "sources:\r\n  - name: 'raw'\r\n    # database note\r\n"
        "    database: \"{{ env_var('RAW_DB') }}\" # trusted\r\n"
        "    tables: [{name: orders}, {name: customers}]\r\n..."
    )
    doc = YamlFormDocument(source)
    entry = ("sources", 0)
    assert doc.children(entry) == (entry + ("name",), entry + ("database",), entry + ("tables",))
    assert doc.kind(("version",)) == "integer"
    assert doc.change("edit", ("version",), kind="integer", value="2") == source
    assert doc.change("edit", entry, kind="mapping") == source
    path = entry + ("database",)
    result = doc.change("edit", path, key="catalog", value=doc.value(path))
    assert result == source.replace("    database:", "    catalog:")
    assert doc.text == source and doc.kind(path) == "text"
    with pytest.raises(FrozenInstanceError):
        doc.text = result


def test_comment_only_document_can_be_built_without_losing_its_header():
    source = "# Models with tests\r\n# Still a draft"
    doc = YamlFormDocument(source)
    assert doc.kind(()) == "mapping" and not doc.children(())
    assert doc.change("edit", (), kind="mapping") == source
    added = doc.change("add", (), key="models", kind="list")
    assert added == source + "\r\nmodels: []"
    added = YamlFormDocument(added).change("add", ("models",), kind="mapping")
    added = YamlFormDocument(added).change("add", ("models", 0), key="name", value="orders")
    assert yaml.safe_load(added) == {"models": [{"name": "orders"}]}
    assert added.startswith(source) and not added.endswith("\n")
    assert YamlFormDocument("").change("edit", (), kind="list") == "[]"


@pytest.mark.parametrize("value", ["", "true", "NULL", "12", "2026-05-01", "1e4", "{a: b}", "[a, b]",
                                   "{{ ref('orders') }}", "# important", "  leading\n # data\n\n", "\U0001f681\u2028\\n"])
def test_typed_text_is_never_implicitly_parsed_or_rendered(value):
    source = "config: {description: off, tags: ['off']} # config\n"
    result = YamlFormDocument(source).change("edit", ("config", "description"), value=value)
    assert yaml.safe_load(result)["config"] == {"description": value, "tags": ["off"]}
    assert result.endswith(", tags: ['off']} # config\n")


@pytest.mark.parametrize(("kind", "value", "expected"), [("integer", "-0002", -2), ("number", "1e30", 1e30),
                                                         ("number", "-0", -0.0), ("boolean", "TRUE", True),
                                                         ("null", "", None)])
def test_scalar_kinds_on_root(kind, value, expected):
    changed = YamlFormDocument("'seed' # note").change("edit", (), kind=kind, value=value)
    doc = YamlFormDocument(changed)
    assert doc.kind(()) == kind
    assert yaml.safe_load(changed) == expected and changed.endswith(" # note")
    assert doc.change("edit", (), kind=kind, value=doc.value(())) == changed


def test_literal_header_comments_are_not_scalar_data():
    source = "description: |+ # docs\n  prior line\n  # part of the description\n\n# config note\nconfig: {materialized: view}\n"
    result = YamlFormDocument(source).change("edit", ("description",), value="{{ doc('order') }}\nnew\n")
    assert yaml.safe_load(result)["description"] == "{{ doc('order') }}\nnew\n"
    assert result.count("# docs") == 1 and "# part of the description" not in result
    assert result.endswith("# config note\nconfig: {materialized: view}\n")


def test_add_remove_around_compact_maps_and_neighbor_comments():
    source = ("sources:\n- name: raw  # source name\n  tables:\n  - name: old\n    description: previous\n"
              "  # important table\n  - name: next # next table\n# other resources\nmodels: []\n")
    path = ("sources", 0, "tables")
    doc = YamlFormDocument(source)
    added = doc.change("add", path, kind="mapping")
    assert added == source.replace("# other resources", "  - {}\n# other resources")
    removed = doc.change("remove", path + (0,))
    assert removed == source.replace("  - name: old\n    description: previous\n", "")
    renamed = doc.change("edit", path, key="resources", kind="list")
    assert renamed == source.replace("  tables:", "  resources:")


def test_remove_subtree_preserves_comments_but_not_quoted_hashes():
    source = "a:\n  # old section\n  b: '# data' # note\n  # title\n  c: 2\n# keep below\nother: tail\n"
    changed = YamlFormDocument(source).change("remove", ("a",))
    assert yaml.safe_load(changed) == {"other": "tail"}
    assert [line.lstrip() for line in changed.splitlines() if "#" in line] == ["# old section", "# note", "# title", "# keep below"]
    assert changed.endswith("# keep below\nother: tail\n")


def test_indentless_singleton_and_flow_trailing_comma():
    source = "tags:\r- old # tag note\runrelated: [one, two,] # tags\r"
    changed = YamlFormDocument(source).change("remove", ("tags", 0))
    assert yaml.safe_load(changed) == {"tags": [], "unrelated": ["one", "two"]}
    assert "# tag note\r" in changed and "\n" not in changed
    changed = YamlFormDocument(changed).change("add", ("unrelated",), value="off")
    changed = YamlFormDocument(changed).change("remove", ("unrelated", 1))
    assert yaml.safe_load(changed) == {"tags": [], "unrelated": ["one", "off"]}
    assert changed.endswith(" # tags\r")


def test_final_row_remove_does_not_chomp_neighbor_literal():
    source = "notes: |\n  keep line break\nobsolete: 123"
    assert YamlFormDocument(source).change("remove", ("obsolete",)) == "notes: |\n  keep line break\n"


def test_editing_implicit_nulls_and_quoted_keys():
    doc = YamlFormDocument("params:\n  '0': # implicit null\n  'true': enabled\n")
    assert doc.kind(("params", "0")) == "null"
    changed = doc.change("edit", ("params", "0"), key="null", value="null")
    assert yaml.safe_load(changed) == {"params": {"null": "null", "true": "enabled"}}
    assert changed.endswith(" # implicit null\n  'true': enabled\n")


@pytest.mark.parametrize("source", ["a: [", "x: old\nx: new", "x: {a: yes, 'a': no}", "a: &same [*same]",
                                    "b: *undefined", "a: !!str tag", "foo: !unknown bar", "a: {<<: {k: v}}",
                                    "1: text", "? [a, b]\n: c", "a: 2026-01-01", "a: .inf", "a: .nan",
                                    "---\na: 1\n---\na: 2", "[" * 100 + "1" + "]" * 100,
                                    pytest.param("- x\n" * 20_001, id="node-limit"),
                                    pytest.param("#" * (512 * 1024 + 1), id="text-limit")])
def test_unsupported_or_oversize_documents_refer_to_raw(source):
    with pytest.raises(YamlFormError, match="RAW"):
        YamlFormDocument(source)


def test_validation_is_transactional_and_does_not_leak_input_values():
    source = "name: secret\nvalues: [0, 1]\n"
    doc = YamlFormDocument(source)
    invalid = [("edit", ("name",), {"key": "values"}), ("add", (), {"key": "name"}),
               ("remove", (), {}), ("add", ("values",), {"key": "child"}), ("add", (), {}),
               ("edit", ("values",), {"kind": "integer", "value": "0"}), ("remove", ("values", True), {}),
               ("edit", ("values", -1), {}), ("edit", ("name",), {"kind": "integer", "value": "1.5"}),
               ("edit", ("name",), {"kind": "number", "value": "1e1000"}),
               ("edit", ("name",), {"kind": "boolean", "value": "not a boolean"})]
    for op, path, args in invalid:
        with pytest.raises(YamlFormError) as error:
            doc.change(op, path, **args)
        assert "secret" not in str(error.value)
    assert doc.text == source and doc.value(("name",)) == "secret"


def test_append_nested_named_rows_in_block_preserves_neighbors_and_actual_paths():
    source = ("# authored\r\nsources:\r\n- name: raw\r\n  tables:\r\n  - name: 'orders' # old table\r\n"
              "# other resources\r\nversion: 02")
    doc = YamlFormDocument(source)
    path = ("sources", 0, "tables")
    added, table = doc.append_named_item(path, "returns: US")
    assert table == path + (1,)
    row = '  - name: "returns: US"\r\n'
    assert added == source.replace("# other resources", row + "# other resources")
    with_table = YamlFormDocument(added)
    finished, column = with_table.append_named_item(table + ("columns",), "id")
    assert column == table + ("columns", 0)
    assert finished == added.replace(row, row + "    columns:\r\n      - name: id\r\n")
    assert yaml.safe_load(finished)["sources"][0]["tables"][1] == {"name": "returns: US", "columns": [{"name": "id"}]}
    assert doc.text == source and with_table.children(table) == (table + ("name",),)


@pytest.mark.parametrize("name", ["true", "NULL", "12", "2001-01-01", "  spaces  ", "{x: y}", "{{ var('x') }}"])
def test_flow_list_append_quotes_literal_names_next_to_trailing_comma_and_comment(name):
    source = "outer: {rows: [{name: prior}, # history\n    ]} # outer\n"
    changed, item = YamlFormDocument(source).append_named_item(("outer", "rows"), name)
    assert item == ("outer", "rows", 1)
    assert changed.startswith(source[:source.index("]")]) and changed.endswith("]} # outer\n")
    assert yaml.safe_load(changed)["outer"]["rows"] == [{"name": "prior"}, {"name": name}]


@pytest.mark.parametrize("old", ["", " ~", " null", " ''", ' ""'])
def test_first_named_row_replaces_only_empty_block_field(old):
    source = "group:\n  rows:" + old + " # collection\n  other: 01\n"
    changed, item = YamlFormDocument(source).append_named_item(("group", "rows"), "off")
    assert changed == 'group:\n  rows: # collection\n    - name: "off"\n  other: 01\n'
    assert item == ("group", "rows", 0)


@pytest.mark.parametrize("fields", ["note: kept", "note: kept, rows: null", "note: kept, rows: ''", "note: kept, rows: []"])
def test_first_named_row_in_flow_mapping_stays_flow(fields):
    source = "- {" + fields + "} # record"
    changed, item = YamlFormDocument(source).append_named_item((0, "rows"), "new")
    assert changed == "- {note: kept, rows: [{name: new}]} # record"
    assert item == (0, "rows", 0)


def test_first_named_row_in_virtual_root():
    changed, item = YamlFormDocument("# title").append_named_item(("true",), "seed")
    assert changed == '# title\n"true":\n  - name: seed'
    assert item == ("true", 0)


def test_only_literal_names_in_target_list_collide():
    doc = YamlFormDocument("a: [legacy, {name: 'NULL'}, {name: true}]\nb: [{name: elsewhere}]\n")
    for duplicate in ("legacy", "NULL"):
        with pytest.raises(YamlFormError, match="already"):
            doc.append_named_item(("a",), duplicate)
    for distinct in ("true", "null", "elsewhere"):
        changed, item = doc.append_named_item(("a",), distinct)
        assert item == ("a", 3) and yaml.safe_load(changed)["a"][-1] == {"name": distinct}
    assert len(doc.children(("a",))) == 3


@pytest.mark.parametrize("name", ["", " \t ", "two\nlines", "last\r", "\u0085first", "bad\u2028line", None])
def test_invalid_named_add_does_not_create_the_missing_field(name):
    doc = YamlFormDocument("{}")
    with pytest.raises(YamlFormError):
        doc.append_named_item(("rows",), name)
    assert doc.text == "{}" and not doc.children(())


@pytest.mark.parametrize("path", [("text",), ("number",), ("bool",), ("map",), ("rows", 0), ("rows", 9),
                                  ("missing", "deep"), ("text", "deep"), ("rows", False), ()])
def test_named_add_does_not_clobber_or_invent_parents(path):
    source = "text: ' '\nnumber: 0\nbool: false\nmap: {keep: yes}\nrows: [null]\n"
    doc = YamlFormDocument(source)
    with pytest.raises(YamlFormError):
        doc.append_named_item(path, "first")
    assert doc.text == source and doc.kind(("rows", 0)) == "null"


def test_named_append_rejects_patch_that_would_change_prior_literal_at_eof():
    source = "- name: prior\n  description: |+\n    no final break"
    doc = YamlFormDocument(source)
    with pytest.raises(YamlFormError, match="RAW"):
        doc.append_named_item((), "next")
    assert doc.text == source and doc.value((0, "description")) == "no final break"


def test_promote_only_selected_legacy_row_then_add_children():
    source = "tables:\r\n- keep\r\n- orders # import\r\nnext: null"
    old = YamlFormDocument(source)
    table = ("tables", 1)
    doc = old.as_named_mapping(table)
    assert doc.text == source.replace("- orders", "- name: orders")
    assert old.value(table) == "orders" and doc.as_named_mapping(table) is doc
    changed, column = doc.append_named_item(table + ("columns",), "id")
    assert column == table + ("columns", 0)
    assert yaml.safe_load(changed)["tables"] == ["keep", {"name": "orders", "columns": [{"name": "id"}]}]


@pytest.mark.parametrize("source", ["['off', untouched] # docs", "- |+ # docs\n  text\n  # literal\n\n- untouched\n"])
def test_promote_retains_exact_scalar_value_and_comments(source):
    doc = YamlFormDocument(source)
    promoted = doc.as_named_mapping((0,))
    assert yaml.safe_load(promoted.text) == [{"name": doc.value((0,))}, "untouched"]
    assert promoted.text.count("# docs") == 1
    if source.startswith("["):
        assert promoted.text == '[{name: "off"}, untouched] # docs'


@pytest.mark.parametrize(("source", "path"), [("value", ()), ("value: text", ("value",)), ("[3]", (0,)),
                                               ("[null]", (0,)), ("[true]", (0,)), ("[[]]", (0,)), ("[]", (0,))])
def test_promotion_requires_mapping_or_existing_text_list_item(source, path):
    doc = YamlFormDocument(source)
    with pytest.raises(YamlFormError):
        doc.as_named_mapping(path)
    assert doc.text == source
